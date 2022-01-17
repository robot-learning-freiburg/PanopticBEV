from collections import OrderedDict
import torch
import torch.nn as nn
from panoptic_bev.utils.sequence import pad_packed_images

NETWORK_INPUTS = ["img", "bev_msk", "front_msk", "weights_msk", "cat", "iscrowd", "bbx", "calib"]

class PanopticBevNet(nn.Module):
    def __init__(self,
                 body,
                 transformer,
                 rpn_head,
                 roi_head,
                 sem_head,
                 transformer_algo,
                 rpn_algo,
                 inst_algo,
                 sem_algo,
                 po_fusion_algo,
                 dataset,
                 classes=None,
                 front_vertical_classes=None,  # In the frontal view
                 front_flat_classes=None,  # In the frontal view
                 bev_vertical_classes=None,  # In the BEV
                 bev_flat_classes=None):  # In the BEV
        super(PanopticBevNet, self).__init__()

        # Backbone
        self.body = body

        # Transformer
        self.transformer = transformer

        # Modules
        self.rpn_head = rpn_head
        self.roi_head = roi_head
        self.sem_head = sem_head

        # Algorithms
        self.transformer_algo = transformer_algo
        self.rpn_algo = rpn_algo
        self.inst_algo = inst_algo
        self.sem_algo = sem_algo
        self.po_fusion_algo = po_fusion_algo

        # Params
        self.dataset = dataset
        self.num_stuff = classes["stuff"]
        self.front_vertical_classes = front_vertical_classes
        self.front_flat_classes = front_flat_classes
        self.bev_vertical_classes = bev_vertical_classes
        self.bev_flat_classes = bev_flat_classes

    def make_region_mask(self, msk):
        if (self.bev_vertical_classes is None) or (self.bev_flat_classes is None):
            return

        B = len(msk)
        W, Z = msk[0].shape[0], msk[0].shape[1]
        v_region_msk = torch.zeros((B, 1, W, Z), dtype=torch.long).to(msk[0].device)
        f_region_msk = torch.zeros((B, 1, W, Z), dtype=torch.long).to(msk[0].device)

        for b in range(B):
            for c in self.bev_vertical_classes:
                v_region_msk[b, 0, msk[b] == int(c)] = 1
            for c in self.bev_flat_classes:
                f_region_msk[b, 0, msk[b] == int(c)] = 1

        return v_region_msk, f_region_msk

    def make_vf_mask(self, msk):
        if (self.front_vertical_classes is None) or (self.front_flat_classes is None):
            return

        B = msk.shape[0]
        H, W = msk.shape[2], msk.shape[3]
        vf_msk = torch.ones((B, 1, H, W), dtype=torch.long).to(msk.device) * 2  # Everything is initially unknown

        sem_msk = msk.detach().clone()
        sem_msk[sem_msk >= 1000] = sem_msk[sem_msk >= 1000] // 1000

        for c in self.front_vertical_classes:
            vf_msk[sem_msk == int(c)] = 0
        for c in self.front_flat_classes:
            vf_msk[sem_msk == int(c)] = 1

        return vf_msk

    def prepare_inputs(self, msk, cat, iscrowd, bbx):
        cat_out, iscrowd_out, bbx_out, ids_out, sem_out, po_out, po_vis_out = [], [], [], [], [], [], []
        for msk_i, cat_i, iscrowd_i, bbx_i in zip(msk, cat, iscrowd, bbx):
            msk_i = msk_i.squeeze(0)
            thing = (cat_i >= self.num_stuff) & (cat_i != 255)
            valid = thing & ~(iscrowd_i > 0)

            if valid.any().item():
                cat_out.append(cat_i[valid])
                bbx_out.append(bbx_i[valid])
                ids_out.append(torch.nonzero(valid))
            else:
                cat_out.append(None)
                bbx_out.append(None)
                ids_out.append(None)

            if iscrowd_i.any().item():
                iscrowd_i = (iscrowd_i > 0) & thing
                iscrowd_out.append(iscrowd_i[msk_i].type(torch.uint8))
            else:
                iscrowd_out.append(None)

            sem_out.append(cat_i[msk_i])

            # Panoptic GT
            po_msk = torch.ones_like(msk_i) * 255
            po_msk_vis = torch.ones_like(msk_i) * 255
            inst_id = 0
            for lbl_idx in range(cat_i.shape[0]):
                if cat_i[lbl_idx] == 255:
                    continue
                if cat_i[lbl_idx] < self.num_stuff:
                    po_msk[msk_i == lbl_idx] = cat_i[lbl_idx]
                    po_msk_vis[msk_i == lbl_idx] = cat_i[lbl_idx]
                else:
                    po_msk[msk_i == lbl_idx] = self.num_stuff + inst_id
                    po_msk_vis[msk_i == lbl_idx] = (cat_i[lbl_idx] * 1000) + inst_id
                    inst_id += 1
            po_out.append(po_msk)
            po_vis_out.append(po_msk_vis)

        return cat_out, iscrowd_out, bbx_out, ids_out, sem_out, po_out, po_vis_out

    def forward(self, img, bev_msk=None, front_msk=None, weights_msk=None, cat=None, iscrowd=None, bbx=None, calib=None,
                do_loss=False, do_prediction=False):
        result = OrderedDict()
        loss = OrderedDict()
        stats = OrderedDict()

        # Get some parameters
        img, _ = pad_packed_images(img)

        if bev_msk is not None:
            bev_msk, valid_size = pad_packed_images(bev_msk)
            img_size = bev_msk.shape[-2:]
        else:
            valid_size = [torch.Size([896, 768])] * img.shape[0]
            img_size = torch.Size([896, 768])

        if front_msk is not None:
            front_msk, _ = pad_packed_images(front_msk)

        if do_loss:
            # Prepare the input data and the ground truth labels
            cat, iscrowd, bbx, ids, sem_gt, po_gt, po_gt_vis = self.prepare_inputs(bev_msk, cat, iscrowd, bbx)
            if self.dataset == "Kitti360":
                vf_mask_gt = [self.make_vf_mask(front_msk)]
            elif self.dataset == "nuScenes":
                vf_mask_gt = [front_msk]  # List to take care of the "rgb_cameras"
            v_region_mask_gt, f_region_mask_gt = self.make_region_mask(sem_gt)
            v_region_mask_gt = [v_region_mask_gt]
            f_region_mask_gt = [f_region_mask_gt]
        else:
            po_gt, po_gt_vis = None, None

        # Get the image features
        ms_feat = self.body(img)

        # Transform from the front view to the BEV and upsample the height dimension
        ms_bev, vf_logits_list, v_region_logits_list, f_region_logits_list = self.transformer(ms_feat, calib)
        if do_loss:
            vf_loss, v_region_loss, f_region_loss = self.transformer_algo.training(vf_logits_list, v_region_logits_list,
                                                                                   f_region_logits_list, vf_mask_gt,
                                                                                   v_region_mask_gt, f_region_mask_gt)
        elif do_prediction:
            vf_loss, v_region_loss, f_region_loss = None, None, None
        else:
            vf_logits_list, ms_bev, vf_loss, v_region_loss, f_region_loss = None, None, None, None, None

        # RPN Part
        if do_loss:
            obj_loss, bbx_loss, proposals = self.rpn_algo.training(self.rpn_head, ms_bev, bbx, iscrowd, valid_size,
                                                                   training=self.training, do_inference=True)
        elif do_prediction:
            proposals = self.rpn_algo.inference(self.rpn_head, ms_bev, valid_size, self.training)
            obj_loss, bbx_loss = None, None
        else:
            obj_loss, bbx_loss, proposals = None, None, None

        # ROI Part
        if do_loss:
            roi_cls_loss, roi_bbx_loss, roi_msk_loss, roi_cls_logits, roi_bbx_logits, roi_msk_logits = \
                self.inst_algo.training(self.roi_head, ms_bev, proposals, bbx, cat, iscrowd, ids, bev_msk, img_size)
        else:
            roi_cls_loss, roi_bbx_loss, roi_msk_loss = None, None, None
            roi_cls_logits, roi_bbx_logits, roi_msk_logits = None, None, None
        if do_prediction:
            bbx_pred, cls_pred, obj_pred, msk_pred, roi_msk_logits = self.inst_algo.inference(self.roi_head, ms_bev,
                                                                                              proposals, valid_size,
                                                                                              img_size)
        else:
            bbx_pred, cls_pred, obj_pred, msk_pred = None, None, None, None

        # Segmentation Part
        if do_loss:
            sem_loss, sem_conf_mat, sem_pred, sem_logits, sem_feat = self.sem_algo.training(self.sem_head, ms_bev,
                                                                                            sem_gt, bbx, valid_size,
                                                                                            img_size, weights_msk,
                                                                                            calib)
        elif do_prediction:
            sem_pred, sem_logits, sem_feat = self.sem_algo.inference(self.sem_head, ms_bev, valid_size, img_size)
            sem_loss, sem_reg_loss, sem_conf_mat = None, None, None
        else:
            sem_loss, sem_reg_loss, sem_conf_mat, sem_pred, sem_logits, sem_feat = None, None, None, None, None, None

        # Panoptic Fusion. Fuse the semantic and instance predictions to generate a coherent output
        if do_prediction:
            # The first channel of po_pred contains the semantic labels
            # The second channel contains the instance masks with the instance label being the corresponding semantic label
            po_pred, po_loss, po_logits = self.po_fusion_algo.inference(sem_logits, roi_msk_logits, bbx_pred, cls_pred,
                                                                        img_size)
        elif do_loss:
            po_loss = self.po_fusion_algo.training(sem_logits, roi_msk_logits, bbx, cat, po_gt, img_size)
            po_pred, po_logits = None, None
        else:
            po_pred, po_loss, po_logits = None, None, None

        # Prepare outputs
        # LOSSES
        loss['obj_loss'] = obj_loss
        loss['bbx_loss'] = bbx_loss
        loss['roi_cls_loss'] = roi_cls_loss
        loss['roi_bbx_loss'] = roi_bbx_loss
        loss['roi_msk_loss'] = roi_msk_loss
        loss["sem_loss"] = sem_loss
        loss['vf_loss'] = vf_loss
        loss['v_region_loss'] = v_region_loss
        loss['f_region_loss'] = f_region_loss
        loss['po_loss'] = po_loss

        # PREDICTIONS
        result['bbx_pred'] = bbx_pred
        result['cls_pred'] = cls_pred
        result['obj_pred'] = obj_pred
        result['msk_pred'] = msk_pred
        result["sem_pred"] = sem_pred
        result['sem_logits'] = sem_logits
        result['vf_logits'] = vf_logits_list
        result['v_region_logits'] = v_region_logits_list
        result['f_region_logits'] = f_region_logits_list
        if po_pred is not None:
            result['po_pred'] = po_pred[0]
            result['po_class'] = po_pred[1]
            result['po_iscrowd'] = po_pred[2]

        # STATS
        stats['sem_conf'] = sem_conf_mat

        return loss, result, stats
