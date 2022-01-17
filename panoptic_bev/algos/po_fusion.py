import torch
import torch.nn.functional as F
from panoptic_bev.utils.parallel import PackedSequence


class PanopticLoss:
    """Semantic segmentation loss

    Parameters
    ----------
    ohem : float or None
        Online hard example mining fraction, or `None` to disable OHEM
    ignore_index : int
        Index of the void class
    """

    def __init__(self, num_stuff, inst_weight=50, ignore_index=255):
        self.num_stuff = num_stuff
        self.ignore_index = ignore_index
        self.inst_weight = inst_weight

    def __call__(self, po_logits, po_gt):
        """Compute the semantic segmentation loss

        Parameters
        ----------
        sem_logits : sequence of torch.Tensor
            A sequence of N tensors of segmentation logits with shapes C x H_i x W_i
        sem : sequence of torch.Tensor
            A sequence of N tensors of ground truth semantic segmentations with shapes H_i x W_i

        Returns
        -------
        sem_loss : torch.Tensor
            A scalar tensor with the computed loss
        """
        po_loss = []
        for po_logits_i, po_i in zip(po_logits, po_gt):
            weight = torch.ones(po_logits_i.shape[0], dtype=torch.float, device=po_logits_i.device)
            weight[self.num_stuff:] = self.inst_weight
            po_loss_i = F.cross_entropy(po_logits_i.unsqueeze(0), po_i.unsqueeze(0), ignore_index=self.ignore_index,
                                        weight=weight, reduction="none")
            po_loss_i = po_loss_i.view(-1)

            po_loss.append(po_loss_i.mean())

        return sum(po_loss) / len(po_logits)


class PanopticFusionAlgo:
    def __init__(self, loss, num_stuff, num_thing, sem_stride, min_stuff_area=0):
        self.loss = loss
        self.num_stuff = num_stuff
        self.num_thing = num_thing
        self.sem_stride = sem_stride
        self.min_stuff_area = min_stuff_area

    def _process_semantic_logits(self, sem_logits, boxes, classes, inference=False):
        po_logits_stuff, po_logits_inst = [], []

        # Handle features from the semantic head
        for sem_logits_i, bbx_i, cat_i in zip(sem_logits, boxes, classes):
            # Handle stuff
            po_logits_stuff_i = sem_logits_i[:self.num_stuff, ...]

            # Handle thing
            if (bbx_i is None) or (cat_i is None):
                po_logits_stuff.append(po_logits_stuff_i)
                po_logits_inst.append(None)
            else:
                bbx_i = bbx_i / self.sem_stride

                po_logits_inst_i = torch.ones((cat_i.shape[0], po_logits_stuff_i.shape[1], po_logits_stuff_i.shape[2]),
                                              device=sem_logits.device) * -100
                for box_id in range(cat_i.shape[0]):
                    y_min = int(bbx_i[box_id][0])
                    y_max = int(bbx_i[box_id][2].round() + 1)
                    x_min = int(bbx_i[box_id][1])
                    x_max = int(bbx_i[box_id][3].round() + 1)
                    if not inference:
                        po_logits_inst_i[box_id, y_min:y_max, x_min:x_max] = \
                            sem_logits_i[cat_i[box_id], y_min:y_max, x_min:x_max]
                    else:
                        po_logits_inst_i[box_id, y_min:y_max, x_min:x_max] = \
                            sem_logits_i[cat_i[box_id] + self.num_stuff, y_min:y_max, x_min:x_max]

                po_logits_stuff.append(po_logits_stuff_i)
                po_logits_inst.append(po_logits_inst_i)

        return po_logits_stuff, po_logits_inst

    def process_mask_logits(self, sem_logits, roi_msk_logits, boxes, classes, img_size):
        po_logits_mask = []

        # Handle features from the instance head
        for sem_logits_i, masks_i, bbx_i, cat_i in zip(sem_logits, roi_msk_logits, boxes, classes):
            if (bbx_i is None) or (cat_i is None):
                po_logits_mask.append(None)
            else:
                po_logits_mask_i = torch.ones((cat_i.shape[0], sem_logits_i.shape[1], sem_logits_i.shape[2]),
                                              device=sem_logits.device) * -100

                for box_id in range(cat_i.shape[0]):
                    ref_box = bbx_i[box_id, :].long()
                    y_min = int(bbx_i[box_id][0])
                    y_max = int(bbx_i[box_id][2])
                    x_min = int(bbx_i[box_id][1])
                    x_max = int(bbx_i[box_id][3])
                    w = max((x_max - x_min + 1), 1)
                    h = max((y_max - y_min + 1), 1)

                    roi_edge = masks_i.shape[2]
                    mask = F.upsample(masks_i[box_id, :, :].view(1, 1, roi_edge, roi_edge), size=(h, w),
                                      mode="bilinear", align_corners=False).squeeze(0)
                    x_min = max(ref_box[1], 0)
                    x_max = min(ref_box[3] + 1, img_size[1])
                    y_min = max(ref_box[0], 0)
                    y_max = min(ref_box[2] + 1, img_size[0])

                    po_logits_mask_i[box_id, y_min:y_max, x_min:x_max] = \
                        mask[0, (y_min - ref_box[0]):(y_max - ref_box[0]), (x_min - ref_box[1]):(x_max - ref_box[1])]

                po_logits_mask.append(po_logits_mask_i)

        return po_logits_mask

    def assign_class_label(self, po_pred, sem_logits, cls):
        po_2ch = []
        for po_pred_i, sem_logits_i, cls_i in zip(po_pred, sem_logits, cls):
            sem_pred_i = torch.max(torch.softmax(sem_logits_i, dim=0), dim=0)[1]
            po_sem_i = po_pred_i.clone()
            po_inst_i = po_pred_i.clone()

            ids = torch.unique(po_pred_i)
            ids_inst = ids[ids >= self.num_stuff]
            po_inst_i[po_inst_i < self.num_stuff] = -1

            if cls_i is not None:
                for idx, inst_id in enumerate(ids_inst):
                    region = (po_inst_i == inst_id)
                    # if inst_id == 255:
                    #     po_sem_i[region] = 255
                    #     po_inst_i[region] = -1
                    #     continue

                    # Get the different semantic class IDs in the instance region
                    sem_cls_i, sem_cnt_i = torch.unique(sem_pred_i[region], return_counts=True)

                    if sem_cls_i[torch.argmax(sem_cnt_i)] == cls_i[inst_id - self.num_stuff] + self.num_stuff:
                        # The semantic and instance class IDs agree with each other.
                        po_sem_i[region] = cls_i[inst_id - self.num_stuff] + self.num_stuff
                        po_inst_i[region] = idx
                    else:
                        # The semantic and instance class IDs do not agree with each other
                        if (torch.max(sem_cnt_i).type(torch.float) / torch.sum(sem_cnt_i).type(torch.float) >= 0.5) \
                                and (sem_cls_i[torch.argmax(sem_cnt_i)] < self.num_stuff):
                            # If the frequency of the mode is more than 0.5 and the sem label is a "stuff",
                            # assign the stuff label to it
                            po_sem_i[region] = sem_cls_i[torch.argmax(sem_cnt_i)]
                            po_inst_i[region] = -1
                        else:
                            # Else assign the instance segmentation class label to it
                            po_sem_i[region] = cls_i[inst_id - self.num_stuff] + self.num_stuff
                            po_inst_i[region] = idx

                idx_sem = torch.unique(po_sem_i)
                for i in range(idx_sem.shape[0]):
                    if idx_sem[i] < self.num_stuff:
                        region = (po_sem_i == idx_sem[i])
                        if region.sum() < self.min_stuff_area:
                            po_sem_i[region] = 255

            po_2ch_i = torch.zeros((po_pred_i.shape[0], po_pred_i.shape[1], 2), dtype=torch.int)
            po_2ch_i[:, :, 0] = po_sem_i
            po_2ch_i[:, :, 1] = po_inst_i
            po_2ch.append(po_2ch_i)

        return po_2ch

    def generate_seamless_output(self, po_2ch):
        po_cls = []
        po_pred_seamless = []
        po_iscrowd = []

        for po_2ch_i in po_2ch:
            po_sem_i = po_2ch_i[:, :, 0]
            po_inst_i = po_2ch_i[:, :, 1]

            # Generate seamless-style panoptic output
            po_cls_i = [255]
            po_pred_seamless_i = torch.zeros_like(po_sem_i, dtype=torch.long)

            # Handle stuff
            classes = torch.unique(po_sem_i)
            stuff_classes = classes[classes < self.num_stuff]
            for idx, cls in enumerate(stuff_classes):
                region = (po_sem_i == cls)
                po_pred_seamless_i[region] = len(po_cls_i)  # Give the new index
                po_cls_i.append(cls.item())

            # Handle instances
            instances = torch.unique(po_inst_i)
            valid_instances = instances[instances >= 0]
            for idx, inst_id in enumerate(valid_instances):
                region = (po_inst_i == inst_id)
                po_pred_seamless_i[region] = len(po_cls_i)
                po_cls_i.append(torch.unique(po_sem_i[region])[0].item())
            po_iscrowd_i = [0] * len(po_cls_i)

            po_pred_seamless.append(po_pred_seamless_i)
            po_cls.append(torch.tensor(po_cls_i))
            po_iscrowd.append(torch.tensor(po_iscrowd_i))

        return po_pred_seamless, po_cls, po_iscrowd

    def training(self, sem_logits, roi_msk_logits, bbx, cat, po_gt, img_size):
        # During training cat has the GT instance labels, i.e, they have thing labels, i.e, [num_stuff, num_stuff + num_thing)
        # Get the roi mask containing the GT
        msk_logits = []
        for roi_msk_logits_i, cat_i in zip(roi_msk_logits, cat):
            if roi_msk_logits_i is None:
                msk_logits.append(None)
            else:
                msk_logits_i = torch.cat([roi_msk_logits_i[idx, cat_i[idx] - self.num_stuff, :, :].unsqueeze(0)
                                          for idx in range(roi_msk_logits_i.shape[0])], dim=0)
                msk_logits.append(msk_logits_i)

        po_logits_stuff, po_logits_inst = self._process_semantic_logits(sem_logits, bbx, cat)
        po_logits_mask = self.process_mask_logits(sem_logits, msk_logits, bbx, cat, img_size)

        po_logits = []
        for stuff_i, inst_i, mask_i in zip(po_logits_stuff, po_logits_inst, po_logits_mask):
            if (inst_i is None) or (mask_i is None):
                po_logits_i = stuff_i
            else:
                # inst_i = logit_scaling_head()
                combined_inst_i = (inst_i.sigmoid() + mask_i.sigmoid()) * (inst_i + mask_i)
                po_logits_i = torch.cat([stuff_i, combined_inst_i], dim=0)
            po_logits.append(po_logits_i)
        po_logits = PackedSequence(po_logits)

        # Compute panoptic loss
        po_loss = self.loss(po_logits, po_gt)

        return po_loss

    def inference(self, sem_logits, roi_msk_logits, bbx, cls, img_size):
        # During inference, cls has instance classes starting from 0, i.e, from [0, num_thing)
        # Get the roi mask containing the GT
        msk_logits = []
        for roi_msk_logits_i, cls_i in zip(roi_msk_logits, cls):
            if roi_msk_logits_i is None:
                msk_logits.append(None)
            else:
                msk_logits_i = torch.cat([roi_msk_logits_i[idx, cls_i[idx], :, :].unsqueeze(0)
                                          for idx in range(roi_msk_logits_i.shape[0])], dim=0)
                msk_logits.append(msk_logits_i)

        po_logits_stuff, po_logits_inst = self._process_semantic_logits(sem_logits, bbx, cls, inference=True)
        po_logits_mask = self.process_mask_logits(sem_logits, msk_logits, bbx, cls, img_size)

        po_pred = []
        po_logits = []
        for stuff_i, inst_i, mask_i in zip(po_logits_stuff, po_logits_inst, po_logits_mask):
            if (inst_i is None) or (mask_i is None):
                po_logits_i = stuff_i
            else:
                combined_inst_i = (inst_i.sigmoid() + mask_i.sigmoid()) * (inst_i + mask_i)
                po_logits_i = torch.cat([stuff_i, combined_inst_i], dim=0)
            po_logits.append(po_logits_i)
            po_pred_i = torch.max(torch.softmax(po_logits_i, dim=0), dim=0)[1]
            po_pred.append(po_pred_i)

        # Get the panoptic instance labels for every pixel.
        # There could be some discrepancy between the class predicted by semantic seg and instance seg
        po_2ch = self.assign_class_label(po_pred, sem_logits, cls)
        po_pred_seamless, po_cls, po_iscrowd = self.generate_seamless_output(po_2ch)

        po_pred_seamless = PackedSequence(po_pred_seamless)
        po_cls = PackedSequence(po_cls)
        po_iscrowd = PackedSequence(po_iscrowd)

        po_loss = torch.tensor(0., device=sem_logits.device)

        return [po_pred_seamless, po_cls, po_iscrowd], po_loss, po_logits
