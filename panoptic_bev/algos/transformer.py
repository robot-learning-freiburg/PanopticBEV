import torch
import torch.nn.functional as F


class TransformerVFAlgo:
    def __init__(self, vf_loss, region_supervision_loss):
        self.vf_loss = vf_loss
        self.region_supervision_loss = region_supervision_loss

    def training(self, vf_logits_list, v_region_logits_list, f_region_logits_list, vf_mask_gt, v_region_mask_gt, f_region_mask_gt):
        vf_loss = self.vf_loss(vf_logits_list, vf_mask_gt)
        v_region_loss = self.region_supervision_loss(v_region_logits_list, v_region_mask_gt)
        f_region_loss = self.region_supervision_loss(f_region_logits_list, f_region_mask_gt)

        return vf_loss, v_region_loss, f_region_loss

    def inference(self, transformer, ms_cam_feat):
        pass


class TransformerVFLoss:
    """ Cross entropy loss for the vertical-flat sem mask """

    def __init__(self):
        pass

    def __call__(self, vf_logits_list, vf_gt_list):
        vf_sem_loss = []

        gt = vf_gt_list[0]
        for idx in range(len(vf_logits_list)):
            logits_i = vf_logits_list[idx]

            scale = logits_i.shape[2] / gt.shape[2]
            gt_i = F.interpolate(gt.type(torch.float), scale_factor=scale, mode="nearest").squeeze(0).type(torch.long)

            vf_loss = F.cross_entropy(logits_i, gt_i.squeeze(1), ignore_index=2, reduction=('mean'))
            vf_sem_loss.append(vf_loss)

        return sum(vf_sem_loss) / len(vf_logits_list)


class TransformerRegionSupervisionLoss:
    """ Binary Cross Entropy loss for supervising the intermediate predictions in the vertical and flat transformers """

    def __init__(self):
        pass

    def __call__(self, v_region_logits_list, v_region_gt_list):
        v_region_loss = []

        gt = v_region_gt_list[0]
        for idx in range(len(v_region_logits_list)):
            logits_i = v_region_logits_list[idx]

            scale = logits_i.shape[2] / gt.shape[2]
            gt_i = F.interpolate(gt.type(torch.float), scale_factor=scale, mode="nearest")

            loss_i = F.binary_cross_entropy_with_logits(logits_i, gt_i, reduction='none')
            v_region_loss.append(loss_i.mean())

        return sum(v_region_loss) / len(v_region_logits_list)
