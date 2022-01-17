import torch
from inplace_abn import active_group, set_active_group

from panoptic_bev.utils.bbx import shift_boxes
from panoptic_bev.utils.misc import Empty
from panoptic_bev.utils.parallel import PackedSequence
from panoptic_bev.utils.roi_sampling import roi_sampling
from panoptic_bev.algos.detection import DetectionAlgo
from panoptic_bev.algos.instance_seg import InstanceSegAlgo
from panoptic_bev.algos.rpn import RPNAlgo


class RPNAlgoFPN(RPNAlgo):
    """RPN algorithm for FPN-based region proposal networks

    Parameters
    ----------
    proposal_generator : RPNProposalGenerator
    anchor_matcher : RPNAnchorMatcher
    loss : RPNLoss
    anchor_scale : list
        Anchor scale factor, this is multiplied by the RPN stride at each level to determine the actual anchor sizes
    anchor_ratios : sequence of float
        Anchor aspect ratios
    anchor_strides: sequence of int
        Effective strides of the RPN outputs at each FPN level
    min_level : int
        First FPN level to work on
    levels : int
        Number of FPN levels to work on
    """

    def __init__(self,
                 proposal_generator,
                 anchor_matcher,
                 loss,
                 anchor_scale,
                 anchor_ratios,
                 anchor_strides,
                 min_level,
                 levels):
        super(RPNAlgoFPN, self).__init__(anchor_scale, anchor_ratios)
        self.proposal_generator = proposal_generator
        self.anchor_matcher = anchor_matcher
        self.loss = loss
        self.min_level = min_level
        self.levels = levels

        # Cache per-cell anchors
        self.anchor_strides = anchor_strides[min_level:min_level + levels]
        self.anchors = [self._base_anchors(stride) for stride in self.anchor_strides]

    @staticmethod
    def _get_logits(head, x):
        obj_logits, bbx_logits, h, w = [], [], [], []
        for x_i in x:
            obj_logits_i, bbx_logits_i = head(x_i)
            h_i, w_i = (int(s) for s in obj_logits_i.shape[-2:])

            obj_logits_i = obj_logits_i.permute(0, 2, 3, 1).contiguous().view(obj_logits_i.size(0), -1)
            bbx_logits_i = bbx_logits_i.permute(0, 2, 3, 1).contiguous().view(bbx_logits_i.size(0), -1, 4)

            obj_logits.append(obj_logits_i)
            bbx_logits.append(bbx_logits_i)
            h.append(h_i)
            w.append(w_i)

        return torch.cat(obj_logits, dim=1), torch.cat(bbx_logits, dim=1), h, w

    def _inference(self, obj_logits, bbx_logits, anchors, valid_size, training):
        # Compute shifted boxes
        boxes = shift_boxes(anchors, bbx_logits)

        # Clip boxes to their image sizes
        for i, (height, width) in enumerate(valid_size):
            boxes[i, :, [0, 2]] = boxes[i, :, [0, 2]].clamp(min=0, max=height)
            boxes[i, :, [1, 3]] = boxes[i, :, [1, 3]].clamp(min=0, max=width)

        return self.proposal_generator(boxes, obj_logits, training)

    def training(self, head, x, bbx, iscrowd, valid_size, training=True, do_inference=False):
        # Calculate logits for the levels that we need
        x = x[self.min_level:self.min_level + self.levels]
        obj_logits, bbx_logits, h, w = self._get_logits(head, x)
        with torch.no_grad():
            # Compute anchors for each scale and merge them
            anchors = []
            for h_i, w_i, stride_i, anchors_i in zip(h, w, self.anchor_strides, self.anchors):
                anchors.append(self._shifted_anchors(
                    anchors_i, stride_i, h_i, w_i, bbx_logits.dtype, bbx_logits.device))
            anchors = torch.cat(anchors, dim=0)

            match = self.anchor_matcher(anchors, bbx, iscrowd, valid_size)
            obj_lbl, bbx_lbl = self._match_to_lbl(anchors, bbx, match)
        # Compute losses
        obj_loss, bbx_loss = self.loss(obj_logits, bbx_logits, obj_lbl, bbx_lbl)

        # Optionally, also run inference
        if do_inference:
            with torch.no_grad():
                proposals = self._inference(obj_logits, bbx_logits, anchors, valid_size, training)
        else:
            proposals = None

        return obj_loss, bbx_loss, proposals

    def inference(self, head, x, valid_size, training):
        # Calculate logits for the levels that we need
        x = x[self.min_level:self.min_level + self.levels]
        obj_logits, bbx_logits, h, w = self._get_logits(head, x)

        # Compute anchors for each scale and merge them
        anchors = []
        for h_i, w_i, stride_i, anchors_i in zip(h, w, self.anchor_strides, self.anchors):
            anchors.append(self._shifted_anchors(
                anchors_i, stride_i, h_i, w_i, bbx_logits.dtype, bbx_logits.device))
        anchors = torch.cat(anchors, dim=0)

        return self._inference(obj_logits, bbx_logits, anchors, valid_size, training)


class DetectionAlgoFPN(DetectionAlgo):
    """Detection algorithm for FPN networks

    Parameters
    ----------
    prediction_generator : PredictionGenerator
    proposal_matcher : ProposalMatcher
    loss : FasterRCNNLoss
    classes : dict
        Dictionary with the number of classes in the dataset -- expected keys: "total", "stuff", "thing"
    bbx_reg_weights : sequence of float
        Weights assigned to the bbx regression coordinates
    canonical_scale : int
        Reference scale for ROI to FPN level assignment
    canonical_level : int
        Reference level for ROI to FPN level assignment
    roi_size : tuple of int
        Spatial size of the ROI features as `(height, width)`
    min_level : int
        First FPN level to work on
    levels : int
        Number of FPN levels to work on
    """

    def __init__(self,
                 prediction_generator,
                 proposal_matcher,
                 loss,
                 classes,
                 bbx_reg_weights,
                 canonical_scale,
                 canonical_level,
                 roi_size,
                 min_level,
                 levels):
        super(DetectionAlgoFPN, self).__init__(classes, bbx_reg_weights)
        self.prediction_generator = prediction_generator
        self.proposal_matcher = proposal_matcher
        self.loss = loss
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level
        self.roi_size = roi_size
        self.min_level = min_level
        self.levels = levels

    def _target_level(self, boxes):
        scales = (boxes[:, 2:] - boxes[:, :2]).prod(dim=-1).sqrt()
        target_level = torch.floor(self.canonical_level + torch.log2(scales / self.canonical_scale + 1e-6))
        return target_level.clamp(min=self.min_level, max=self.min_level + self.levels - 1)

    def _rois(self, x, proposals, proposals_idx, img_size):
        stride = proposals.new([fs / os for fs, os in zip(x.shape[-2:], img_size)])
        proposals = (proposals - 0.5) * stride.repeat(2) + 0.5
        return roi_sampling(x, proposals, proposals_idx, self.roi_size)

    def _head(self, head, x, proposals, proposals_idx, img_size):
        # Find target levels
        target_level = self._target_level(proposals)

        # Sample rois
        rois = x[0].new_zeros(proposals.size(0), x[0].size(1), self.roi_size[0], self.roi_size[1])
        for level_i, x_i in enumerate(x):
            idx = target_level == (level_i + self.min_level)
            if idx.any().item():
                rois[idx] = self._rois(x_i, proposals[idx], proposals_idx[idx], img_size)

        # Run head
        return head(rois)

    def training(self, head, x, proposals, bbx, cat, iscrowd, img_size):
        x = x[self.min_level:self.min_level + self.levels]

        try:
            if proposals.all_none:
                raise Empty

            with torch.no_grad():
                # Match proposals to ground truth
                proposals, match = self.proposal_matcher(proposals, bbx, cat, iscrowd)
                cls_lbl, bbx_lbl = self._match_to_lbl(proposals, bbx, cat, match)
                
            if proposals.all_none:
                raise Empty

            # Run head
            set_active_group(head, active_group(True))
            proposals, proposals_idx = proposals.contiguous
            cls_logits, bbx_logits = self._head(head, x, proposals, proposals_idx, img_size)
            # Calculate loss
            cls_loss, bbx_loss = self.loss(cls_logits, bbx_logits, cls_lbl, bbx_lbl)
        except Empty:
            active_group(False)
            cls_loss = bbx_loss = sum(x_i.sum() for x_i in x) * 0

        return cls_loss, bbx_loss

    def inference(self, head, x, proposals, valid_size, img_size):
        x = x[self.min_level:self.min_level + self.levels]

        if not proposals.all_none:
            # Run head on the given proposals
            proposals, proposals_idx = proposals.contiguous
            cls_logits, bbx_logits = self._head(head, x, proposals, proposals_idx, img_size)

            # Shift the proposals according to the logits
            bbx_reg_weights = x[0].new(self.bbx_reg_weights)
            boxes = shift_boxes(proposals.unsqueeze(1), bbx_logits / bbx_reg_weights)
            scores = torch.softmax(cls_logits, dim=1)

            # Split boxes and scores by image, clip to valid size
            boxes, scores = self._split_and_clip(boxes, scores, proposals_idx, valid_size)

            bbx_pred, cls_pred, obj_pred = self.prediction_generator(boxes, scores)
        else:
            bbx_pred = PackedSequence([None for _ in range(x[0].size(0))])
            cls_pred = PackedSequence([None for _ in range(x[0].size(0))])
            obj_pred = PackedSequence([None for _ in range(x[0].size(0))])

        return bbx_pred, cls_pred, obj_pred


class InstanceSegAlgoFPN(InstanceSegAlgo):
    """Instance segmentation algorithm for FPN networks

    Parameters
    ----------
    bbx_prediction_generator : faster_rcnn.PredictionGenerator
    msk_prediction_generator : mask_rcnn.PredictionGenerator
    proposal_matcher : faster_rcnn.ProposalMatcher
    bbx_loss : FasterRCNNLoss
    msk_loss : MaskRCNNLoss
    classes : dict
        Dictionary with the number of classes in the dataset -- expected keys: "total", "stuff", "thing"
    bbx_reg_weights : sequence of float
        Weights assigned to the bbx regression coordinates
    canonical_scale : int
        Reference scale for ROI to FPN level assignment
    canonical_level : int
        Reference level for ROI to FPN level assignment
    roi_size : tuple of int
        Spatial size of the ROI features as `(height, width)`
    min_level : int
        First FPN level to work on
    levels : int
        Number of FPN levels to work on
    lbl_roi_size : tuple of int
        Spatial size of the ROI mask labels as `(height, width)`
    void_is_background : bool
        If True treat void areas as background in the instance mask loss instead of void
    """

    def __init__(self,
                 bbx_prediction_generator,
                 msk_prediction_generator,
                 proposal_matcher,
                 bbx_loss,
                 msk_loss,
                 classes,
                 bbx_reg_weights,
                 canonical_scale,
                 canonical_level,
                 roi_size,
                 min_level,
                 levels,
                 lbl_roi_size=(28, 28),
                 void_is_background=False,
                 debug = False):
        super(InstanceSegAlgoFPN, self).__init__(classes, bbx_reg_weights, lbl_roi_size, void_is_background)
        self.bbx_prediction_generator = bbx_prediction_generator
        self.msk_prediction_generator = msk_prediction_generator
        self.proposal_matcher = proposal_matcher
        self.bbx_loss = bbx_loss
        self.msk_loss = msk_loss
        self.canonical_scale = canonical_scale
        self.canonical_level = canonical_level
        self.roi_size = roi_size
        self.min_level = min_level
        self.levels = levels
        self.debug = debug

    def _target_level(self, boxes):
        scales = (boxes[:, 2:] - boxes[:, :2]).prod(dim=-1).sqrt()
        target_level = torch.floor(self.canonical_level + torch.log2(scales / self.canonical_scale + 1e-6))
        return target_level.clamp(min=self.min_level, max=self.min_level + self.levels - 1)

    def _rois(self, x, proposals, proposals_idx, img_size):
        stride = proposals.new([fs / os for fs, os in zip(x.shape[-2:], img_size)])
        proposals = (proposals - 0.5) * stride.repeat(2) + 0.5
        return roi_sampling(x, proposals, proposals_idx, self.roi_size)

    def _head1(self, head, x, proposals, proposals_idx, img_size, do_cls_bbx, do_msk):
        # Find target levels
        target_level = self._target_level(proposals)

        # Sample rois
        rois = x[0][0].new_zeros(proposals.size(0), x[0][0].size(1), self.roi_size[0], self.roi_size[1])
        level_i = 0
        for  x_i, fx_i in zip(x[0],x[1]):
            idx = target_level == (level_i + self.min_level)
            if idx.any().item():
                rois[idx] = self._rois(x_i, proposals[idx], proposals_idx[idx], img_size)
            
            level_i+=1
        z,l,s = head(rois, do_cls_bbx, do_msk)
        level_i = 0
        for  x_i, fx_i in zip(x[0],x[1]):
            idx = target_level == (level_i + self.min_level)
            if idx.any().item():
                rois[idx] = self._rois(fx_i, proposals[idx], proposals_idx[idx], img_size)
            
            level_i+=1
        z1,l1,s1 = head(rois, do_cls_bbx, do_msk)
        if do_msk:
            s = (s+s1)/2.0
        else:
            z = (z1+z)/2.0
            l = (l1+l)/2.0    

        # Run head
        return z,l,s #head(rois, do_cls_bbx, do_msk)

    def _head(self, head, x, proposals, proposals_idx, img_size, do_cls_bbx, do_msk):
        # Find target levels
        target_level = self._target_level(proposals)

        # Sample rois
        rois = x[0].new_zeros(proposals.size(0), x[0].size(1), self.roi_size[0], self.roi_size[1])
        for level_i, x_i in enumerate(x):
            idx = target_level == (level_i + self.min_level)
            if idx.any().item():
                rois[idx] = self._rois(x_i, proposals[idx], proposals_idx[idx], img_size)

        # Run head
        # This is to prevent batch norm from crashing when there is only ony proposal.
        prune = False
        if rois.shape[0] == 1:
            prune = True
            rois = torch.cat([rois, rois], dim=0)
        cls_logits, bbx_logits, msk_logits = head(rois, do_cls_bbx, do_msk)
        if prune:
            if cls_logits is not None:
                cls_logits = cls_logits[0, ...].unsqueeze(0)
            if bbx_logits is not None:
                bbx_logits = bbx_logits[0, ...].unsqueeze(0)
            if msk_logits is not None:
                msk_logits = msk_logits[0, ...].unsqueeze(0)

        return cls_logits, bbx_logits, msk_logits


    def _get_bbox_idxs(self, bbx, bbx_gt):
        bbx_gt_idxs = torch.zeros(bbx_gt.shape[0], dtype=torch.int64, device=bbx[0].device)
        start = 0
        for idx, bbx_i in enumerate(bbx):
            bbx_gt_idxs[start:start+bbx_i.shape[0]] = idx
            start += bbx_i.shape[0]
        return bbx_gt_idxs

    def _make_batch_list(self, cls_gt_logits, bbx_gt_logits, msk_gt_logits, bbx_gt_idx, batch_size):
        cls_gt_list, bbx_gt_list, msk_gt_list = [], [], []
        unique_idxs = torch.unique(bbx_gt_idx)
        for entry in range(batch_size):
            if torch.sum(unique_idxs == entry) == 0:
                cls_gt_list.append(None)
                bbx_gt_list.append(None)
                msk_gt_list.append(None)
            else:
                mask = (bbx_gt_idx == entry)
                if cls_gt_logits is not None:
                    cls_gt_list.append(cls_gt_logits[mask, ...])
                if bbx_gt_logits is not None:
                    bbx_gt_list.append(bbx_gt_logits[mask, ...])
                if msk_gt_logits is not None:
                    msk_gt_list.append(msk_gt_logits[mask, ...])

        return cls_gt_list, bbx_gt_list, msk_gt_list

    def training(self, head, x, proposals, bbx, cat, iscrowd, ids, msk, img_size):
        x = x[self.min_level:self.min_level + self.levels]
        
        try:
            if proposals.all_none:
                raise Empty

            # Match proposals to ground truth
            with torch.no_grad():
                proposals, match = self.proposal_matcher(proposals, bbx, cat, iscrowd)
                cls_lbl, bbx_lbl, msk_lbl = self._match_to_lbl(proposals, bbx, cat, ids, msk, match)

            if proposals.all_none:
                raise Empty

            # Run head
            if not self.debug:
                set_active_group(head, active_group(True))
            else:
                pass
            proposals, proposals_idx = proposals.contiguous
            cls_logits, bbx_logits, msk_logits = self._head(head, x, proposals, proposals_idx, img_size, True, True)

            # Predict the masks using the ground truth. This is used for the panoptic fusion
            batch_size = len(bbx)
            bbx_ps = PackedSequence(bbx)
            if bbx_ps.all_none:
                cls_gt_logits = [None] * batch_size
                bbx_gt_logits = [None] * batch_size
                msk_gt_logits = [None] * batch_size
            else:
                bbx_gt, bbx_gt_idx = bbx_ps.contiguous
                cls_gt_logits, bbx_gt_logits, msk_gt_logits = self._head(head, x, bbx_gt, bbx_gt_idx, img_size, True, True)
                cls_gt_logits, bbx_gt_logits, msk_gt_logits = self._make_batch_list(cls_gt_logits, bbx_gt_logits,
                                                                                    msk_gt_logits, bbx_gt_idx,
                                                                                    batch_size)
            cls_gt_logits = PackedSequence(cls_gt_logits)
            bbx_gt_logits = PackedSequence(bbx_gt_logits)
            msk_gt_logits = PackedSequence(msk_gt_logits)

            # Calculate losses
            cls_loss, bbx_loss = self.bbx_loss(cls_logits, bbx_logits, cls_lbl, bbx_lbl)
            msk_loss = self.msk_loss(msk_logits, cls_lbl, msk_lbl)

        except Empty:
            if not self.debug:
                active_group(False)
            cls_loss = bbx_loss = msk_loss = sum(x_i.sum() for x_i in x) * 0
            batch_size = len(bbx)
            cls_gt_logits, bbx_gt_logits, msk_gt_logits = [None] * batch_size, [None] * batch_size, [None] * batch_size
 
        return cls_loss, bbx_loss, msk_loss, cls_gt_logits, bbx_gt_logits, msk_gt_logits

    def inference(self, head, x, proposals, valid_size, img_size):
        x = x[self.min_level:self.min_level + self.levels]

        try:
            if proposals.all_none:
                raise Empty

            # Run head on the given proposals
            proposals, proposals_idx = proposals.contiguous
            cls_logits, bbx_logits, _ = self._head(head, x, proposals, proposals_idx, img_size, True, False)
            
            # Shift the proposals according to the logits
            bbx_reg_weights = x[0].new(self.bbx_reg_weights)
            boxes = shift_boxes(proposals.unsqueeze(1), bbx_logits / bbx_reg_weights)
            scores = torch.softmax(cls_logits, dim=1)

            # Split boxes and scores by image, clip to valid size
            boxes, scores = self._split_and_clip(boxes, scores, proposals_idx, valid_size)

            # Do nms to find final predictions
            bbx_pred, cls_pred, obj_pred = self.bbx_prediction_generator(boxes, scores)

            if bbx_pred.all_none:
                raise Empty

            # Run head again on the finalized boxes to compute instance masks
            proposals, proposals_idx = bbx_pred.contiguous
            _, _, msk_logits = self._head(head, x, proposals, proposals_idx, img_size, False, True)

            # Finalize instance mask computation
            batch_size = len(bbx_pred)
            msk_pred = self.msk_prediction_generator(cls_pred, msk_logits)
            _, _, msk_logits = self._make_batch_list(None, None, msk_logits, proposals_idx, batch_size)
        except Empty:
            bbx_pred = PackedSequence([None for _ in range(x[0].size(0))])
            cls_pred = PackedSequence([None for _ in range(x[0].size(0))])
            obj_pred = PackedSequence([None for _ in range(x[0].size(0))])
            msk_pred = PackedSequence([None for _ in range(x[0].size(0))])
            msk_logits = PackedSequence([None for _ in range(x[0].size(0))])

        return bbx_pred, cls_pred, obj_pred, msk_pred, msk_logits
