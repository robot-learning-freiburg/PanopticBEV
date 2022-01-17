from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as functional
from inplace_abn import ABN
from panoptic_bev.utils.roi_sampling.functions import roi_sampling
from panoptic_bev.utils.parallel import PackedSequence


class FPNMaskHead(nn.Module):
    class _seperable_conv(nn.Module):
        def __init__(self, in_channels, out_channels, dilation, norm_act, bias=False):
            super(FPNMaskHead._seperable_conv, self).__init__()
            self.depthwise = nn.Conv2d(in_channels, in_channels, 3, dilation=dilation, padding=dilation,
                                       groups=in_channels, bias=bias)
            self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

        def forward(self, x):
            x = self.depthwise(x)
            x = self.pointwise(x)
            return x

    """ROI head module for FPN

    Parameters
    ----------
    in_channels : int
        Number of input channels
    classes : dict
        Dictionary with the number of classes in the dataset -- expected keys: "total", "stuff", "thing"
    roi_size : tuple of int
        `(height, width)` of the ROIs extracted from the input feature map, these will be average-pooled 2x2 before
        feeding to the fully-connected branch
    fc_hidden_channels : int
        Number of channels in the hidden layers of the fully-connected branch
    conv_hidden_channels : int
        Number of channels in the hidden layers of the convolutional branch
    norm_act : callable
        Function to create normalization + activation modules
    """

    def __init__(self, in_channels, classes, roi_size, fc_hidden_channels=1024, conv_hidden_channels=256, norm_act=ABN):
        super(FPNMaskHead, self).__init__()

        # ROI section
        self.fc = nn.Sequential(OrderedDict([
            ("fc1", nn.Linear(int(roi_size[0] * roi_size[1] * in_channels / 4), fc_hidden_channels, bias=False)),
            ("bn1", norm_act(fc_hidden_channels)),
            ("fc2", nn.Linear(fc_hidden_channels, fc_hidden_channels, bias=False)),
            ("bn2", norm_act(fc_hidden_channels)),
        ]))
        self.roi_cls = nn.Linear(fc_hidden_channels, classes["thing"] + 1)
        self.roi_bbx = nn.Linear(fc_hidden_channels, classes["thing"] * 4)

        # Mask section

        self.conv = nn.Sequential(OrderedDict([
            ("conv1", nn.Conv2d(in_channels, conv_hidden_channels, 3, padding=1, bias=False)),
            ("bn1", norm_act(conv_hidden_channels)),
            ("conv2", nn.Conv2d(conv_hidden_channels, conv_hidden_channels, 3, padding=1, bias=False)),
            ("bn2", norm_act(conv_hidden_channels)),
            ("conv3", nn.Conv2d(conv_hidden_channels, conv_hidden_channels, 3, padding=1, bias=False)),
            ("bn3", norm_act(conv_hidden_channels)),
            ("conv4", nn.Conv2d(conv_hidden_channels, conv_hidden_channels, 3, padding=1, bias=False)),
            ("bn4", norm_act(conv_hidden_channels)),
            ("conv_up", nn.ConvTranspose2d(conv_hidden_channels, conv_hidden_channels, 2, stride=2, bias=False)),
            ("bn_up", norm_act(conv_hidden_channels)),
        ]))
        self.roi_msk = nn.Conv2d(conv_hidden_channels, classes["thing"], 1)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(self.fc.bn1.activation, self.fc.bn1.activation_param)

        for name, mod in self.named_modules():
            if isinstance(mod, nn.Linear) or isinstance(mod, nn.Conv2d) or isinstance(mod, nn.ConvTranspose2d):
                if "roi_cls" in name or "roi_msk" in name:
                    nn.init.xavier_normal_(mod.weight, .01)
                elif "roi_bbx" in name:
                    nn.init.xavier_normal_(mod.weight, .001)
                else:
                    nn.init.xavier_normal_(mod.weight, gain)
            elif isinstance(mod, ABN):
                nn.init.constant_(mod.weight, 1.)

            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)

    def forward(self, x, do_cls_bbx=True, do_msk=True):
        """ROI head module for FPN

        Parameters
        ----------
        x : torch.Tensor
            A tensor of input features with shape N x C x H x W
        do_cls_bbx : bool
            Whether to compute or not the class and bounding box regression predictions
        do_msk : bool
            Whether to compute or not the mask predictions

        Returns
        -------
        cls_logits : torch.Tensor
            A tensor of classification logits with shape S x (num_thing + 1)
        bbx_logits : torch.Tensor
            A tensor of class-specific bounding box regression logits with shape S x num_thing x 4
        msk_logits : torch.Tensor
            A tensor of class-specific mask logits with shape S x num_thing x (H_roi * 2) x (W_roi * 2)
        """
        # Run fully-connected head
        if do_cls_bbx:
            x_fc = functional.avg_pool2d(x, 2)
            x_fc_out = self.fc(x_fc.view(x_fc.size(0), -1))

            cls_logits = self.roi_cls(x_fc_out)
            bbx_logits = self.roi_bbx(x_fc_out).view(x_fc_out.size(0), -1, 4)
        else:
            cls_logits = None
            bbx_logits = None

        # Run convolutional head
        if do_msk:
            x = self.conv(x)
            msk_logits = self.roi_msk(x)
        else:
            msk_logits = None

        return cls_logits, bbx_logits, msk_logits


class FPNSemanticHeadDPC(nn.Module):
    """Semantic segmentation head for FPN-style networks as in Paper"""

    class _seperable_conv(nn.Module):
        def __init__(self, in_channels, out_channels, dilation, norm_act, bias=False):
            super(FPNSemanticHeadDPC._seperable_conv, self).__init__()
            self.depthwise = nn.Conv2d(in_channels, in_channels, 3, dilation=dilation, padding=dilation,
                                       groups=in_channels, bias=bias)
            self.pointwise = nn.Conv2d(in_channels, out_channels, 1, bias=bias)

        def forward(self, x):
            x = self.depthwise(x)
            x = self.pointwise(x)
            return x

    class _3x3box(nn.Module):
        def __init__(self, seperable_conv, in_channels, out_channels, dilation, norm_act):
            super(FPNSemanticHeadDPC._3x3box, self).__init__()

            self.conv1_3x3_1 = seperable_conv(in_channels, out_channels, (1, 1), norm_act, bias=False)
            self.conv1_3x3_1_bn = norm_act(out_channels)
            self.conv1_3x3_2 = seperable_conv(out_channels, out_channels, (1, 1), norm_act, bias=False)
            self.conv1_3x3_2_bn = norm_act(out_channels)

        def forward(self, x):
            x = self.conv1_3x3_1_bn(self.conv1_3x3_1(x))
            x = self.conv1_3x3_2_bn(self.conv1_3x3_2(x))
            return x

    class _DPC(nn.Module):
        def __init__(self, seperable_conv, in_channels, out_channels, dilation, norm_act):
            super(FPNSemanticHeadDPC._DPC, self).__init__()

            self.conv1_3x3_1 = seperable_conv(in_channels, in_channels, (1, 6), norm_act, bias=False)
            self.conv1_3x3_1_bn = norm_act(in_channels)
            self.conv1_3x3_2 = seperable_conv(in_channels, in_channels, (1, 1), norm_act, bias=False)
            self.conv1_3x3_2_bn = norm_act(in_channels)
            self.conv1_3x3_3 = seperable_conv(in_channels, in_channels, (6, 21), norm_act, bias=False)
            self.conv1_3x3_3_bn = norm_act(in_channels)
            self.conv1_3x3_4 = seperable_conv(in_channels, in_channels, (18, 15), norm_act, bias=False)
            self.conv1_3x3_4_bn = norm_act(in_channels)
            self.conv1_3x3_5 = seperable_conv(in_channels, in_channels, (6, 3), norm_act, bias=False)
            self.conv1_3x3_5_bn = norm_act(in_channels)

            self.conv2 = nn.Conv2d(in_channels * 5, out_channels, 1, bias=False)
            self.bn2 = norm_act(out_channels)

        #            self._swish = MemoryEfficientSwish()

        def forward(self, x):
            x = self.conv1_3x3_1_bn(self.conv1_3x3_1(x))
            x1 = self.conv1_3x3_2_bn(self.conv1_3x3_2(x))
            x2 = self.conv1_3x3_3_bn(self.conv1_3x3_3(x))
            x3 = self.conv1_3x3_4_bn(self.conv1_3x3_4(x))
            x4 = self.conv1_3x3_4_bn(self.conv1_3x3_4(x3))
            x = torch.cat([
                x,
                x1,
                x2,
                x3,
                x4
            ], dim=1)
            x = self.conv2(x)
            x = self.bn2(x)
            return x

    def __init__(self,
                 in_channels,
                 min_level,
                 levels,
                 num_classes,
                 hidden_channels=128,
                 dilation=6,
                 out_size=(768, 704),
                 pooling_size=(64, 64),
                 sem_stride=8,
                 roi_size=(28, 28),
                 norm_act=ABN,
                 interpolation="bilinear", ):
        super(FPNSemanticHeadDPC, self).__init__()
        self.min_level = min_level
        self.levels = levels
        self.interpolation = interpolation
        self.sem_stride = sem_stride
        self.roi_size = roi_size

        self.output_1 = nn.ModuleList([
            self._DPC(self._seperable_conv, in_channels, hidden_channels, dilation, norm_act) for _ in range(levels - 2)
        ])
        self.output_2 = nn.ModuleList([
            self._3x3box(self._seperable_conv, in_channels, hidden_channels, dilation, norm_act) for _ in range(2)
        ])
        self.pre_process = nn.ModuleList([
            self._3x3box(self._seperable_conv, 128, 128, dilation, norm_act) for _ in range(2)
        ])
        self.conv_sem = nn.Conv2d(hidden_channels * levels, num_classes, 1)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain(self.output_1[0].bn2.activation, self.output_1[0].bn2.activation_param)
        for name, mod in self.named_modules():
            if isinstance(mod, nn.Conv2d):
                if "conv_sem" not in name:
                    nn.init.xavier_normal_(mod.weight, gain)
                else:
                    nn.init.xavier_normal_(mod.weight, .1)
            elif isinstance(mod, ABN):
                nn.init.constant_(mod.weight, 1.)
            if hasattr(mod, "bias") and mod.bias is not None:
                nn.init.constant_(mod.bias, 0.)

    def _getBBoxIdxs(self, bbx, bbx_gt):
        bbx_gt_idxs = torch.zeros(bbx_gt.shape[0], dtype=torch.int64, device=bbx[0].device)
        start = 0
        for idx, bbx_i in enumerate(bbx):
            bbx_gt_idxs[start:start + bbx_i.shape[0]] = idx
            start += bbx_i.shape[0]
        return bbx_gt_idxs

    def _rois(self, x, proposals, proposals_idx, img_size):
        stride = proposals.new([fs / os for fs, os in zip(x.shape[-2:], img_size)])
        proposals = (proposals - 0.5) * stride.repeat(2) + 0.5
        return roi_sampling(x, proposals, proposals_idx, self.roi_size)

    def _roi_head(self, sem_feat, bbx, img_size):
        batch_size = len(bbx)
        bbx_ps = PackedSequence(bbx)
        if bbx_ps.all_none:
            roi_logits = [None] * batch_size
        else:
            bbx_gt, bbx_gt_idx = bbx_ps.contiguous
            rois = self._rois(sem_feat, bbx_gt, bbx_gt_idx, img_size)
            roi_logits = self.conv_sem(rois)
            roi_logits = self._makeBatchList(roi_logits, bbx_gt_idx, batch_size)

        return roi_logits

    def _makeBatchList(self, roi_logits, bbx_gt_idx, batch_size):
        roi_logits_list = []
        unique_idxs = torch.unique(bbx_gt_idx)
        for entry in range(batch_size):
            if torch.sum(unique_idxs == entry) == 0:
                roi_logits_list.append(None)
            else:
                mask = (bbx_gt_idx == entry)
                roi_logits_list.append(roi_logits[mask, ...])

        return roi_logits_list

    def forward(self, xs, bbx, img_size, roi=False):
        xs = xs[self.min_level:self.min_level + self.levels]

        ref_size = xs[0].shape[-2:]
        interp_params = {"mode": self.interpolation}
        if self.interpolation == "bilinear":
            interp_params["align_corners"] = False

        i = self.min_level + self.levels - 1
        js = 0
        for output in self.output_1:
            xs[i] = output(xs[i])
            i = i - 1
        interm = self.pre_process[js](
            xs[i + 1] + functional.interpolate(xs[i + 2], size=xs[i + 1].shape[-2:], **interp_params))
        for output in self.output_2:
            xs[i] = output(xs[i])
            if js == 1:
                interm = self.pre_process[js](xs[i + 1])

            xs[i] = xs[i] + functional.interpolate(interm, size=xs[i].shape[-2:], **interp_params)
            js += 1
            i = i - 1
        for i in range(self.min_level, self.min_level + self.levels):
            if i > 0:
                xs[i] = functional.interpolate(xs[i], size=ref_size, **interp_params)

        sem_feat = torch.cat(xs, dim=1)
        xs = self.conv_sem(sem_feat)

        if roi and bbx is not None:
            roi_logits = self._roi_head(sem_feat, bbx, img_size)
        else:
            roi_logits = None

        return xs, sem_feat, roi_logits
