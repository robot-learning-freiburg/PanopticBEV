import torch
import torch.nn as nn
from inplace_abn import ABN

from panoptic_bev.modules.transformer import TransformerVF


class MultiScaleTransformerVF(nn.Module):
    """
    Apply the TransformerVF spatial transformer to each of the four feature scales.
    """

    def __init__(self, in_ch, tfm_ch, out_ch, extrinsics=None, bev_params=None, H_in=None, W_in=None, W_out=None,
                 Z_out=None, tfm_scales=None, use_init_theta=None, norm_act=ABN):
        super(MultiScaleTransformerVF, self).__init__()

        self.transformer_list = nn.ModuleList()

        for scale_idx, scale in enumerate(tfm_scales):
            if use_init_theta:
                transformer = TransformerVF(in_ch, tfm_ch, out_ch, extrinsics, bev_params,
                                            H_in=H_in, W_in=W_in, Z_out=Z_out, W_out=W_out, img_scale=1/scale,
                                            norm_act=norm_act)
            else:
                transformer = TransformerVF(in_ch, tfm_ch, out_ch, bev_params=bev_params,
                                            H_in=H_in, W_in=W_in, Z_out=Z_out, W_out=W_out, img_scale=1/scale,
                                            norm_act=norm_act)

            self.transformer_list.append(transformer)

        self.reset_parameters()

    def reset_parameters(self):
        def init_weights(m):
            if type(m) == nn.Conv2d or type(m) == nn.ConvTranspose2d:
                torch.nn.init.xavier_uniform_(m.weight)

        for idx in range(len(self.transformer_list)):
            self.transformer_list[idx].apply(init_weights)

    def forward(self, ms_feat, intrinsics):
        # Run the multi-scale features from each camera
        ms_feat_trans = []
        vf_logits_list = []
        v_region_logits_list = []
        f_region_logits_list = []

        for idx, (feat, transformer) in enumerate(zip(ms_feat, self.transformer_list)):
            bev_feat, vf_logits, v_region_logits, f_region_logits = transformer(feat, intrinsics)
            del feat

            ms_feat_trans.append(bev_feat)
            vf_logits_list.append(vf_logits)
            v_region_logits_list.append(v_region_logits)
            f_region_logits_list.append(f_region_logits)
        del ms_feat

        return ms_feat_trans, vf_logits_list, v_region_logits_list, f_region_logits_list,
