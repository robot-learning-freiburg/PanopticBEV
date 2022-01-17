import torch.nn as nn
from torch.nn import functional as F
import torch
import kornia
from inplace_abn import ABN

from panoptic_bev.utils.transformer import get_init_homography


class VerticalTransformer(nn.Module):
    def __init__(self, in_ch, v_2d_ch, v_3d_ch, img_size_in, img_size_out, extents, img_scale, resolution, norm_act=ABN):
        super(VerticalTransformer, self).__init__()

        H_in = int(img_size_in[0] * img_scale)
        self.Z_out = Z_out = int(img_size_out[0] * img_scale)

        self.ch_mapper_in = nn.Sequential(nn.Conv2d(in_ch, v_2d_ch, 1, padding=0, bias=False),
                                          norm_act(v_2d_ch))

        self.depth_extender = nn.Sequential(nn.Conv3d(1, Z_out, 3, 1, 1),
                                            norm_act(Z_out))
        self.height_flattener = nn.Sequential(nn.Conv3d(H_in, 1, 3, 1, 1),
                                              norm_act(1))

        self.ch_mapper_out = nn.Sequential(nn.Conv2d(v_2d_ch, in_ch, 1, padding=0, bias=False),
                                           norm_act(in_ch))

        # Supervised Spatial Attention
        self.depth_estimation = nn.Sequential(nn.Conv2d(in_ch, Z_out, 3, 1, 1, bias=False),
                                              norm_act(Z_out),
                                              nn.Conv2d(Z_out, Z_out, 3, 1, 1, bias=False),
                                              norm_act(Z_out),
                                              nn.Conv2d(Z_out, Z_out, 3, 1, 1, bias=False),
                                              norm_act(Z_out))

        self.v_region_estimation = nn.Sequential(nn.Conv2d(H_in, H_in, 3, padding=1),
                                                 norm_act(H_in),
                                                 nn.Conv2d(H_in, 1, 1, 1, padding=0))

        # Function to unwarp the perspective distortion of vertical features
        self.warper = Perspective2OrthographicWarper(extents, img_scale, resolution)

        # Fixed dummy convolutions to prevent ABNSync from crashing on backward()
        self.depth_extender_dummy = nn.Conv3d(Z_out, Z_out, 1, padding=0, bias=False)
        self.depth_estimation_dummy = nn.Conv2d(Z_out, Z_out, 1, padding=0, bias=False)

    def forward(self, v_feat, intrinsics):
        
        # Generate the depth-based spatial occupancy map (attention map)
        v_depth_feat = self.depth_estimation_dummy(self.depth_estimation(v_feat))
        v_depth_feat = v_depth_feat.permute(0, 2, 1, 3).contiguous()
        v_region_logits = self.v_region_estimation(v_depth_feat)
        v_region_attention = v_region_logits.sigmoid().unsqueeze(1)
        del v_depth_feat

        v_feat = self.ch_mapper_in(v_feat)

        # Transform the vertical features in the FV to the BEV
        v_feat = v_feat.unsqueeze(1)  # Add a new dimension for applying 3D convolutions
        v_feat = self.depth_extender_dummy(self.depth_extender(v_feat))  # Create the volumetric lattice. Shape --> N x 1 x C_2d x H x W
        B, C_3d, C_2d, _, W = v_feat.shape  # Shape --> N x C_3d x C_2d x H x W
        v_feat = v_feat.permute(0, 2, 3, 1, 4).contiguous()  # Shape --> N x C_2d x H x C_3d x W
        v_feat = v_feat * v_region_attention  # Apply the depth-based spatial attention mask across all channels
        v_feat = v_feat.permute(0, 2, 1, 3, 4).contiguous()  # Shape --> N x H x C_2d x C_3d x W
        v_feat = self.height_flattener(v_feat)  # Flatten the volumetric lattice. Shape --> N x 1 x C_2d X C_3d x W
        v_feat = v_feat.squeeze(1)

        v_feat = self.ch_mapper_out(v_feat)

        # Unwarp the vertical features using the known camera intrinsics
        v_feat = self.warper(v_feat, intrinsics)
        v_region_logits = self.warper(v_region_logits, intrinsics)

        # Flip the features to align it for future merge with the flat features
        v_feat = torch.flip(v_feat, dims=[3])
        v_region_logits = torch.flip(v_region_logits, dims=[3])

        return v_feat, v_region_logits


class ErrorCorrectionModule(nn.Module):
    def __init__(self, in_ch, f_2d_ch, img_size_in, img_size_out, img_scale, norm_act=ABN):
        super(ErrorCorrectionModule, self).__init__()
        H_in = img_size_in[0] * img_scale
        Z_out = img_size_out[0] * img_scale

        self.feat_flatten = nn.Conv2d(int(H_in), int(f_2d_ch), 3, padding=1, bias=False)
        self.bottleneck_conv = nn.Conv2d(int(f_2d_ch), int(f_2d_ch), 3, padding=1, bias=False)
        self.feat_expand = nn.Conv2d(int(f_2d_ch), int(Z_out), 3, padding=1, bias=False)
        self.bev_conv = nn.Conv2d(int(f_2d_ch), int(f_2d_ch), 3, padding=1, bias=False)

    def forward(self, f_feat):
        B, C, _, W = f_feat.shape
        f_feat = f_feat.permute(0, 2, 1, 3).contiguous()
        f_feat = self.feat_flatten(f_feat)
        f_feat = f_feat.permute(0, 2, 1, 3).contiguous()
        f_feat = self.bottleneck_conv(f_feat)
        f_feat = self.feat_expand(f_feat)
        f_feat = f_feat.permute(0, 2, 1, 3).contiguous()
        f_feat = self.bev_conv(f_feat)

        return f_feat


class FlatTransformer(nn.Module):
    def __init__(self, in_ch, f_2d_ch, extrinsics=None, bev_params=None, in_img_size=None, out_img_size=None,
                 img_scale=None, extents=None, resolution=None, norm_act=ABN):
        super(FlatTransformer, self).__init__()
        self.f_2d_ch = f_2d_ch
        self.img_scale = img_scale
        self.fc_shape = list(in_img_size)
        self.extrinsics = extrinsics
        self.bev_params = bev_params
        self.in_img_size = in_img_size
        self.out_img_size_reverse = (out_img_size[1], out_img_size[0])  # (Z_out, W_out)
        self.W_out = out_img_size[0] * img_scale
        self.Z_out = out_img_size[1] * img_scale

        self.ch_mapper_in = nn.Sequential(nn.Conv2d(in_ch, f_2d_ch, 1, padding=0, bias=False),
                                          norm_act(f_2d_ch))

        self.ch_mapper_out = nn.Sequential(nn.Conv2d(f_2d_ch, in_ch, 1, padding=0, bias=False),
                                           norm_act(in_ch))

        # Account for the errors made by the IPM part of the flat transformer
        self.ecm = ErrorCorrectionModule(f_2d_ch, f_2d_ch, in_img_size, out_img_size, img_scale, norm_act)

        self.f_region_estimation = nn.Sequential(nn.Conv2d(f_2d_ch, f_2d_ch, 3, padding=1),
                                                 norm_act(f_2d_ch),
                                                 nn.Conv2d(f_2d_ch, 1, 1, 1, padding=0))

        self.ipm_confident_region_estimation = nn.Sequential(nn.Conv2d(f_2d_ch, f_2d_ch, 3, padding=1),
                                                             norm_act(f_2d_ch),
                                                             nn.Conv2d(f_2d_ch, 1, 1, 1, padding=0))

        self.post_process_residual = nn.Sequential(nn.Conv2d(f_2d_ch, f_2d_ch, 3, padding=1),
                                          norm_act(f_2d_ch),
                                          nn.Conv2d(f_2d_ch, f_2d_ch, 3, padding=1),
                                          norm_act(f_2d_ch),
                                          nn.Conv2d(f_2d_ch, f_2d_ch, 3, padding=1))

        self.f_dummy = nn.Conv2d(f_2d_ch, f_2d_ch, 1, bias=False)

        self.warper = Perspective2OrthographicWarper(extents, img_scale, resolution)

    def forward(self, feat, intrinsics):
        feat = self.f_dummy(self.ch_mapper_in(feat))

        # Apply the IPM algorithm on the flat regions.
        # We compute the homography using the known camera intrinsics and kornia applies the homography
        theta_ipm_list = []
        for b_idx in range(feat.shape[0]):
            theta_ipm_i = get_init_homography(intrinsics[b_idx].cpu().numpy(), self.extrinsics, self.bev_params,
                                              self.img_scale, self.out_img_size_reverse).view(-1, 3, 3).to(feat.device)
            theta_ipm_list.append(theta_ipm_i)
        theta_ipm = torch.cat(theta_ipm_list, dim=0)
        feat_bev_ipm = kornia.warp_perspective(feat, theta_ipm, (int(self.Z_out), int(self.W_out)))
        feat_bev_ipm = torch.rot90(feat_bev_ipm, k=2, dims=[2, 3])

        # Find the regions where IPM goes wrong and apply the ECN to those regions
        ipm_f_logits = self.ipm_confident_region_estimation(feat_bev_ipm)  # Get the logits where the IPM is "confident"
        ipm_incorrect = 1 - ipm_f_logits.sigmoid()  # Get the mask where the IPM is assumed to be incorrect

        # Convert the incorrect mask back into the FV and use it to get the erroneous features in the FV
        ipm_incorrect = torch.rot90(ipm_incorrect, k=2, dims=[2, 3])
        ipm_incorrect_fv = kornia.warp_perspective(ipm_incorrect, torch.inverse(theta_ipm), (feat.shape[2], feat.shape[3]))
        feat_ecm_fv = (feat * ipm_incorrect_fv)

        # Add the regions that are ignored by the IPM algorithm --> Regions above the principal point.
        # These points have the ipm_incorrect_fv value to 0. Just use that as a mask.
        ignored_by_ipm_mask = (ipm_incorrect_fv == 0)
        feat = feat * ignored_by_ipm_mask
        feat_ecm_fv = feat_ecm_fv + feat
        del feat

        # Apply the ErrorCorrectionModule to the regions where IPM made a mistake
        feat_bev_ecm = self.ecm(feat_ecm_fv)
        feat_bev_ecm = self.warper(feat_bev_ecm, intrinsics)
        feat_bev_ecm = F.interpolate(feat_bev_ecm, size=(int(self.Z_out), int(self.W_out)), mode="bilinear", align_corners=True)
        feat_bev_ecm = torch.flip(feat_bev_ecm, dims=[3])

        f_feat = feat_bev_ecm + feat_bev_ipm
        f_feat = f_feat + self.post_process_residual(f_feat)
        f_logits = self.f_region_estimation(f_feat)
        f_feat = self.ch_mapper_out(f_feat)

        return f_feat, f_logits


class MergeFeaturesVF(nn.Module):
    def __init__(self, in_ch, norm_act=ABN):
        super(MergeFeaturesVF, self).__init__()

        self.preprocess_v = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False),
                                          norm_act(in_ch))

        self.preprocess_f = nn.Sequential(nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=False),
                                          norm_act(in_ch))

        self.merge_vf = nn.Sequential(nn.Conv2d(2 * in_ch, in_ch, 3, 1, 1, bias=False),
                                      norm_act(in_ch))

    def forward(self, feat_v, feat_f):
        pp_v = self.preprocess_v(feat_v)
        pp_f = self.preprocess_f(feat_f)

        cat_vf = torch.cat((pp_v, pp_f), dim=1)
        merge_vf = self.merge_vf(cat_vf)

        return merge_vf


class Perspective2OrthographicWarper(nn.Module):

    def __init__(self, extents, img_scale, resolution):
        super().__init__()
        self.img_scale = img_scale

        # Store z positions of the near and far planes
        self.near = extents[1]
        self.far = extents[3]

        # Make a grid in the x-z plane
        self.grid = self._make_grid(extents, resolution)

    def forward(self, features, intrinsics):
        # Get the intrinsics for the current scale
        intrinsics_scale = intrinsics * self.img_scale
        intrinsics_scale[:, 2, 2] = 1  # The last row doesn't change anyway

        # Copy grid to the correct device
        self.grid = self.grid.to(features)

        # We ignore the image v-coordinate, and assume the world Y-coordinate
        # is zero, so we only need a 2x2 submatrix of the original 3x3 matrix
        intrinsics_scale = intrinsics_scale[:, [0, 2]][..., [0, 2]].view(-1, 1, 1, 2, 2)

        # Transform grid center locations into image u-coordinates
        cam_coords = torch.matmul(intrinsics_scale, self.grid.unsqueeze(-1)).squeeze(-1)

        # Apply perspective projection and normalize
        ucoords = cam_coords[..., 0] / (cam_coords[..., 1] + 1e-8)
        ucoords = ucoords / features.size(-1) * 2 - 1  # Normalise to [-1, 1]

        # Normalize z coordinates
        zcoords = (cam_coords[..., 1] - self.near) / (self.far - self.near) * 2 - 1

        # Resample 3D feature map
        grid_coords = torch.stack([ucoords, zcoords], -1).clamp(-1.1, 1.1)
        return F.grid_sample(features, grid_coords)


    def _make_grid(self, extents, resolution):
    # Create a grid of coordinates in the birds-eye-view
        x1, z1, x2, z2 = extents
        zz, xx = torch.meshgrid(torch.arange(z1, z2, resolution), torch.arange(x1, x2, resolution))

        return torch.stack([xx, zz], dim=-1)


class TransformerVF(nn.Module):
    def __init__(self, in_ch, tfm_ch, out_ch,  extrinsics=None, bev_params=None, H_in=None, W_in=None, Z_out=None,
                 W_out=None, img_scale=None, norm_act=ABN):
        super(TransformerVF, self).__init__()
        self.img_scale = img_scale

        self.ch_mapper_in = nn.Sequential(nn.Conv2d(in_ch, tfm_ch, 1, padding=0, bias=False),
                                          norm_act(tfm_ch))

        self.vf_estimation = nn.Sequential(nn.Conv2d(tfm_ch, tfm_ch//2, 1, 1, padding=0, bias=False),
                                           norm_act(tfm_ch//2),
                                           nn.Conv2d(tfm_ch//2, tfm_ch//2, 3, 1, padding=1, bias=False),
                                           norm_act(tfm_ch//2),
                                           nn.Conv2d(tfm_ch // 2, tfm_ch // 2, 3, 1, padding=1, bias=False),
                                           norm_act(tfm_ch // 2),
                                           nn.Conv2d(tfm_ch//2, 2, 1, 1, padding=0, bias=False))

        self.Z_out = int(Z_out * img_scale)
        self.W_out = int(W_out * img_scale)
        resolution = bev_params['cam_z'] / bev_params['f'] / img_scale
        # This is the output mask grid which will be used to correct the perspective distortion.
        # (W_out * img_scale) to bring it the current scale. (* resolution) to convert it into metres
        # /2 to centre the car in the image vertically
        extents = [-(self.W_out * resolution / 2), 0, (self.W_out * resolution / 2), self.Z_out * resolution]

        self.v_transform = VerticalTransformer(in_ch=tfm_ch, v_2d_ch=tfm_ch // 2, v_3d_ch=tfm_ch // 2, img_size_in=(H_in, W_in),
                                               img_size_out=(Z_out, W_out), extents=extents, img_scale=img_scale,
                                               resolution=resolution, norm_act=norm_act)
        self.f_transform = FlatTransformer(tfm_ch, tfm_ch, extrinsics, bev_params, (H_in, W_in), (W_out, Z_out),
                                           img_scale, extents, resolution, norm_act)

        self.merge_feat_vf = MergeFeaturesVF(tfm_ch, norm_act=norm_act)

        self.ch_mapper_out = nn.Sequential(nn.Conv2d(tfm_ch, out_ch, 1, padding=0, bias=False),
                                           norm_act(out_ch))

        # Placeholder to prevent ABNSync from crashing on backward()
        self.dummy = nn.Conv2d(out_ch, out_ch, 1, padding=0, bias=False)


    def forward(self, feat, intrinsics):
        feat = self.ch_mapper_in(feat)

        # Compute the vertical-flat attention mask
        # Vertical = Channel 0; Flat = Channel 1
        vf_logits = self.vf_estimation(feat)
        vf_softmax = vf_logits.softmax(dim=1)
        v_att = vf_softmax[:, 0, :, :].unsqueeze(1)
        f_att = vf_softmax[:, 1, :, :].unsqueeze(1)

        # Get the vertical and flat features by applying the generated attention masks to the frontal-view features
        feat_v = feat * v_att
        feat_f = feat * f_att

        del v_att, f_att

        # Perform the transformations on vertical and flat regions of the image plane feature map
        feat_v, v_region_logits = self.v_transform(feat_v, intrinsics)
        feat_f, f_region_logits = self.f_transform(feat_f, intrinsics)

        # Resize the feature maps to the output size
        # This takes into account the extreme cases where one dimension is a few pixels short
        feat_v = F.interpolate(feat_v, (self.Z_out, self.W_out), mode="bilinear", align_corners=True)
        feat_f = F.interpolate(feat_f, (self.Z_out, self.W_out), mode="bilinear", align_corners=True)

        # Merge the vertical and flat transforms
        feat_merged = self.merge_feat_vf(feat_v, feat_f)
        feat_merged = self.dummy(self.ch_mapper_out(feat_merged))

        del feat_v, feat_f

        # In the merged features, the ego car is at the bottom and something far away is at the top.
        # Rotate it to match the output --> The ego car is on the left and something far away is towards the right
        feat_merged = torch.rot90(feat_merged, k=1, dims=[2, 3])
        v_region_logits = torch.rot90(v_region_logits, k=1, dims=[2, 3])
        f_region_logits = torch.rot90(f_region_logits, k=1, dims=[2, 3])

        return feat_merged, vf_logits, v_region_logits, f_region_logits
