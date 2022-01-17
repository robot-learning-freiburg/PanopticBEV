import random
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import functional as tfn
from torchvision.transforms import ColorJitter
import torchvision.transforms.functional as F

from panoptic_bev.utils.bbx import extract_boxes


################### START - COPY FROM OLDER PYTORCH VERSION FOR BACKWARD COMPATIBILITY ###################
class Lambda:
    """Apply a user-defined lambda as a transform. This transform does not support torchscript.
    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        if not callable(lambd):
            raise TypeError("Argument lambd should be callable, got {}".format(repr(type(lambd).__name__)))
        self.lambd = lambd

    def __call__(self, img):
        return self.lambd(img)

    def __repr__(self):
        return self.__class__.__name__ + '()'


class Compose:
    """Composes several transforms together. This transform does not support torchscript. """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            format_string += '\n'
            format_string += '    {0}'.format(t)
        format_string += '\n)'
        return format_string

################### END - COPY FROM OLDER PYTORCH VERSION FOR BACKWARD COMPATIBILITY ###################


class BEVTransform:
    def __init__(self,
                 shortest_size,
                 longest_max_size,
                 rgb_mean=None,
                 rgb_std=None,
                 front_resize=None,
                 bev_crop=None,
                 scale=None,
                 random_flip=False,
                 random_brightness=None,
                 random_contrast=None,
                 random_saturation=None,
                 random_hue=None):
        self.shortest_size = shortest_size
        self.longest_max_size = longest_max_size
        self.front_resize = front_resize
        self.bev_crop = bev_crop
        self.scale = scale
        self.rgb_mean = rgb_mean
        self.rgb_std = rgb_std
        self.random_flip = random_flip
        self.random_brightness = random_brightness
        self.random_contrast = random_contrast
        self.random_saturation = random_saturation
        self.random_hue = random_hue

    def _scale(self, img, bev_msk, front_msk, weights_msk):
        # Scale the image and the mask
        if img is not None:
            in_img_w, in_img_h = img[0].size[0], img[0].size[1]
            out_img_w, out_img_h = int(in_img_w * self.scale), int(in_img_h * self.scale)
            img = [rgb.resize((out_img_w, out_img_h)) for rgb in img]

        if bev_msk is not None:
            in_msk_w, in_msk_h = bev_msk[0].size[0], bev_msk[0].size[1]
            out_msk_w, out_msk_h = int(in_msk_w * self.scale), int(in_msk_h * self.scale)
            bev_msk = [m.resize((out_msk_w, out_msk_h), Image.NEAREST) for m in bev_msk]

        if front_msk is not None:
            in_msk_w, in_msk_h = front_msk[0].size[0], front_msk[0].size[1]
            out_msk_w, out_msk_h = int(in_msk_w * self.scale), int(in_msk_h * self.scale)
            front_msk = [m.resize((out_msk_w, out_msk_h), Image.NEAREST) for m in front_msk]

        if weights_msk is not None:
            in_msk_w, in_msk_h = weights_msk[0].size[0], weights_msk[0].size[1]
            out_msk_w, out_msk_h = int(in_msk_w * self.scale), int(in_msk_h * self.scale)
            weights_msk = [m.resize((out_msk_w, out_msk_h), Image.BILINEAR) for m in weights_msk]

        return img, bev_msk, front_msk, weights_msk

    def _resize(self, img, mode):
        if img is not None:
            # Resize the image
            out_img_w, out_img_h = self.front_resize[1], self.front_resize[0]
            img = [rgb.resize((out_img_w, out_img_h), mode) for rgb in img]

        return img

    def _crop(self, msk):
        if msk is not None:
            ip_height, ip_width = msk[0].size[1], msk[0].size[0]

            # Check that the crop dimensions are not larger than the input dimensions
            if self.bev_crop[0] > ip_height or self.bev_crop[1] > ip_width:
                raise RuntimeError("Crop dimensions need to be smaller than the input dimensions."
                                   "Crop: {}, Input: {}".format(self.bev_crop, (ip_height, ip_width)))

            # We want to crop from the centre
            min_row = 0
            max_row = self.bev_crop[0]
            min_col = 0
            max_col = self.bev_crop[1]

            # (Left, Top, Right, Bottom)
            msk_cropped = [m.crop((min_col, min_row, max_col, max_row)) for m in msk]
            return msk_cropped
        else:
            return msk

    @staticmethod
    def _random_flip(img, bev_msk, front_msk, weights_msk):
        if random.random() < 0.5:
            # Horizontally flip the RGB image and the front mask
            if img is not None:
                img = [rgb.transpose(Image.FLIP_LEFT_RIGHT) for rgb in img]
            if front_msk is not None:
                front_msk = [m.transpose(Image.FLIP_LEFT_RIGHT) for m in front_msk]

            # Flip the BEV panoptic mask. The mask is sideways, so that mask has to be flipped top-down
            if bev_msk is not None:
                bev_msk = [m.transpose(Image.FLIP_TOP_BOTTOM) for m in bev_msk]
            if weights_msk is not None:
                weights_msk = [m.transpose(Image.FLIP_TOP_BOTTOM) for m in weights_msk]

            return img, bev_msk, front_msk, weights_msk
        else:
            return img, bev_msk, front_msk, weights_msk

    def _normalize_image(self, img):
        if img is not None:
            if (self.rgb_mean is not None) and (self.rgb_std is not None):
                img.sub_(img.new(self.rgb_mean).view(-1, 1, 1))
                img.div_(img.new(self.rgb_std).view(-1, 1, 1))
        return img

    @staticmethod
    def _compact_labels(msk, cat, iscrowd):
        ids = np.unique(msk)
        if 0 not in ids:
            ids = np.concatenate((np.array([0], dtype=np.int32), ids), axis=0)

        ids_to_compact = np.zeros((ids.max() + 1,), dtype=np.int32)
        ids_to_compact[ids] = np.arange(0, ids.size, dtype=np.int32)

        msk = ids_to_compact[msk]
        cat = cat[ids]
        iscrowd = iscrowd[ids]

        return msk, cat, iscrowd

    @staticmethod
    def get_params(brightness, contrast, saturation, hue):
        """Get a randomized transform to be applied on image.
        Arguments are same as that of __init__.
        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def __call__(self, img, bev_msk, front_msk=None, weights_msk=None, cat=None, iscrowd=None, calib=None):
        # Random flip
        if self.random_flip:
            img, bev_msk, front_msk, weights_msk = self._random_flip(img, bev_msk, front_msk, weights_msk)

        # Crop the BEV mask to the required dimensions
        if self.bev_crop:
            bev_msk = self._crop(bev_msk)

        # Resize the RGB image and the front mask to the given dimension
        if self.front_resize:
            img = self._resize(img, Image.BILINEAR)
            front_msk = self._resize(front_msk, Image.NEAREST)

        # Scale the images and the mask to a smaller value
        if self.scale:
            img, bev_msk, front_msk, weights_msk = self._scale(img, bev_msk, front_msk, weights_msk)

        # Random Colour Jitter. Apply the same colour jitter to all the images
        if (self.random_brightness is not None) and (self.random_contrast is not None) \
                and (self.random_hue is not None) and (self.random_flip is not None):
            colour_jitter = ColorJitter(brightness=self.random_brightness, contrast=self.random_contrast,
                                        saturation=self.random_saturation, hue=self.random_hue)
            colour_jitter_transform = self.get_params(colour_jitter.brightness, colour_jitter.contrast,
                                                      colour_jitter.saturation, colour_jitter.hue)
            img = [colour_jitter_transform(rgb) for rgb in img]
  
        # Wrap in np.array
        if cat is not None:
            cat = np.array(cat, dtype=np.int32)
        if iscrowd is not None:
            iscrowd = np.array(iscrowd, dtype=np.uint8)

        # Adjust calib and wrap in np.array
        if calib is not None:
            calib = np.array(calib, dtype=np.float32)
            if len(calib.shape) == 3:
                calib[:, 0, 0] *= float(self.front_resize[1]) / self.longest_max_size
                calib[:, 1, 1] *= float(self.front_resize[0]) / self.shortest_size
                calib[:, 0, 2] *= float(self.front_resize[1]) / self.longest_max_size
                calib[:, 1, 2] *= float(self.front_resize[0]) / self.shortest_size
            else:
                calib[0, 0] *= float(self.front_resize[1]) / self.longest_max_size
                calib[1, 1] *= float(self.front_resize[0]) / self.shortest_size
                calib[0, 2] *= float(self.front_resize[1]) / self.longest_max_size
                calib[1, 2] *= float(self.front_resize[0]) / self.shortest_size

        # Image transformations
        img = [tfn.to_tensor(rgb) for rgb in img]
        img = [self._normalize_image(rgb) for rgb in img]
        # Concatenate the images to make it easier to process downstream
        img = torch.cat(img, dim=0)

        # Label transformations
        if bev_msk is not None:
            bev_msk = np.stack([np.array(m, dtype=np.int32, copy=False) for m in bev_msk], axis=0)
            bev_msk, cat, iscrowd = self._compact_labels(bev_msk, cat, iscrowd)

        if weights_msk is not None:
            weights_msk = [np.array(m, dtype=np.int32, copy=False) for m in weights_msk]
            weights_msk = np.stack(weights_msk, axis=0)

        if front_msk is not None:
            front_msk = np.stack([np.array(m, dtype=np.int32, copy=False) for m in front_msk], axis=0)

        # Convert labels to torch and extract bounding boxes
        if bev_msk is not None:
            bev_msk = torch.from_numpy(bev_msk.astype(np.long))
        if front_msk is not None:
            front_msk = torch.from_numpy(front_msk.astype(np.long))
        if weights_msk is not None:
            weights_msk = torch.from_numpy(weights_msk.astype(np.float))
        if cat is not None:
            cat = torch.from_numpy(cat.astype(np.long))
            bbx = extract_boxes(bev_msk, cat.numel())
        else:
            bbx = None
        if iscrowd is not None:
            iscrowd = torch.from_numpy(iscrowd)
        if calib is not None:
            calib = torch.from_numpy(calib)

        return dict(img=img, bev_msk=bev_msk, front_msk=front_msk, weights_msk=weights_msk, cat=cat, iscrowd=iscrowd,
                    bbx=bbx, calib=calib)
