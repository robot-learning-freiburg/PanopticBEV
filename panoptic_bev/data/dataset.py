import glob
from itertools import chain
import os
import cv2
import torch
import torch.utils.data as data
import umsgpack
import json
from panoptic_bev.data.transform import *


class BEVKitti360Dataset(data.Dataset):
    _IMG_DIR = "img"
    _BEV_MSK_DIR = "bev_msk"
    _FRONT_MSK_DIR = "front_msk_trainid"
    _WEIGHTS_MSK_DIR = "class_weights"
    _BEV_DIR = "bev_ortho"
    _LST_DIR = "split"
    _METADATA_FILE = "metadata_ortho.bin"

    def __init__(self, seam_root_dir, dataset_root_dir, split_name, transform):
        super(BEVKitti360Dataset, self).__init__()
        self.seam_root_dir = seam_root_dir
        self.kitti_root_dir = dataset_root_dir
        self.split_name = split_name
        self.transform = transform
        self.rgb_cameras = ['front']

        # Folders
        self._img_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._IMG_DIR)
        self._bev_msk_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._BEV_MSK_DIR, BEVKitti360Dataset._BEV_DIR)
        self._front_msk_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._FRONT_MSK_DIR, "front")
        self._weights_msk_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._WEIGHTS_MSK_DIR)
        self._lst_dir = os.path.join(seam_root_dir, BEVKitti360Dataset._LST_DIR)

        # Load meta-data and split
        self._meta, self._images, self._img_map = self._load_split()

    # Load the train or the validation split
    def _load_split(self):
        with open(os.path.join(self.seam_root_dir, BEVKitti360Dataset._METADATA_FILE), "rb") as fid:
            metadata = umsgpack.unpack(fid, encoding="utf-8")

        with open(os.path.join(self._lst_dir, self.split_name + ".txt"), "r") as fid:
            lst = fid.readlines()
            lst = [line.strip() for line in lst]

        # Remove elements from lst if they are not in _FRONT_MSK_DIR
        front_msk_frames = os.listdir(self._front_msk_dir)
        front_msk_frames = [frame.split(".")[0] for frame in front_msk_frames]
        lst = [entry for entry in lst if entry in front_msk_frames]
        lst = set(lst)  # Remove any potential duplicates

        img_map = {}
        for camera in self.rgb_cameras:
            with open(os.path.join(self._img_dir, "{}.json".format(camera))) as fp:
                map_list = json.load(fp)
                map_dict = {k: v for d in map_list for k, v in d.items()}
                img_map[camera] = map_dict

        meta = metadata["meta"]
        images = [img_desc for img_desc in metadata["images"] if img_desc["id"] in lst]

        return meta, images, img_map

    def _load_item(self, item_idx):
        img_desc = self._images[item_idx]
        scene, frame_id = img_desc["id"].split(";")

        # Get the RGB file names
        img_file = [os.path.join(self.kitti_root_dir, self._img_map[camera]["{}.png".format(img_desc['id'])])
                    for camera in self.rgb_cameras]
        if all([(not os.path.exists(img)) for img in img_file]):
            raise IOError("RGB image not found! Scene: {}, Frame: {}".format(scene, frame_id))

        # Load the images
        img = [Image.open(rgb).convert(mode="RGB") for rgb in img_file]

        # Load the BEV mask
        bev_msk_file = os.path.join(self._bev_msk_dir, "{}.png".format(img_desc['id']))
        bev_msk = [Image.open(bev_msk_file)]

        # Load the front mask
        front_msk_file = os.path.join(self._front_msk_dir, "{}.png".format(img_desc['id']))
        front_msk = [Image.open(front_msk_file)]

        # Load the weight map
        weights_msk_file = os.path.join(self._weights_msk_dir, "{}.png".format(img_desc['id']))
        weights_msk = cv2.imread(weights_msk_file, cv2.IMREAD_UNCHANGED).astype(float)
        if weights_msk is not None:
            weights_msk_combined = (weights_msk[:, :, 0] + (weights_msk[:, :, 1] / 10000)) * 10000
            weights_msk_combined = [Image.fromarray(weights_msk_combined.astype(np.int32))]
        else:
            weights_msk_combined = None

        cat = img_desc["cat"]
        iscrowd = img_desc["iscrowd"]
        calib = img_desc['cam_intrinsic']
        return img, bev_msk, front_msk, weights_msk_combined, cat, iscrowd, calib, img_desc["id"]

    @property
    def categories(self):
        """Category names"""
        return self._meta["categories"]

    @property
    def num_categories(self):
        """Number of categories"""
        return len(self.categories)

    @property
    def num_stuff(self):
        """Number of "stuff" categories"""
        return self._meta["num_stuff"]

    @property
    def num_thing(self):
        """Number of "thing" categories"""
        return self.num_categories - self.num_stuff

    @property
    def original_ids(self):
        """Original class id of each category"""
        return self._meta["original_ids"]

    @property
    def palette(self):
        """Default palette to be used when color-coding semantic labels"""
        return np.array(self._meta["palette"], dtype=np.uint8)

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        return [img_desc["size"] for img_desc in self._images]

    @property
    def img_categories(self):
        """Categories present in each image of the dataset"""
        return [img_desc["cat"] for img_desc in self._images]

    @property
    def dataset_name(self):
        return "Kitti360"

    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        img, bev_msk, front_msk, weights_msk,cat, iscrowd, calib, idx = self._load_item(item)
        rec = self.transform(img=img, bev_msk=bev_msk, front_msk=front_msk, weights_msk=weights_msk, cat=cat,
                             iscrowd=iscrowd, calib=calib)
        size = (img[0].size[1], img[0].size[0])

        # Close the files
        for i in img:
            i.close()
        for m in bev_msk:
            m.close()
        for m in front_msk:
            m.close()

        rec["idx"] = idx
        rec["size"] = size
        return rec

    def get_image_desc(self, idx):
        """Look up an image descriptor given the id"""
        matching = [img_desc for img_desc in self._images if img_desc["id"] == idx]
        if len(matching) == 1:
            return matching[0]
        else:
            raise ValueError("No image found with id %s" % idx)


class BEVNuScenesDataset(data.Dataset):
    _IMG_DIR = "img"
    _BEV_MSK_DIR = "bev_msk"
    _FRONT_MSK_DIR = "front_msk_trainid"
    _VF_MSK_DIR = "vf_mask"
    _WEIGHTS_MSK_DIR = "class_weights"
    _BEV_DIR = "bev_ortho"
    _LST_DIR = "split"
    _METADATA_FILE = "metadata_ortho.bin"

    def __init__(self, seam_root_dir, dataset_root_dir, split_name, transform):
        super(BEVNuScenesDataset, self).__init__()
        self.seam_root_dir = seam_root_dir
        self.nuscenes_root_dir = dataset_root_dir
        self.split_name = split_name
        self.transform = transform
        self.rgb_cameras = ['front']

        # Folders
        self._img_dir = os.path.join(seam_root_dir, BEVNuScenesDataset._IMG_DIR)
        self._bev_msk_dir = os.path.join(seam_root_dir, BEVNuScenesDataset._BEV_MSK_DIR, BEVNuScenesDataset._BEV_DIR)
        self._front_msk_dir = os.path.join(seam_root_dir, BEVNuScenesDataset._FRONT_MSK_DIR, "front")
        self._weights_msk_dir = os.path.join(seam_root_dir, BEVNuScenesDataset._WEIGHTS_MSK_DIR)
        self._lst_dir = os.path.join(seam_root_dir, BEVNuScenesDataset._LST_DIR)
        self._vf_msk_dir = os.path.join(seam_root_dir, BEVNuScenesDataset._VF_MSK_DIR)

        # Load meta-data and split
        self._meta, self._images, self._img_map = self._load_split()

    # Load the train or the validation split
    def _load_split(self):
        with open(os.path.join(self.seam_root_dir, BEVNuScenesDataset._METADATA_FILE), "rb") as fid:
            metadata = umsgpack.unpack(fid, encoding="utf-8")

        with open(os.path.join(self._lst_dir, self.split_name + ".txt"), "r") as fid:
            lst = fid.readlines()
            lst = [line.strip() for line in lst]

        bev_msk_frames = os.listdir(self._bev_msk_dir)
        bev_msk_frames = [frame.split(".")[0] for frame in bev_msk_frames]
        lst = [entry for entry in lst if entry in bev_msk_frames]
        lst = set(lst)  # Remove any potential duplicates

        img_map = {}
        for camera in self.rgb_cameras:
            with open(os.path.join(self._img_dir, "{}.json".format(camera))) as fp:
                map_list = json.load(fp)
                map_dict = {k: v for d in map_list for k, v in d.items()}
                img_map[camera] = map_dict

        meta = metadata["meta"]
        images = [img_desc for img_desc in metadata["images"] if img_desc["id"] in lst]

        return meta, images, img_map

    def _load_item(self, item_idx):
        img_desc = self._images[item_idx]

        # Get the RGB file names
        img_file = [os.path.join(self.nuscenes_root_dir, self._img_map[camera]["{}.png".format(img_desc['id'])])
                    for camera in self.rgb_cameras]
        if all([(not os.path.exists(img)) for img in img_file]):
            raise IOError("RGB image not found! Name: {}".format(img_desc['id']))

        # Load the images
        img = [Image.open(rgb).convert(mode="RGB") for rgb in img_file]

        # Load the BEV mask
        bev_msk_file = os.path.join(self._bev_msk_dir, "{}.png".format(img_desc['id']))
        bev_msk = [Image.open(bev_msk_file)]

        # Load the VF mask
        vf_msk_file = os.path.join(self._vf_msk_dir, "{}.png".format(img_desc["id"]))
        vf_msk = [Image.open(vf_msk_file)]

        # Load the weight map
        weights_msk_file = os.path.join(self._weights_msk_dir, "{}.png".format(img_desc['id']))
        weights_msk = cv2.imread(weights_msk_file, cv2.IMREAD_UNCHANGED).astype(float)
        if weights_msk is not None:
            weights_msk_combined = (weights_msk[:, :, 0] + (weights_msk[:, :, 1] / 10000)) * 10000
            weights_msk_combined = [Image.fromarray(weights_msk_combined.astype(np.int32))]
        else:
            weights_msk_combined = None

        cat = img_desc["cat"]
        iscrowd = img_desc["iscrowd"]
        calib = img_desc['cam_intrinsic']
        return img, bev_msk, vf_msk, weights_msk_combined, cat, iscrowd, calib, img_desc["id"]

    @property
    def categories(self):
        """Category names"""
        return self._meta["categories"]

    @property
    def num_categories(self):
        """Number of categories"""
        return len(self.categories)

    @property
    def num_stuff(self):
        """Number of "stuff" categories"""
        return self._meta["num_stuff"]

    @property
    def num_thing(self):
        """Number of "thing" categories"""
        return self.num_categories - self.num_stuff

    @property
    def original_ids(self):
        """Original class id of each category"""
        return self._meta["original_ids"]

    @property
    def palette(self):
        """Default palette to be used when color-coding semantic labels"""
        return np.array(self._meta["palette"], dtype=np.uint8)

    @property
    def img_sizes(self):
        """Size of each image of the dataset"""
        return [img_desc["size"] for img_desc in self._images]

    @property
    def img_categories(self):
        """Categories present in each image of the dataset"""
        return [img_desc["cat"] for img_desc in self._images]

    @property
    def dataset_name(self):
        return "nuScenes"

    def __len__(self):
        return len(self._images)

    def __getitem__(self, item):
        img, bev_msk, vf_msk, wt_mask, cat, iscrowd, calib, idx = self._load_item(item)
        rec = self.transform(img=img, bev_msk=bev_msk, front_msk=vf_msk, weights_msk=wt_mask, cat=cat, iscrowd=iscrowd,
                             calib=calib)
        size = (img[0].size[1], img[0].size[0])

        # Close the files
        for i in img:
            i.close()
        for m in bev_msk:
            m.close()
        if vf_msk is not None:
            for m in vf_msk:
                m.close()

        rec["idx"] = idx
        rec["size"] = size
        return rec

    def get_image_desc(self, idx):
        """Look up an image descriptor given the id"""
        matching = [img_desc for img_desc in self._images if img_desc["id"] == idx]
        if len(matching) == 1:
            return matching[0]
        else:
            raise ValueError("No image found with id %s" % idx)
