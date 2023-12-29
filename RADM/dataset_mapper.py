# ========================================
# Modified by Fengheng Li
# ========================================
# Modified by Shoufa Chen
# ========================================
# Modified by Peize Sun, Rufeng Zhang
# Contact: {sunpeize, cxrfzhang}@foxmail.com
#
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import copy
import logging
import numpy as np
import torch
import os
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T


__all__ = ["RADMDatasetMapper"]


def build_transform_gen(cfg, is_train):
    """
    Create a list of :class:`TransformGen` from config.
    Returns:
        list[TransformGen]
    """
    if is_train:
        min_size = cfg.INPUT.MIN_SIZE_TRAIN
        max_size = cfg.INPUT.MAX_SIZE_TRAIN
        sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
    else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
    if sample_style == "range":
        assert len(min_size) == 2, "more than 2 ({}) min_size(s) are provided for ranges".format(len(min_size))

    logger = logging.getLogger(__name__)
    tfm_gens = []
    if is_train:
        tfm_gens.append(T.RandomFlip())
    # ResizeShortestEdge
    tfm_gens.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

    if is_train:
        logger.info("TransformGens used in training: " + str(tfm_gens))
    return tfm_gens


class RADMDatasetMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by RADM.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    def __init__(self, cfg, is_train=True):
        if cfg.INPUT.CROP.ENABLED and is_train:
            self.crop_gen = [
                T.ResizeShortestEdge([400, 500, 600], sample_style="choice"),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
            ]
        else:
            self.crop_gen = None

        self.tfm_gens = build_transform_gen(cfg, is_train)
        logging.getLogger(__name__).info(
            "Full TransformGens used in training: {}, crop: {}".format(str(self.tfm_gens), str(self.crop_gen))
        )

        if is_train:
            self.text_feature_dir = os.path.join(cfg.DATASETS.TEXT_FEATURE_PATH, 'train')
        else:
            self.text_feature_dir = os.path.join(cfg.DATASETS.TEXT_FEATURE_PATH, 'test')

        self.img_format = cfg.INPUT.FORMAT
        self.is_train = is_train
        self.proposal_num = cfg.MODEL.RADM.NUM_PROPOSALS
        
    def load_text(self, text_name, max_text_num = 20):

        fea_path = os.path.join(
            self.text_feature_dir, text_name.split('.')[0].split('/')[-1]+'_feats.pth'
            )
        
        if os.path.exists(fea_path):
            text_fea = torch.load(fea_path)
            text_num = len(text_fea['feats'])
            text_mask = torch.full((text_num,1), False)
            padding = torch.full((max_text_num - text_num, 1), True)
            text_mask = torch.cat((text_mask, padding), dim=0)
            for i in range(text_num):
                text_fea['feats'][i] = text_fea['feats'][i].to('cpu')
            for i in range(max_text_num - text_num):
                text_fea['feats'].append(torch.zeros(1, 768))
            text_fea['feats'] = torch.cat(text_fea['feats'], dim=0)# (3, 768)
            # print('load text features {}'.format(text_name))
        else:
            # print('text features {} not found'.format(text_name))
            text_fea = {'feats': torch.zeros(max_text_num, 768)}
            text_mask = torch.full((max_text_num, 1), True)
        return text_fea, ~text_mask
    
    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        image = utils.read_image(dataset_dict["file_name"], format=self.img_format)
    
        utils.check_image_size(dataset_dict, image)
        if self.crop_gen is None:
            image, transforms = T.apply_transform_gens(self.tfm_gens, image)
        else:
            if np.random.rand() > 0.5:
                image, transforms = T.apply_transform_gens(self.tfm_gens, image)
            else:
                image, transforms = T.apply_transform_gens(
                    self.tfm_gens[:-1] + self.crop_gen + self.tfm_gens[-1:], image
                )

        image_shape = image.shape[:2]  # h, w

        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)))

        # text_fea: {'pos':[], 'feats':[]}
        # 'pos': [[x1, y1, x2, y2]...[]]
        # 'feats': [[feature]...[]] feature (1,768)
        text_fea, text_mask = self.load_text(dataset_dict["file_name"]) 
        dataset_dict['text_fea'] = text_fea
        dataset_dict['text_mask'] = text_mask
        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            # USER: Modify this if you want to keep them for some reason.
            for anno in dataset_dict["annotations"]:
                anno.pop("segmentation", None)
                anno.pop("keypoints", None)

            # USER: Implement additional transformations if you have other types of data
            annos = [
                utils.transform_instance_annotations(obj, transforms, image_shape)
                for obj in dataset_dict.pop("annotations")
                if obj.get("iscrowd", 0) == 0
            ]
            instances = utils.annotations_to_instances(annos, image_shape)
            dataset_dict["instances"] = utils.filter_empty_instances(instances)
        return dataset_dict
