# -*- coding : utf-8 -*-
# @FileName  : FSC147.py
# @Author    : Ruixiang JIANG (Songrise)
# @Time      : Aug 13, 2023
# @Github    : https://github.com/songrise
# @Description: FSC-147 dataset. Implementation borrowed from CounTR.
import json
import numpy as np
import random
from torchvision import transforms
import torch
import cv2
import torchvision.transforms.functional as TF
import scipy.ndimage as ndimage
from PIL import Image
from torch.utils.data import Dataset
import os
import imgaug as ia
import imgaug.augmenters as iaa
import pickle
from imgaug.augmentables import Keypoint, KeypointsOnImage
from transformers import CLIPTokenizer
import torchvision.transforms.functional as F
from .groundingDino import GetExampler
from .GroundingDINO.groundingdino.datasets import transforms as T
from typing import List, Tuple

from .GroundingDINO.groundingdino.util.inference import (
    load_model,
    load_image,
    predict,
    annotate
)

# import utils.debug_utils
MAX_HW = 384
IM_NORM_MEAN = [0.485, 0.456, 0.406]
IM_NORM_STD = [0.229, 0.224, 0.225]
BOX_THRESHOLD = 0
KEEP_AREA = 0.4

Normalize = transforms.Compose([   
        transforms.ToTensor(),
        transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)
        ])

class FSC147(Dataset):
    def __init__(self, config,
                 split:str , subset_scale: float = 1.0):
        """
        Parameters
        ----------
        split : str, 'train', 'val' or 'test'
        subset_scale : float, scale of the subset of the dataset to use
        resize_val : bool, whether to random crop validation images to 384x384
        """
        assert split in ['train', 'val', 'test' , 'val_coco', 'test_coco']

        #!HARDCODED Dec 25: 
        self.data_dir = config['training']['data_dir'] 
        self.dataset_type = config['training']['dataset_type'] # FSC147 
        additional_prompt = config['training'].get('additional_prompt', False)
        subset_scale = subset_scale

        self.resize_val = config['training']['resize_val'] if split == 'val' else False

        self.im_dir = os.path.join(self.data_dir,'images_384_VarV2')
        self.gt_dir = os.path.join(self.data_dir, 'gt_density_map_adaptive_384_VarV2')
        self.anno_file = os.path.join(self.data_dir,  f'annotation_FSC147_384.json')
        self.data_split_file = os.path.join(self.data_dir, f'Train_Test_Val_FSC_147.json')
        self.class_file = os.path.join(self.data_dir,f'ImageClasses_FSC147.txt')
        self.split = split
        with open(self.data_split_file) as f:
            data_split = json.load(f)

        with open(self.anno_file) as f:
            self.annotations = json.load(f)

        self.idx_running_set = data_split[split]
        # subsample the dataset
        self.idx_running_set = self.idx_running_set[:int(subset_scale*len(self.idx_running_set))]

        self.class_dict = {}
        with open(self.class_file) as f:
            for line in f:
                key = line.split()[0]
                val = line.split()[1:]
                # concat word as string
                val = ' '.join(val)
                self.class_dict[key] = val
        self.all_classes = list(set(self.class_dict.values()))
        self.transform = None
        if self.split == 'train' or (self.split == 'val' and self.resize_val):
            use_aug = self.split == 'train'
            self.transform = transforms.Compose([ResizeTrainImage(MAX_HW, self, aug=use_aug)])
        random.shuffle(self.idx_running_set)

        self.additional_prompt = None
        self.use_additional_prompt = additional_prompt
        if additional_prompt:
            additional_prompt_path = "util/CLIP_caption.pkl"
            with open(additional_prompt_path, 'rb') as f:
                self.additional_prompt = pickle.load(f)
        
            
    def __len__(self):
        return len(self.idx_running_set)

    def __getitem__(self, idx):
        im_id = self.idx_running_set[idx]
        anno = self.annotations[im_id]
        text = self.class_dict[im_id]
        if self.use_additional_prompt:
            additional_prompt = self.additional_prompt[im_id]
        bboxes = anno['box_examples_coordinates']
        if self.split == 'train' or (self.split == 'val' and self.resize_val):
            rects = list()
            for bbox in bboxes:
                x1 = bbox[0][0]
                y1 = bbox[0][1]
                x2 = bbox[2][0]
                y2 = bbox[2][1]
                rects.append([y1, x1, y2, x2])
            
            dots = np.array(anno['points'])

            img_path = '{}/{}'.format(self.im_dir, im_id)
            image = Image.open(img_path)
            image.load()
            density_path = self.gt_dir + '/' + im_id.split(".jpg")[0] + ".npy"
            density = np.load(density_path).astype('float32')   
            m_flag = 0

            sample = {'image':image,'lines_boxes':rects,'gt_density':density, 'dots':dots, 'id':im_id, 'm_flag': m_flag}
            sample = self.transform(sample)

            
            # img, den_map, prompt, prompt_attn_mask, img_attn_map, img_gd, img_src
            img_src, img_gd = load_image(img_path) 

            return sample['image'].float(), sample['gt_density'], text, im_id, img_gd, img_src
        elif self.split == "test" or self.split == "test_coco" or self.split == "val_coco" or (self.split == "val" and not self.resize_val):
            dots = np.array(anno['points'])
            img_path = '{}/{}'.format(self.im_dir, im_id)
            image = Image.open(img_path)
            text = self.class_dict[im_id]
            image.load()
            W, H = image.size

            new_H = 16*int(H/16)
            new_W = 16*int(W/16)
            scale_factor = float(new_W)/ W
            image = transforms.Resize((new_H, new_W))(image)
            Normalize = transforms.Compose([transforms.ToTensor()])
            image = Normalize(image)

            rects = list()
            for bbox in bboxes:
                x1 = int(bbox[0][0]*scale_factor)
                y1 = bbox[0][1]
                x2 = int(bbox[2][0]*scale_factor)
                y2 = bbox[2][1]
                rects.append([y1, x1, y2, x2])

            boxes = list()
            cnt = 0
            for box in rects:
                cnt+=1
                if cnt>3:
                    break
                box2 = [int(k) for k in box]
                y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
                bbox = image[:,y1:y2+1,x1:x2+1]
                bbox = transforms.Resize((64, 64))(bbox)
                boxes.append(bbox.numpy())


            # Only for visualisation purpose, no need for ground truth density map indeed.
            gt_map = np.zeros((image.shape[1], image.shape[2]),dtype='float32')
            for i in range(dots.shape[0]):
                gt_map[min(new_H-1,int(dots[i][1]))][min(new_W-1,int(dots[i][0]*scale_factor))]=1
            gt_map = ndimage.gaussian_filter(gt_map, sigma=(1, 1), order=0)
            gt_map = torch.from_numpy(gt_map)
            gt_map = gt_map * 60

            sample = {'image':image,'dots':dots, 'boxes':boxes, 'pos':rects, 'gt_map':gt_map, 'id':im_id}

            img_src, img_gd = load_image(img_path) 

            return sample['image'].float(), sample['gt_map'], text, im_id,  img_gd, img_src


class FSC147_pre_extracted(Dataset):
    def __init__(self, config,
                 split:str , subset_scale: float = 1.0):
        """
        Parameters
        ----------
        split : str, 'train', 'val' or 'test'
        subset_scale : float, scale of the subset of the dataset to use
        resize_val : bool, whether to random crop validation images to 384x384
        """
        assert split in ['train', 'val', 'test' , 'val_coco', 'test_coco']

        #!HARDCODED Dec 25: 
        self.data_dir = config['training']['data_dir'] 
        self.dataset_type = config['training']['dataset_type'] # FSC147 
        additional_prompt = config['training'].get('additional_prompt', False)
        subset_scale = subset_scale

        self.resize_val = config['training']['resize_val'] if split == 'val' else False

        self.im_dir = os.path.join(self.data_dir,'images_384_VarV2')
        self.gt_dir = os.path.join(self.data_dir, 'gt_density_map_adaptive_384_VarV2')
        self.anno_file = os.path.join(self.data_dir,  f'annotation_FSC147_384.json')
        self.data_split_file = os.path.join(self.data_dir, f'Train_Test_Val_FSC_147.json')
        self.class_file = os.path.join(self.data_dir,f'ImageClasses_FSC147.txt')
        self.extracted_dir = os.path.join(self.data_dir,'pre_extracted_path')

        self.split = split
        with open(self.data_split_file) as f:
            data_split = json.load(f)

        with open(self.anno_file) as f:
            self.annotations = json.load(f)

        self.idx_running_set = data_split[split]
        # subsample the dataset
        self.idx_running_set = self.idx_running_set[:int(subset_scale*len(self.idx_running_set))]

        self.class_dict = {}
        with open(self.class_file) as f:
            for line in f:
                key = line.split()[0]
                val = line.split()[1:]
                # concat word as string
                val = ' '.join(val)
                self.class_dict[key] = val
        self.all_classes = list(set(self.class_dict.values()))
        self.transform = None
        if self.split == 'train' or (self.split == 'val' and self.resize_val):
            use_aug = self.split == 'train'
            self.transform = transforms.Compose([ResizeTrainImage(MAX_HW, self, aug=use_aug)])
        random.shuffle(self.idx_running_set)

        self.additional_prompt = None
        self.use_additional_prompt = additional_prompt
        if additional_prompt:
            additional_prompt_path = "util/CLIP_caption.pkl"
            with open(additional_prompt_path, 'rb') as f:
                self.additional_prompt = pickle.load(f)
        
            
    def __len__(self):
        return len(self.idx_running_set)

    def __getitem__(self, idx):
        im_id = self.idx_running_set[idx]
        anno = self.annotations[im_id]
        text = self.class_dict[im_id]
        if self.use_additional_prompt:
            additional_prompt = self.additional_prompt[im_id]
        bboxes = anno['box_examples_coordinates']
        if self.split == 'train' or (self.split == 'val' and self.resize_val):
            rects = list()
            for bbox in bboxes:
                x1 = bbox[0][0]
                y1 = bbox[0][1]
                x2 = bbox[2][0]
                y2 = bbox[2][1]
                rects.append([y1, x1, y2, x2])
            
            dots = np.array(anno['points'])

            img_path = '{}/{}'.format(self.im_dir, im_id)
            image = Image.open(img_path)
            image.load()
            density_path = self.gt_dir + '/' + im_id.split(".jpg")[0] + ".npy"
            density = np.load(density_path).astype('float32')   
            m_flag = 0

            sample = {'image':image,'lines_boxes':rects,'gt_density':density, 'dots':dots, 'id':im_id, 'm_flag': m_flag}
            sample = self.transform(sample)

            
            extracted_img_path = '{}/{}'.format(self.extracted_dir, im_id)
            exampler = Image.open(extracted_img_path)
            exampler.load()
            W, H = image.size

            new_H = 16*int(H/16)
            new_W = 16*int(W/16)
            scale_factor = float(new_W)/ W
            exampler = transforms.Resize((new_H, new_W))(exampler)
            exampler = Normalize(exampler)



            return sample['image'].float(), sample['gt_density'], text, im_id, exampler
        elif self.split == "test" or self.split == "test_coco" or self.split == "val_coco" or (self.split == "val" and not self.resize_val):
            dots = np.array(anno['points'])
            img_path = '{}/{}'.format(self.im_dir, im_id)
            image = Image.open(img_path)
            text = self.class_dict[im_id]
            image.load()
            W, H = image.size

            new_H = 16*int(H/16)
            new_W = 16*int(W/16)
            scale_factor = float(new_W)/ W
            image = transforms.Resize((new_H, new_W))(image)
            Normalize1 = transforms.Compose([transforms.ToTensor()])
            image = Normalize1(image)

            rects = list()
            for bbox in bboxes:
                x1 = int(bbox[0][0]*scale_factor)
                y1 = bbox[0][1]
                x2 = int(bbox[2][0]*scale_factor)
                y2 = bbox[2][1]
                rects.append([y1, x1, y2, x2])

            boxes = list()
            cnt = 0
            for box in rects:
                cnt+=1
                if cnt>3:
                    break
                box2 = [int(k) for k in box]
                y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
                bbox = image[:,y1:y2+1,x1:x2+1]
                bbox = transforms.Resize((64, 64))(bbox)
                boxes.append(bbox.numpy())


            # Only for visualisation purpose, no need for ground truth density map indeed.
            gt_map = np.zeros((image.shape[1], image.shape[2]),dtype='float32')
            for i in range(dots.shape[0]):
                gt_map[min(new_H-1,int(dots[i][1]))][min(new_W-1,int(dots[i][0]*scale_factor))]=1
            gt_map = ndimage.gaussian_filter(gt_map, sigma=(1, 1), order=0)
            gt_map = torch.from_numpy(gt_map)
            gt_map = gt_map * 60

            sample = {'image':image,'dots':dots, 'boxes':boxes, 'pos':rects, 'gt_map':gt_map, 'id':im_id}

            extracted_img_path = '{}/{}'.format(self.extracted_dir, im_id)
            exampler = Image.open(extracted_img_path)
            exampler.load()
            W, H = image.size

            new_H = 16*int(H/16)
            new_W = 16*int(W/16)
            scale_factor = float(new_W)/ W
            exampler = transforms.Resize((new_H, new_W))(exampler)
            exampler = Normalize(exampler)


            return sample['image'].float(), sample['gt_map'], text, im_id,  exampler

def collate_fn_pre_extracted(batch):
    img, den_map, prompt, id,  exampler = zip(*batch)

    return {
        'image': torch.stack(img, 0),
        'density': torch.stack(den_map, 0),
        'img_id': id,
        'exampler': torch.stack(exampler, 0),
        'text': prompt
    }

def collate_fn(batch):
    img, den_map, prompt, id,  img_gd, img_src = zip(*batch)

    return {
        'image': torch.stack(img, 0),
        'density': torch.stack(den_map, 0),
        'img_id': id,
        'img_gd': img_gd,
        'img_src': img_src,
        'text': prompt
    }

def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w



# def load_image(image_path: str) -> Tuple[np.array, torch.Tensor]:
#     transform = T.Compose(
#         [
#             T.RandomResize([800], max_size=1333),
#             T.ToTensor(),
#             T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
#         ]
#     )
#     image_source = Image.open(image_path).convert("RGB")
#     image = np.asarray(image_source)
#     image_transformed, _ = transform(image_source, None)
#     return image, image_transformed

class ObjectCount(Dataset):
    def __init__(self, config, split = 'train'): # root, crop_size, downsample_ratio, method='train', concat_size=224
        super(ObjectCount, self).__init__()
        #self.im_list = sorted(glob(os.path.join(root, 'images/*.jpg')))
        crop_size = config['training'].get('crop_size', 384)
        downsample_ratio = config['training'].get('downsample_ratio', 4)
        concat_size = config['training'].get('concat_size', 224)

        assert crop_size % downsample_ratio == 0
        assert split in ['train', 'val', 'test'], f"Invalid method: {split}. Must be 'train', 'val', or 'test'."
        self.crop_size = crop_size
        self.down_ratio = downsample_ratio
        self.method = split
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
        self.concat_size = concat_size

        root = config['training']['data_dir']  #!HARDCODED Dec 25
        with open(os.path.join(root, 'Train_Test_Val_FSC_147.json'), 'r') as f:
            data_split = json.load(f)[split]
            self.im_list = [os.path.join(root, 'images_384_VarV2', x) for x in data_split]

        with open(os.path.join(root, 'annotation_FSC147_384.json'), 'r') as f:
            self.annotations = json.load(f)

        self.cls_dict = {}
        with open(os.path.join(root, 'ImageClasses_FSC147.txt'), "r", encoding="utf-8") as f:
            for line in f:
                self.cls_dict[line.strip().split('\t')[0]] = line.strip().split('\t')[1]
        self.im_list = self.im_list[:int(0.001 * len(self.im_list))]

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        im_path = self.im_list[item]
        den_path = im_path.replace('images_384_VarV2', 'gt_density_map_adaptive_384_VarV2').replace('jpg', 'npy')

        img = Image.open(im_path).convert('RGB')
        cls_name = self.cls_dict[os.path.basename(im_path)]
        pts = self.annotations[os.path.basename(im_path)]['points']

        prompt = cls_name

        prompt_attn_mask = torch.zeros(77)
        cls_name_tokens = self.tokenizer(cls_name, add_special_tokens=False, return_tensors='pt')
        cls_name_length = cls_name_tokens['input_ids'].shape[1]
        prompt_attn_mask[1: 1 + cls_name_length] = 1

        if self.method == 'train':
            if random.random() > 0.5:
                out_img = img
                wd, ht = img.size
                den_map = np.load(den_path)
                img_attn_map = np.ones((ht, wd))

            else:
                if random.random() > 0.5:
                    rand_img = random.sample(self.im_list, 1)[0]
                    rand_cls = self.cls_dict[os.path.basename(rand_img)]

                    out_img = img
                    if rand_cls != cls_name:
                        wd, ht = img.size
                        den_map = np.zeros((ht, wd))
                        prompt = rand_cls
                        img_attn_map = np.zeros((ht, wd))
                        prompt_attn_mask = torch.zeros(77)
                        cls_name_tokens = self.tokenizer(prompt, add_special_tokens=False, return_tensors='pt')
                        cls_name_length = cls_name_tokens['input_ids'].shape[1]
                        prompt_attn_mask[1: 1 + cls_name_length] = 1
                    else:
                        wd, ht = img.size
                        den_map = np.load(den_path)
                        img_attn_map = np.ones((ht, wd))
                else:
                    rand_imgs = random.sample(self.im_list, 3)
                    imgs_info = []
                    wd, ht = img.size
                    i, j, h, w = random_crop(ht, wd, self.concat_size, self.concat_size)
                    img = F.crop(img, i, j, h, w)
                    den_map = np.load(den_path)[i: (i + h), j: (j + w)]
                    img_attn_map = np.ones((self.concat_size, self.concat_size))
                    imgs_info.append({'img': img, 'den_map': den_map, 'img_attention_map': img_attn_map})
                    for rand_img in rand_imgs:
                        extra_img = Image.open(rand_img).convert('RGB')
                        wd, ht = extra_img.size
                        i, j, h, w = random_crop(ht, wd, self.concat_size, self.concat_size)
                        extra_img = F.crop(extra_img, i, j, h, w)
                        if self.cls_dict[os.path.basename(rand_img)] == cls_name:
                            extra_den_map = np.load(rand_img.replace('images_384_VarV2', 'gt_density_map_adaptive_384_VarV2').replace('jpg', 'npy'))[i: (i + h), j: (j + w)]
                            extra_img_attention = np.ones((self.concat_size, self.concat_size))
                        else:
                            extra_den_map = np.zeros((self.concat_size, self.concat_size))
                            extra_img_attention = np.zeros((self.concat_size, self.concat_size))
                        imgs_info.append({'img': extra_img, 'den_map': extra_den_map, 'img_attention_map': extra_img_attention})

                    random.shuffle(imgs_info)
                    out_img = Image.new('RGB', (self.concat_size * 2, self.concat_size * 2))
                    out_img.paste(imgs_info[0]['img'], (0, 0))
                    out_img.paste(imgs_info[1]['img'], (self.concat_size, 0))
                    out_img.paste(imgs_info[2]['img'], (0, self.concat_size))
                    out_img.paste(imgs_info[3]['img'], (self.concat_size, self.concat_size))

                    den_map = np.zeros((self.concat_size * 2, self.concat_size * 2))
                    den_map[0:self.concat_size, 0:self.concat_size] = imgs_info[0]['den_map']
                    den_map[0:self.concat_size, self.concat_size:self.concat_size * 2] = imgs_info[1]['den_map']
                    den_map[self.concat_size:self.concat_size * 2, 0:self.concat_size] = imgs_info[2]['den_map']
                    den_map[self.concat_size:self.concat_size * 2, self.concat_size:self.concat_size * 2] = imgs_info[3][
                        'den_map']

                    img_attn_map = np.zeros((self.concat_size * 2, self.concat_size * 2))
                    img_attn_map[0:self.concat_size, 0:self.concat_size] = imgs_info[0]['img_attention_map']
                    img_attn_map[0:self.concat_size, self.concat_size:self.concat_size * 2] = imgs_info[1][
                        'img_attention_map']
                    img_attn_map[self.concat_size:self.concat_size * 2, 0:self.concat_size] = imgs_info[2][
                        'img_attention_map']
                    img_attn_map[self.concat_size:self.concat_size * 2, self.concat_size:self.concat_size * 2] = \
                        imgs_info[3]['img_attention_map']

            img, den_map, img_attn_map = self.train_transform_density(out_img, den_map, img_attn_map)
            img_src, img_gd = load_image(im_path) 
            # img_gd = TF.resize(img_gd, (384, 384))        # resize Tensor
            # img_gd = img_gd.float()

            
            return img, den_map, prompt, prompt_attn_mask, img_attn_map, img_gd, img_src
        else:
            img = img.resize((384, 384), Image.Resampling.BICUBIC)
            img_src, img_gd = load_image(im_path)          # img_gd là Tensor (3,H,W)

            return self.transform(img), len(pts), prompt, prompt_attn_mask, os.path.basename(im_path).split('.')[0], img_gd, img_src
            # sample['image'].float(), sample['gt_map'], sample['boxes'], sample['pos'], text

    def train_transform_density(self, img, den_map, img_attention_map):
        wd, ht = img.size
        if random.random() >= 0.5:
            re_size = random.random() * 1 + 1
            wd = int(wd * re_size)
            ht = int(ht * re_size)
            img = img.resize((wd, ht), Image.Resampling.BICUBIC)
            den_map = cv2.resize(den_map, (wd, ht), interpolation=cv2.INTER_CUBIC) / (re_size ** 2)
            img_attention_map = cv2.resize(img_attention_map, (wd, ht), interpolation=cv2.INTER_NEAREST)

        i, j, h, w = random_crop(ht, wd, self.crop_size, self.crop_size)
        img = F.crop(img, i, j, h, w)
        den_map = den_map[i: (i + h), j: (j + w)]
        # den_map = den_map.reshape([h // self.down_ratio, self.down_ratio, w // self.down_ratio, self.down_ratio]).sum(
        #     axis=(1, 3)) # avoiding errors

        img_attention_map = img_attention_map[i: (i + h), j: (j + w)]
        img_attention_map = cv2.resize(img_attention_map, (int(w / 8), int(h / 8)), interpolation=cv2.INTER_NEAREST)

        if random.random() > 0.5:
            img = F.hflip(img)
            den_map = np.fliplr(den_map)
            img_attention_map = np.fliplr(img_attention_map)

        return self.transform(img), torch.from_numpy(den_map.copy()).float().unsqueeze(0), torch.from_numpy(img_attention_map.copy()).float().unsqueeze(0)


def collate_fn_train_object_count(batch):
    img, den_map, prompt, prompt_attn_mask, img_attn_map, img_gd, img_src = zip(*batch)

    return {
        'image': torch.stack(img, 0),
        'density': torch.stack(den_map, 0),
        'prompt_attn_mask': torch.stack(prompt_attn_mask, 0),
        'img_gd': img_gd,
        'img_src': img_src,
        'img_attn_map': torch.stack(img_attn_map, 0),
        'text': prompt
    }

def collate_fn_test_object_count(batch):
    img, batch_cnt, prompt, prompt_attn_mask, img_name, img_gd, img_src = zip(*batch)

   # batch_crops = get_exampler.get_highest_score_crop(img, prompt, box_threshold=BOX_THRESHOLD, keep_area=KEEP_AREA, device='cuda')

    batch_cnt = torch.tensor(batch_cnt)
    return {
        'image': torch.stack(img, 0),
        'batch_cnt': batch_cnt,
        'prompt_attn_mask': torch.stack(prompt_attn_mask, 0),
        'img_gd': img_gd,
        'img_src': img_src,
        'img_name': img_name,
        'text': prompt
    }

        # 
class ResizePreTrainImage(object):
    """
    Resize the image so that:
        1. Image is equal to 384 * 384
        2. The new height and new width are divisible by 16
        3. The aspect ratio is preserved
    Density and boxes correctness not preserved(crop and horizontal flip)
    """

    def __init__(self, MAX_HW=384):
        self.max_hw = MAX_HW

    def __call__(self, sample):
        image,lines_boxes,density = sample['image'], sample['lines_boxes'],sample['gt_density']
        
        W, H = image.size

        new_H = 16*int(H/16)
        new_W = 16*int(W/16)
        '''scale_factor = float(256)/ H
        new_H = 16*int(H*scale_factor/16)
        new_W = 16*int(W*scale_factor/16)'''
        resized_image = transforms.Resize((new_H, new_W))(image)
        resized_density = cv2.resize(density, (new_W, new_H))
        orig_count = np.sum(density)
        new_count = np.sum(resized_density)

        if new_count > 0: resized_density = resized_density * (orig_count / new_count)
            
        boxes = list()
        for box in lines_boxes:
            box2 = [int(k) for k in box]
            y1, x1, y2, x2 = box2[0], box2[1], box2[2], box2[3]
            boxes.append([0, y1,x1,y2,x2])

        boxes = torch.Tensor(boxes).unsqueeze(0)
        resized_image = PreTrainNormalize(resized_image)
        resized_density = torch.from_numpy(resized_density).unsqueeze(0).unsqueeze(0)
        sample = {'image':resized_image,'boxes':boxes,'gt_density':resized_density}
        return sample

class ResizeTrainImage(object):
    """
    Resize the image so that:
        1. Image is equal to 384 * 384
        2. The new height and new width are divisible by 16
        3. The aspect ratio is possibly preserved
    Density map is cropped to have the same size(and position) with the cropped image
    Exemplar boxes may be outside the cropped area.
    Augmentation including Gaussian noise, Color jitter, Gaussian blur, Random affine, Random horizontal flip and Mosaic (or Random Crop if no Mosaic) is used.
    """

    def __init__(self, MAX_HW=384, dataset:FSC147=None, aug = True):
        self.max_hw = MAX_HW
        self.dataset = dataset
        self.use_out_mosaic = False
        self.use_augmentation = aug
    def __call__(self, sample):
        image, lines_boxes, density, dots, im_id, m_flag = sample['image'], sample['lines_boxes'],sample['gt_density'], sample['dots'], sample['id'], sample['m_flag']
        
        W, H = image.size

        new_H = 16*int(H/16)
        new_W = 16*int(W/16)
        scale_factor = float(new_W)/ W
        resized_image = transforms.Resize((new_H, new_W))(image)
        resized_density = cv2.resize(density, (new_W, new_H))   
        
        # Augmentation probability
        aug_p = random.random()
        aug_flag = 0
        mosaic_flag = 0
        if self.use_augmentation and aug_p < 0.5: # 0.4
            aug_flag = 1
            if aug_p < 0.3: # 0.25
                aug_flag = 0
                mosaic_flag = 1

        # Gaussian noise
        resized_image = TTensor(resized_image)
        if aug_flag == 1:
            noise = np.random.normal(0, 0.1, resized_image.size())
            noise = torch.from_numpy(noise)
            re_image = resized_image + noise
            re_image = torch.clamp(re_image, 0, 1)

        # Color jitter and Gaussian blur
        if aug_flag == 1:
            re_image = Augmentation(re_image)

        # Random affine
        if aug_flag == 1:
            re1_image = re_image.transpose(0,1).transpose(1,2).numpy()
            keypoints = []
            for i in range(dots.shape[0]):
                keypoints.append(Keypoint(x=min(new_W-1,int(dots[i][0]*scale_factor)), y=min(new_H-1,int(dots[i][1]))))
            kps = KeypointsOnImage(keypoints, re1_image.shape)

            seq = iaa.Sequential([
                iaa.Affine(
                    rotate=(-15,15),
                    scale=(0.8, 1.2),
                    shear=(-10,10),
                    translate_percent={"x": (-0.2,0.2), "y": (-0.2,0.2)},
                    mode=ia.ALL, 
                )
            ])
            re1_image, kps_aug = seq(image=re1_image, keypoints=kps)

            # Produce dot annotation map
            resized_density = np.zeros((resized_density.shape[0], resized_density.shape[1]),dtype='float32')
            for i in range(len(kps.keypoints)):
                if(int(kps_aug.keypoints[i].y)<= new_H-1 and int(kps_aug.keypoints[i].x)<=new_W-1) and not kps_aug.keypoints[i].is_out_of_image(re1_image):
                    resized_density[int(kps_aug.keypoints[i].y)][int(kps_aug.keypoints[i].x)]=1
            resized_density = torch.from_numpy(resized_density)

            re_image = TTensor(re1_image)

        # Random horizontal flip
        if aug_flag == 1:
            flip_p = random.random()
            if flip_p > 0.5:
                re_image = TF.hflip(re_image)
                resized_density = TF.hflip(resized_density)
        
        # Random vertical flip
        if aug_flag == 1:
            flip_p = random.random()
            if flip_p > 0.5:
                re_image = TF.vflip(re_image)
                resized_density = TF.vflip(resized_density)

        
        # Random 384*384 crop in a new_W*384 image and 384*new_W density map
        
        if mosaic_flag == 0:
            if aug_flag == 0:
                re_image = resized_image
                resized_density = np.zeros((resized_density.shape[0], resized_density.shape[1]),dtype='float32')
                for i in range(dots.shape[0]):
                    resized_density[min(new_H-1,int(dots[i][1]))][min(new_W-1,int(dots[i][0]*scale_factor))]=1
                resized_density = torch.from_numpy(resized_density)

            start = random.randint(0, new_W-1-383)
            reresized_image = TF.crop(re_image, 0, start, 384, 384)
            reresized_density = resized_density[:, start:start+384]
        # Random self mosaic
        else:
            image_array = []
            map_array = []
            blending_l = random.randint(10, 20)
            resize_l = 192 + 2 * blending_l
            if dots.shape[0] >= 70 or not self.use_out_mosaic: #! Dec 29: ??
                for i in range(4):
                    length =  random.randint(150, 384)
                    start_W = random.randint(0, new_W-length)
                    start_H = random.randint(0, new_H-length)
                    reresized_image1 = TF.crop(resized_image, start_H, start_W, length, length)
                    reresized_image1 = transforms.Resize((resize_l, resize_l))(reresized_image1)
                    reresized_image = Augmentation(reresized_image1)
                    reresized_density1 = np.zeros((resize_l,resize_l),dtype='float32')
                    for i in range(dots.shape[0]):
                        if min(new_H-1,int(dots[i][1])) >= start_H and min(new_H-1,int(dots[i][1])) < start_H + length and min(new_W-1,int(dots[i][0]*scale_factor)) >= start_W and min(new_W-1,int(dots[i][0]*scale_factor)) < start_W + length:
                            reresized_density1[min(resize_l-1,int((min(new_H-1,int(dots[i][1]))-start_H)*resize_l/length))][min(resize_l-1,int((min(new_W-1,int(dots[i][0]*scale_factor))-start_W)*resize_l/length))]=1
                    reresized_density1 = torch.from_numpy(reresized_density1)
                    image_array.append(reresized_image1)
                    map_array.append(reresized_density1)
            elif self.use_out_mosaic: #! Dec 29: else mosaic with other classes?
                m_flag = 1
                prob = random.random()
                if prob > 0.25:
                    gt_pos = random.randint(0,3)
                else:
                    gt_pos = random.randint(0,4) # 5% 0 objects
                for i in range(4):
                    if i == gt_pos:
                        Tim_id = im_id
                        r_image = resized_image
                        Tdots = dots
                        new_TH = new_H
                        new_TW = new_W
                        Tscale_factor = scale_factor
                    else:
                        Tim_id = self.dataset.idx_running_set[random.randint(0, len(self.dataset.idx_running_set)-1)]
                        Tdots = np.array(self.dataset.annotations[Tim_id]['points'])
                        '''while(abs(Tdots.shape[0]-dots.shape[0]<=10)):
                            Tim_id = train_set[random.randint(0, len(train_set)-1)]
                            Tdots = np.array(annotations[Tim_id]['points'])'''
                        Timage = Image.open('{}/{}'.format(self.dataset.im_dir, Tim_id))
                        Timage.load()
                        new_TH = 16*int(Timage.size[1]/16)
                        new_TW = 16*int(Timage.size[0]/16)
                        Tscale_factor = float(new_TW)/ Timage.size[0]
                        r_image = TTensor(transforms.Resize((new_TH, new_TW))(Timage))

                    length =  random.randint(250, 384)
                    start_W = random.randint(0, new_TW-length)
                    start_H = random.randint(0, new_TH-length)
                    r_image1 = TF.crop(r_image, start_H, start_W, length, length)
                    r_image1 = transforms.Resize((resize_l, resize_l))(r_image1)
                    r_density1 = np.zeros((resize_l,resize_l),dtype='float32')
                    if self.dataset.class_dict[im_id] == self.dataset.class_dict[Tim_id]:
                        for i in range(Tdots.shape[0]):
                            if min(new_TH-1,int(Tdots[i][1])) >= start_H and min(new_TH-1,int(Tdots[i][1])) < start_H + length and min(new_TW-1,int(Tdots[i][0]*Tscale_factor)) >= start_W and min(new_TW-1,int(Tdots[i][0]*Tscale_factor)) < start_W + length:
                                r_density1[min(resize_l-1,int((min(new_TH-1,int(Tdots[i][1]))-start_H)*resize_l/length))][min(resize_l-1,int((min(new_TW-1,int(Tdots[i][0]*Tscale_factor))-start_W)*resize_l/length))]=1
                    r_density1 = torch.from_numpy(r_density1)
                    image_array.append(r_image1)
                    map_array.append(r_density1)


            reresized_image5 = torch.cat((image_array[0][:,blending_l:resize_l-blending_l],image_array[1][:,blending_l:resize_l-blending_l]),1)
            reresized_density5 = torch.cat((map_array[0][blending_l:resize_l-blending_l],map_array[1][blending_l:resize_l-blending_l]),0)
            for i in range(blending_l):
                    reresized_image5[:,192+i] = image_array[0][:,resize_l-1-blending_l+i] * (blending_l-i)/(2*blending_l) + reresized_image5[:,192+i] * (i+blending_l)/(2*blending_l)
                    reresized_image5[:,191-i] = image_array[1][:,blending_l-i] * (blending_l-i)/(2*blending_l) + reresized_image5[:,191-i] * (i+blending_l)/(2*blending_l)
            reresized_image5 = torch.clamp(reresized_image5, 0, 1)

            reresized_image6 = torch.cat((image_array[2][:,blending_l:resize_l-blending_l],image_array[3][:,blending_l:resize_l-blending_l]),1)
            reresized_density6 = torch.cat((map_array[2][blending_l:resize_l-blending_l],map_array[3][blending_l:resize_l-blending_l]),0)
            for i in range(blending_l):
                    reresized_image6[:,192+i] = image_array[2][:,resize_l-1-blending_l+i] * (blending_l-i)/(2*blending_l) + reresized_image6[:,192+i] * (i+blending_l)/(2*blending_l)
                    reresized_image6[:,191-i] = image_array[3][:,blending_l-i] * (blending_l-i)/(2*blending_l) + reresized_image6[:,191-i] * (i+blending_l)/(2*blending_l)
            reresized_image6 = torch.clamp(reresized_image6, 0, 1)

            reresized_image = torch.cat((reresized_image5[:,:,blending_l:resize_l-blending_l],reresized_image6[:,:,blending_l:resize_l-blending_l]),2)
            reresized_density = torch.cat((reresized_density5[:,blending_l:resize_l-blending_l],reresized_density6[:,blending_l:resize_l-blending_l]),1)
            for i in range(blending_l):
                    reresized_image[:,:,192+i] = reresized_image5[:,:,resize_l-1-blending_l+i] * (blending_l-i)/(2*blending_l) + reresized_image[:,:,192+i] * (i+blending_l)/(2*blending_l)
                    reresized_image[:,:,191-i] = reresized_image6[:,:,blending_l-i] * (blending_l-i)/(2*blending_l) + reresized_image[:,:,191-i] * (i+blending_l)/(2*blending_l)
            reresized_image = torch.clamp(reresized_image, 0, 1)
        
        # Gaussian distribution density map
        reresized_density = ndimage.gaussian_filter(reresized_density.numpy(), sigma=(1, 1), order=0)

        # Density map scale up
        reresized_density = reresized_density * 60
        reresized_density = torch.from_numpy(reresized_density)
            
        # Crop bboxes and resize as 64x64
        boxes = list()
        cnt = 0
        for box in lines_boxes:
            cnt+=1
            if cnt>3:
                break
            box2 = [int(k) for k in box]
            y1, x1, y2, x2 = box2[0], int(box2[1]*scale_factor), box2[2], int(box2[3]*scale_factor)
            bbox = resized_image[:,y1:y2+1,x1:x2+1]
            bbox = transforms.Resize((64, 64))(bbox)
            boxes.append(bbox.numpy())
        boxes = np.array(boxes)
        boxes = torch.Tensor(boxes)

        
        # boxes shape [3,3,64,64], image shape [3,384,384], density shape[384,384]       
        sample = {'image':reresized_image,'boxes':boxes,'gt_density':reresized_density, 'm_flag': m_flag}

        return sample

PreTrainNormalize = transforms.Compose([   
        transforms.RandomResizedCrop(384, scale=(0.2, 1.0), interpolation=3), 
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        #transforms.Normalize(mean=IM_NORM_MEAN, std=IM_NORM_STD)
        ])

TTensor = transforms.Compose([   
        transforms.ToTensor(),
        ])

Augmentation = transforms.Compose([   
        transforms.ColorJitter(brightness=0.3, contrast=0.15, saturation=0.2, hue=0.2),
        transforms.GaussianBlur(kernel_size=(7,9))
        ])






import logging
import os 

def logg(log_file):
    if not os.path.exists(os.path.dirname(log_file)):
        os.makedirs(os.path.dirname(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()  # vẫn in ra màn hình
        ]
    )



class FSC147Stage1Dataset(Dataset):
    """
    Dataset FSC-147 cho Stage 1:
    - Input: root folder chứa:
        root/
          images_384_VarV2/
          FSC_147/Train_Test_Val_FSC_147.json
          FSC_147/ImageClasses_FSC147.txt
    - Trả ra: (PIL.Image, pos_text, neg_text)
      pos_text  = class name của ảnh
      neg_text  = class name của ảnh khác (cố gắng khác class)
    """

    def __init__(self, root: str, split: str = "train", text_type: str = "class",):
        super().__init__()
        assert split in ["train", "val", "test"]
        assert text_type in ["class", "text"]
        self.root = root
        self.split = split
        self.text_type = text_type

        # Load split file
        split_path = os.path.join(root, "Train_Test_Val_FSC_147.json")
        with open(split_path, "r") as f:
            split_data = json.load(f)[split]
        self.im_list = [os.path.join(root, "images_384_VarV2", x) for x in split_data]

        # Load map: image_name -> class_name
        class_file = os.path.join(root, "ImageClasses_FSC147.txt")
        self.cls_dict = {}
        with open(class_file, "r", encoding="utf-8") as f:
            for line in f:
                name, cls = line.strip().split("\t")
                self.cls_dict[name] = cls

        # Load map: image_name -> describe_name
        describe_file = os.path.join(root, "DescribeImageClasses_FSC147_v2.txt")
        self.describe_dict = {}
        with open(describe_file, "r", encoding="utf-8") as f:
            for line in f:
                name, des = line.strip().split("\t")
                self.describe_dict[name] = des

        # Precompute class for each image & list of all (basename, cls)
        self.im_info = []
        for p in self.im_list:
            basename = os.path.basename(p)
            cls_name = self.cls_dict[basename]
            des_name = self.describe_dict[basename]
            self.im_info.append((p, basename, cls_name, des_name))

        print(f"[FSC147Stage1Dataset] Loaded {len(self.im_info)} images for split={split}")

    def __len__(self):
        return len(self.im_info)

    def __getitem__(self, idx: int):
        img_path, basename, cls_name, des_name = self.im_info[idx]

        # Load image
        img = Image.open(img_path).convert("RGB")

        if self.text_type == "class":
            pos_text = cls_name
        else:
            pos_text = des_name

        # Negative text = class của ảnh khác (cố gắng khác class)
        neg_text = pos_text
        trial = 0
        while neg_text == pos_text and trial < 10:
            j = random.randint(0, len(self.im_info) - 1)
            _, _, cls_j, des_j = self.im_info[j]
            neg_text = cls_j if self.text_type == "class" else des_j
            trial += 1
        # nếu sau 10 lần vẫn trùng, đành chấp nhận (hiếm)

        return img, pos_text, neg_text


def collate_stage1(batch) -> Tuple[List[Image.Image], List[str], List[str]]:
    """
    batch: list of (img, pos_text, neg_text)
    """
    images = [b[0] for b in batch]
    pos_texts = [b[1] for b in batch]
    neg_texts = [b[2] for b in batch]
    return images, pos_texts, neg_texts