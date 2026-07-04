"""CT volume loading and preprocessing for Swin3D feature extraction.
Adapted from the CLIP pretraining pipeline.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import h5py
import SimpleITK as sitk
import tqdm
import os
import matplotlib.pyplot as plt
import sys
# sys.path.append("...")  # user config
from utils.ops import crop_rand_one_img,crop_rand_two_img,crop_rand_two_img_ZEROIMG,crop_img_2_patch
import random
import torch
import torch.nn.functional as F
import math
import random
import pickle
import pandas as pd
# sys.path.append("...")  # user config
from cv2resize3d import cv2resize3d
import cv2
import monai.transforms as mtf
from pathlib import Path

def stardardize(x,mean=None,std=None,args=None):
    # x_std = (x - mean) / (std + 1e-6)
    # try:
    #     x = np.clip(x, -1500, 500) + 1500
    #     x = x / x.max()
    # except:
    if args.standard_func == 'zscore':
        x_std = (x - x.mean()) / (x.std() + 1e-6)
        return x_std
    elif args.standard_func == 'max':
        x = torch.clamp(x, -1500, 500) + 1500
        x = x / x.max()
        return x
    elif args.standard_func == 'minmax':
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        return x



def _over_sample(path_label_dict):
    ## oversample
    ## dictdf
    if isinstance(path_label_dict,dict):
        path_label_dict = pd.DataFrame(path_label_dict.items(),columns=['path','label'])
    else:
        pass
    ## label
    label_0 = path_label_dict[path_label_dict['label']==0]
    label_1 = path_label_dict[path_label_dict['label']==1]
    len_0 = len(label_0)
    len_1 = len(label_1)
    if len_0 > len_1:
        label_1 = label_1.sample(len_0,replace=True)
    else:
        label_0 = label_0.sample(len_1,replace=True)
    path_label_dict = pd.concat([label_0,label_1])
    return path_label_dict


def random_rotate(tensor, angle_range=(-10, 10)):
    angle = random.uniform(*angle_range)
    theta = torch.tensor([
        [math.cos(math.radians(angle)), -math.sin(math.radians(angle)), 0],
        [math.sin(math.radians(angle)),  math.cos(math.radians(angle)), 0]
    ], dtype=tensor.dtype, device=tensor.device)
    theta = theta.unsqueeze(0)  # Add batch dimension
    grid = F.affine_grid(theta, tensor.unsqueeze(0).size(), align_corners=False)
    rotated_tensor = F.grid_sample(tensor.unsqueeze(0), grid, align_corners=False)
    return rotated_tensor.squeeze()

def random_translate(tensor, max_translate=0.1):
    translate_x = random.uniform(-max_translate, max_translate) * tensor.size(-1)
    translate_y = random.uniform(-max_translate, max_translate) * tensor.size(-2)
    theta = torch.tensor([
        [1, 0, translate_x / tensor.size(-1)],
        [0, 1, translate_y / tensor.size(-2)]
    ], dtype=tensor.dtype, device=tensor.device)
    theta = theta.unsqueeze(0)  # Add batch dimension
    grid = F.affine_grid(theta, tensor.unsqueeze(0).size(), align_corners=False)
    translated_tensor = F.grid_sample(tensor.unsqueeze(0), grid, align_corners=False)
    return translated_tensor.squeeze()

def random_scale(tensor, scale_range=(0.9, 1.1)):
    scale = random.uniform(*scale_range)
    theta = torch.tensor([
        [scale, 0, 0],
        [0, scale, 0]
    ], dtype=tensor.dtype, device=tensor.device)
    theta = theta.unsqueeze(0)  # Add batch dimension
    grid = F.affine_grid(theta, tensor.unsqueeze(0).size(), align_corners=False)
    scaled_tensor = F.grid_sample(tensor.unsqueeze(0), grid, align_corners=False)
    return scaled_tensor.squeeze()

def random_augment(tensor):
    # if tensor.isinstance(torch.Tensor):
    tensor = torch.tensor(tensor)

    tensor = tensor.float()
    if len(tensor.shape) == 3:
        pass
    elif len(tensor.shape) == 4:
        tensor = tensor[0]
    tensor = random_rotate(tensor)
    tensor = random_translate(tensor)
    tensor = random_scale(tensor)
    if len(tensor.shape) == 3:
        pass
    elif len(tensor.shape) == 4:
        tensor = tensor.squeeze(0)
    tensor = tensor.numpy()

    return tensor

class M3d_transform():
    def __init__(self,mode='train'):
        self.mode = mode
        train_transform = mtf.Compose(
                [
                    mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),
                    mtf.RandFlip(prob=0.10, spatial_axis=0),
                    mtf.RandFlip(prob=0.10, spatial_axis=1),
                    mtf.RandFlip(prob=0.10, spatial_axis=2),
                    mtf.RandScaleIntensity(factors=0.1, prob=0.5),
                    mtf.RandShiftIntensity(offsets=0.1, prob=0.5),
                    ## 
                    # mtf.RandGaussianNoise(prob=0.5, mean=0, std=0.1),
                    # mtf.RandGaussianBlur(prob=0.5, sigma=(0.5, 1.5)),
                    mtf.ToTensor(dtype=torch.float),
                ]
            )

        val_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )

        self.standard_transform = mtf.Compose(
                [
                    mtf.ToTensor(dtype=torch.float),
                ]
            )

        if mode == 'train':
            self.transform = train_transform
            print('train transform\n\
                   mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),\n\
                    mtf.RandFlip(prob=0.10, spatial_axis=0),\n\
                    mtf.RandFlip(prob=0.10, spatial_axis=1),\n\
                    mtf.RandFlip(prob=0.10, spatial_axis=2),\n\
                    mtf.RandScaleIntensity(factors=0.1, prob=0.5),\n\
                    mtf.RandShiftIntensity(offsets=0.1, prob=0.5),\n\
                    mtf.ToTensor(dtype=torch.float),')
        else:
            self.transform = val_transform
            print('val transform')

    def __call__(self,img):
        # if self.mode == 'train':
        #     if random.random() > 0.5:
        #         img = self.transform(img)
        #     else:
        img = self.standard_transform(img)
        return img


class M3d_transform_seg():
    def __init__(self, mode='train'):
        self.train_transform = mtf.Compose(
            [
                CustomRandRotate90(prob=0.5, spatial_axes=(1, 2)),
                CustomRandFlip(prob=0.10, spatial_axis=0),
                CustomRandFlip(prob=0.10, spatial_axis=1),
                CustomRandFlip(prob=0.10, spatial_axis=2),
                CustomRandScaleIntensity(factors=0.1, prob=0.5),
                CustomRandShiftIntensity(offsets=0.1, prob=0.5),
                mtf.ToTensor(dtype=torch.float),
            ]
        )

        self.val_transform = mtf.Compose(
            [
                mtf.ToTensor(dtype=torch.float),
            ]
        )

        if mode == 'train':
            self.transform = self.train_transform
            print('train transform\n\
                   mtf.RandRotate90(prob=0.5, spatial_axes=(1, 2)),\n\
                    mtf.RandFlip(prob=0.10, spatial_axis=0),\n\
                    mtf.RandFlip(prob=0.10, spatial_axis=1),\n\
                    mtf.RandFlip(prob=0.10, spatial_axis=2),\n\
                    mtf.RandScaleIntensity(factors=0.1, prob=0.5),\n\
                    mtf.RandShiftIntensity(offsets=0.1, prob=0.5),\n\
                    mtf.ToTensor(dtype=torch.float),')
        else:
            self.transform = self.val_transform
            print('val transform')

    def __call__(self, img, label=None):
        if label is None:
            return self.transform(img)
        else:
            img = self.transform(img)
            label = self.transform.apply_transform_to_label(label)
            return img, label


class CustomRandRotate90(mtf.RandRotate90):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_rotated = False
        self.last_k = None
        self.last_spatial_axes = None

    def __call__(self, img):
        self.last_rotated = False
        if self.R.random() < self.prob:
            self.last_rotated = True
            self.last_k = self.R.randint(1, 4)
            self.last_spatial_axes = self.spatial_axes
            return super().__call__(img)
        return img

    def apply_transform_to_label(self, label):
        if self.last_rotated:
            return mtf.Rotate90(k=self.last_k, spatial_axes=self.last_spatial_axes)(label)
        return label


class CustomRandFlip(mtf.RandFlip):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_flipped = False
        self.last_spatial_axis = None

    def __call__(self, img):
        self.last_flipped = False
        if self.R.random() < self.prob:
            self.last_flipped = True
            self.last_spatial_axis = self.spatial_axis
            return super().__call__(img)
        return img

    def apply_transform_to_label(self, label):
        if self.last_flipped:
            return mtf.Flip(spatial_axis=self.last_spatial_axis)(label)
        return label


class CustomRandScaleIntensity(mtf.RandScaleIntensity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_scaled = False
        self.last_factor = None

    def __call__(self, img):
        self.last_scaled = False
        if self.R.random() < self.prob:
            self.last_scaled = True
            self.last_factor = self.R.uniform(-self.factors, self.factors)
            return super().__call__(img)
        return img

    def apply_transform_to_label(self, label):
        # ：，
        return label


class CustomRandShiftIntensity(mtf.RandShiftIntensity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_shifted = False
        self.last_offset = None

    def __call__(self, img):
        self.last_shifted = False
        if self.R.random() < self.prob:
            self.last_shifted = True
            self.last_offset = self.R.uniform(-self.offsets, self.offsets)
            return super().__call__(img)
        return img

    def apply_transform_to_label(self, label):
        # ：，
        return label





class dataset_clip(nn.Module):
    def __init__(self, args=None, rootdir=None, csvpath=None, mode='train',debug=False):
        super(dataset_clip, self).__init__()
        if debug:
            self.imglist = [""]
            self.show_before_train(nums=1)
            exit()
        
        self.args = args
        
        self.imglist,self.label_list = self.read_csv(csvpath)
        if rootdir:
            self.imglist = [rootdir + img for img in self.imglist]
        try:
            self.label_list = [int(label) for label in self.label_list]
        except:
            self.label_list = None
        
        ###assert img exists
        self.not_exists = []
        for img in tqdm.tqdm(self.imglist):
            if not os.path.exists(img):
                self.not_exists.append(img)
        if len(self.not_exists) > 0:
            print('not exists: ', self.not_exists)
            print('total not exists: ', len(self.not_exists))
            if_con = input('img not exists, continue? y/n')
            if if_con == 'y':
                self.imglist = [img for img in self.imglist if img not in self.not_exists]
            else:
                raise ValueError(f'{str(self.not_exists)} not exists')
        else:
            print('all img exists')
        print('total img: ', len(self.imglist))
        
        
        self.train = False
        if mode == 'train':
            self.train = True
    
    def show_before_train(self, nums=10):
        plt.figure(figsize=(20,20))
        for i in range(nums):
            img,imgpath = self.__getitem__(i)[0],self.__getitem__(i)[-1]
            plt.subplot(1,nums,i+1)
            try:
                plt.imshow(img)
            except:
                pass
            try:
                plt.imshow(img[0])
            except:
                plt.imshow(img[0,0,...])
                plt.title(imgpath)
        try:
            plt.savefig(self.args.save_path+'/before_train.png')
        except:
            plt.savefig('./before_train.png')
        return


    def read_csv(self, csvpath):
        imglist = []
        label_list = []
        with open(csvpath, 'r') as f:
            for line in f.readlines():
                if ',' in line:
                    imgname, label = line.strip().split(',')
                else:
                    imgname= line.strip()
                    label = None
                imglist.append(imgname)
                label_list.append(label)
        return imglist,label_list
    
    
    def __len__(self):
        return len(self.imglist)
    
    
    def read_img(self, imgpath):
        if imgpath.endswith('.h5'):
            try:
                img = self._read_h5(imgpath, 'image')
            except:
                img = self._read_h5(imgpath, 'data')
        elif imgpath.endswith('.npy'):
            img = self._read_npy(imgpath)
        elif imgpath.endswith('.nii.gz'):
            img = self._read_nii(imgpath)
        else:
            raise ValueError('imgpath should be h5 or npy')
        return img
    
    def standard_img(self, img):
        img = img.astype(np.float32)
        img = (img - img.mean()) / (img.std() + 1e-6)
        img = np.expand_dims(img, axis=0)
        return img
    
    def random_transform_img(self, img):
        # print(img.shape)
        img,posi = crop_rand_one_img(a=32,b=32,c=32,samples=img)
        return img,posi
    
    def __getitem__(self, idx):
        
        imgpath = self.imglist[idx]
        try:
            img = self.read_img(imgpath)
        except:
            img = np.ones((96,96,96))
            
        img = self.standard_img(img)
        # if self.train:
        img_crop,posi = self.random_transform_img(img)
        img = torch.from_numpy(img)
        # print(img.shape,img_crop.shape)
        
        try:
            label = self.label_list[idx]
            label = torch.tensor(label)
        except:
            return self.__getitem__(idx+1)
        
        return [img, img_crop, posi,label,imgpath]

    def _read_h5(self, h5path, key):
        with h5py.File(h5path, 'r') as f:
            data = f[key][:]
        return data

    def _read_npy(self, npypath):
        data = np.load(npypath)
        return data

    def _read_nii(self,path):
        data = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(data)
        return data

class dataset_clip_twopatch(nn.Module):
    def __init__(self, args=None, rootdir=None, csvpath=None, mode='train',debug=False):
        super(dataset_clip_twopatch, self).__init__()
        self.show = True

        if debug:
            self.imglist = [""]
            self.show_before_train(nums=1)
            exit()
        
        self.args = args
        
        self.imglist,self.label_list = self.read_csv(csvpath)
        if rootdir:
            self.imglist = [rootdir + img for img in self.imglist]
        try:
            self.label_list = [int(label) for label in self.label_list]
        except:
            self.label_list = None
        
        ###assert img exists
        self.not_exists = []
        for img in tqdm.tqdm(self.imglist):
            if not os.path.exists(img):
                self.not_exists.append(img)
        if len(self.not_exists) > 0:
            print('not exists: ', self.not_exists)
            print('total not exists: ', len(self.not_exists))
            if_con = input('img not exists, continue? y/n')
            if if_con == 'y':
                self.imglist = [img for img in self.imglist if img not in self.not_exists]
            else:
                raise ValueError(f'{str(self.not_exists)} not exists')
        else:
            print('all img exists')
        print('total img: ', len(self.imglist))
        
        
        self.train = False
        if mode == 'train':
            self.train = True
    
    def img_aug(self,img):
        if random.random() < 0.5:
            img = random_augment(img)
        return img

    def show_before_train(self, nums=10):
        plt.figure(figsize=(20,20))
        for i in range(nums):
            img,imgpath = self.__getitem__(i)[0],self.__getitem__(i)[-1]
            plt.subplot(1,nums,i+1)
            try:
                plt.imshow(img)
            except:
                pass
            try:
                plt.imshow(img[0])
            except:
                plt.imshow(img[0,0,...])
                plt.title(imgpath)
        try:
            plt.savefig(self.args.save_path+'/before_train.png')
        except:
            plt.savefig('./before_train.png')
        return


    def read_csv(self, csvpath):
        imglist = []
        label_list = []
        with open(csvpath, 'r') as f:
            for line in f.readlines():
                if ',' in line:
                    imgname, label = line.strip().split(',')
                else:
                    imgname= line.strip()
                    label = None
                imglist.append(imgname)
                label_list.append(label)
        return imglist,label_list
    
    
    def __len__(self):
        return len(self.imglist)
    
    
    def read_img(self, imgpath):
        if imgpath.endswith('.h5'):
            try:
                img = self._read_h5(imgpath, 'image')
            except:
                img = self._read_h5(imgpath, 'data')
        elif imgpath.endswith('.npy'):
            img = self._read_npy(imgpath)
        elif imgpath.endswith('.nii.gz'):
            img = self._read_nii(imgpath)
        elif imgpath.endswith(".pt"):
            img = torch.load(imgpath)
        else:
            raise ValueError('imgpath should be h5 or npy')
        return img
    
    def standard_img(self, img):
        try:
            img = img.astype(np.float32)
            img = (img - img.mean()) / (img.std() + 1e-6)
            img = np.expand_dims(img, axis=0)
        except:
            img = img.float()
            img = (img - img.mean()) / (img.std() + 1e-6)
            img = img.unsqueeze(0)
        return img
    
    def random_transform_img(self, img):
        # print(img.shape)
        # img,posi = crop_rand_one_img(a=32,b=32,c=32,samples=img)
        img,posi,img1,posi1,label = crop_rand_two_img(a=32,b=32,c=32,samples=img)
        return img,posi,img1,posi1,label
    
    def __getitem__(self, idx):
        
        imgpath = self.imglist[idx]
        try:
            img = self.read_img(imgpath)
        except:
            print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
            # img = np.ones((96,96,96))
            img = torch.ones((96,96,96))
            
        img = self.standard_img(img)
        # if self.train:
        img_crop,posi,img_crop1,posi1,posilabel = self.random_transform_img(img)
        posilabel = torch.tensor(posilabel,dtype=torch.float32)
        # img = torch.from_numpy(img)
        # print(img.shape,img_crop.shape)
        
        try:
            # label = self.label_list[idx]
            # label = torch.tensor(label)
            label = torch.tensor(0) # 
        except:
            print("dataset errorrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
            return self.__getitem__(idx+1)
        
        if self.show:
            img = img.numpy()
            img2save = img[0,...]
            # os.makedirs("/ssd/lcc/code/ubuntu/EGFR/clip/egfr_img",exist_ok=True)
            sitk.WriteImage(sitk.GetImageFromArray(img2save),f"{self.args.save_path}/{idx}.nii")
            self.show = False

            img = self.img_aug(img)
            img_crop = self.img_aug(img_crop)
            img_crop1 = self.img_aug(img_crop1)

        return [img, img_crop, posi,img_crop1, posi1,posilabel,label,imgpath]

    def _read_h5(self, h5path, key):
        with h5py.File(h5path, 'r') as f:
            data = f[key][:]
        return data

    def _read_npy(self, npypath):
        data = np.load(npypath)
        return data

    def _read_nii(self,path):
        data = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(data)
        return data




class dataset_clip_twopatch_text(nn.Module):
    def __init__(self, args=None, rootdir=None, csvpath=None, mode='train',debug=False,cut=True,text=True,shuffle=True):
        super(dataset_clip_twopatch_text, self).__init__()
        self.shuffle = shuffle
        self.show = True
        self.cut = cut
        self.text = text

        if debug:
            self.imglist = [""]
            self.show_before_train(nums=1)
            exit()
        
        self.args = args
        if self.text:
            if 'clipv3' in self.args.model.lower():
                pass
            else:
                self.text_feature_pth = args.text_feature_file
                self.text_feature_dict = pickle.load(open(self.text_feature_pth, "rb"))
        
        
        
        self.imglist,self.label_list = self.read_csv(csvpath)
        if rootdir:
            self.imglist = [rootdir + img for img in self.imglist]
        # try:
        #     self.label_list = [int(label) for label in self.label_list]
        # except:
        #     self.label_list = None
        
        ###assert img exists
        # self.not_exists = []
        # for img in tqdm.tqdm(self.imglist):
        #     if not os.path.exists(img):
        #         self.not_exists.append(img)
        # if len(self.not_exists) > 0:
        #     print('not exists: ', self.not_exists)
        #     print('total not exists: ', len(self.not_exists))
        #     if_con = input('img not exists, continue? y/n')
        #     if if_con == 'y':
        #         self.imglist = [img for img in self.imglist if img not in self.not_exists]
        #     else:
        #         raise ValueError(f'{str(self.not_exists)} not exists')
        # else:
        #     print('all img exists')
        # print('total img: ', len(self.imglist))
        
        
        self.train = False
        if mode == 'train':
            self.train = True
    
    def load_text_feature(self,patid):
            if 'clipv3' in self.args.model.lower() or "clip_clam" in self.args.model.lower():
                if 'henan' in patid:
                    feadir = ""
                    patname = patid.split('/')[-1].split('.')[0]
                    patname = '_'.join(patname.split('_')[:2])
                elif 'shengjing' in patid:
                    feadir = ""
                    patname = patid.split('/')[-1].split('.')[0].split('_')[0]
                else:
                    feadir = None
                feapath = feadir + patname + '.pt'
                text_feature = torch.load(feapath).pooler_output.detach().cpu()
            else:
                patid = patid.split('/')[-2]
                text_feature = self.text_feature_dict[patid]
            return text_feature

    def img_aug(self,img):
        if random.random() < 0.5:
            img = random_augment(img)
        return img

    def show_before_train(self, nums=10):
        plt.figure(figsize=(20,20))
        for i in range(nums):
            img,imgpath = self.__getitem__(i)[0],self.__getitem__(i)[-1]
            plt.subplot(1,nums,i+1)
            try:
                plt.imshow(img)
            except:
                pass
            try:
                plt.imshow(img[0])
            except:
                plt.imshow(img[0,0,...])
                plt.title(imgpath)
        try:
            plt.savefig(self.args.save_path+'/before_train.png')
        except:
            plt.savefig('./before_train.png')
        return

    def read_csv(self, csvpath,ifcls=False):
        imglist = []
        label_list = []

        try:
            df = pd.read_csv(csvpath)
        except:
            try:
                df = pd.read_csv(csvpath,encoding='gbk')
            except:
                df = pd.read_excel(csvpath)

        
        imglist = df['path'].tolist()

        if 'label' in df.columns:
            label_list = df['label'].tolist()
        else:
            label_list = [0]*len(imglist)
        

        dellist = []
        for i in range(len(label_list)):
            patid = imglist[i].split('/')[-2]
            patpath = imglist[i]
            if self.text:
                if 'clipv3' in self.args.model.lower() or "clip_clam" in self.args.model.lower():
                    if 'henan' in patpath:
                        feadir = ""
                        patname = patid.split('/')[-1].split('.')[0]
                        patname = '_'.join(patname.split('_')[:2])
                    elif 'shengjing' in patpath:
                        feadir = ""
                        patname = patid.split('/')[-1].split('.')[0].split('_')[0]
                    
                    feapath = feadir + patname + '.pt'
                    if os.path.exists(feapath):
                        pass
                    else:
                        # print(f'{feapath} not exists')
                        dellist.append(i)
                        continue

                else:
                    if patid not in self.text_feature_dict.keys():
                        # print(f'{patid} not in text feature dict')
                        dellist.append(i)
                        continue

            if os.path.exists(imglist[i]):
                pass
            else:
                imglist[i] = imglist[i].replace("")
                if os.path.exists(imglist[i]):
                    pass
                else:
                    # print(f'{imglist[i]} not exists')
                    dellist.append(i)
        
        imglist = [imglist[i] for i in range(len(imglist)) if i not in dellist]
        label_list = [label_list[i] for i in range(len(label_list)) if i not in dellist]

        print(f'total img: {len(imglist)}')
        print(f'total label: {len(label_list)}')
        print(f'total not exists: {len(dellist)}')        
                
        return imglist,label_list
    

    # def read_csv(self, csvpath):
    #     imglist = []
    #     label_list = []
    #     with open(csvpath, 'r') as f:
    #         for line in f.readlines():
    #             if ',' in line:
    #                 imgname, label = line.strip().split(',')
    #             else:
    #                 imgname= line.strip()
    #                 label = None
    #             if 'shengjing' in imgname:
    #                 continue
    #             imgname = imgname#.replace("", '/a800/ssd/')
    #             pat_name = imgname.split('/')[-2]
    #             if pat_name not in self.text_feature_dict:
    #                 continue
    #             imglist.append(imgname)
    #             label_list.append(label)

    #     return imglist,label_list
    
    
    def __len__(self):
        return len(self.imglist)
    
    
    def read_img(self, imgpath):
        if imgpath.endswith('.h5'):
            try:
                img = self._read_h5(imgpath, 'image')
            except:
                img = self._read_h5(imgpath, 'data')
        elif imgpath.endswith('.npy'):
            img = self._read_npy(imgpath)
        elif imgpath.endswith('.nii.gz'):
            img = self._read_nii(imgpath)
        elif imgpath.endswith(".pt"):
            img = torch.load(imgpath)
        else:
            raise ValueError('imgpath should be h5 or npy')
        return img
    
    def standard_img(self, img, mean=None ,std=None):
        if mean and std:
            try:
                img = img.astype(np.float32)
                img = (img - mean) / (std + 1e-6)
                # img = np.expand_dims(img, axis=0)
            except:
                img = img.float()
                img = (img - mean) / (std + 1e-6)
                # img = img.unsqueeze(0)
        else:

            try:
                img = img.astype(np.float32)
                img = (img - img.mean()) / (img.std() + 1e-6)
                # img = np.expand_dims(img, axis=0)
            except:
                img = img.float()
                img = (img - img.mean()) / (img.std() + 1e-6)
                # img = img.unsqueeze(0)

        return img
    
    def random_transform_img(self, img):
        # print(img.shape)
        # img,posi = crop_rand_one_img(a=32,b=32,c=32,samples=img)
        # img,posi,img1,posi1,label = crop_rand_two_img(a=32,b=32,c=32,samples=img) # 96 96 96 patch
        img,posi,img1,posi1,label = crop_rand_two_img(a=16,b=64,c=64,samples=img) # 48 256 256 patch

        return img,posi,img1,posi1,label
    
    def __getitem__(self, idx):
        
        if self.train and self.shuffle:
            randint = random.randint(0,self.__len__()-1)
        else:
            randint = idx
        idx = randint
        # random.shuffle(self.imglist)
        imgpath = self.imglist[idx]#.replace("", '/a800/ssd/')
        pat_name = imgpath.split('/')[-2]

        try:
            if self.text:
                text_feature = self.load_text_feature(imgpath)
                # text_feature = (text_feature - text_feature.mean()) / (text_feature.std() + 1e-6)

            try:
                img = self.read_img(imgpath)
                if not imgpath.endswith(".pt"):
                    img = torch.from_numpy(img)
                img = img.unsqueeze(0)

            except:
                print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
                return self.__getitem__(idx+1)
            
            img = img.float()
            

            if self.cut:
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                img_crop,posi,img_crop1,posi1,posilabel = self.random_transform_img(img)
                posilabel = torch.tensor(posilabel,dtype=torch.float32)
            
            label = torch.tensor(0) # 
            
            if self.train:
                # print('img',img.shape)
                if self.cut:
                    img_crop = self.img_aug(img_crop)
                    img_crop1 = self.img_aug(img_crop1)
                    # img_crop1 = img_crop1.unsqueeze(0)
                    # img_crop = img_crop.unsqueeze(0)

                    # print('img',img.shape)
                # img = img.squeeze(0)
                img = self.img_aug(img)
                # img = img.unsqueeze(0)

            img = self.standard_img(img)
            

            # img_mean = img.mean()
            # img_std = img.std()
            if self.cut:
                img_crop = self.standard_img(img_crop)
                img_crop1 = self.standard_img(img_crop1)

            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            
            if self.cut:
                if len(img_crop.shape) == 3:
                    img_crop = img_crop.unsqueeze(0)
                if len(img_crop1.shape) == 3:
                    img_crop1 = img_crop1.unsqueeze(0)
            # if self.show:
            #     img = img.numpy()
            #     img2save = img[0,...]
            #     # os.makedirs("/ssd/lcc/code/ubuntu/EGFR/clip/egfr_img",exist_ok=True)
            #     sitk.WriteImage(sitk.GetImageFromArray(img2save),f"{self.args.save_path}/{idx}.nii")
            #     self.show = False
            #     exit()
            if not self.text:
                return [img, img_crop, posi,img_crop1, posi1,posilabel,label,imgpath]
            if not self.cut:
                return [img, text_feature, label, imgpath]
            else:
                return [img, img_crop, posi,img_crop1, posi1,posilabel,text_feature,label,imgpath]

        except Exception as e:
            print(e)
            print(f"error: {pat_name}")
            # exit()
            return self.__getitem__(idx+1)


    def _read_h5(self, h5path, key):
        with h5py.File(h5path, 'r') as f:
            data = f[key][:]
        return data

    def _read_npy(self, npypath):
        data = np.load(npypath)
        return data

    def _read_nii(self,path):
        data = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(data)
        return data




class dataset_clip_imgfea_text(nn.Module):
    def __init__(self, args=None, rootdir=None, csvpath=None, mode='train',debug=False,cut=True,text=True,shuffle=True):
        super(dataset_clip_imgfea_text, self).__init__()
        self.shuffle = shuffle
        self.show = True
        self.cut = cut
        self.text = text
        self.args = args

        self.read_csv(csvpath)

        if rootdir:
            self.imglist = [rootdir + img for img in self.imglist]

        
        self.train = False
        if mode == 'train':
            self.train = True
    
    def load_textfea(self,textpath):
        text_feature = torch.load(textpath,map_location="cpu")["pooler_output"].detach()
        # text_feature = torch.load(textpath)["pooler_output"].detach()
        return text_feature
    def load_imgfea(self,imgpath):
        img_feature = torch.load(imgpath,map_location="cpu").detach()
        # img_feature = torch.load(imgpath).detach()
        return img_feature


    def read_csv(self, csvpath,ifcls=False):
        imglist = []
        label_list = []

        try:
            df = pd.read_csv(csvpath)
        except:
            try:
                df = pd.read_csv(csvpath,encoding='gbk')
            except:
                df = pd.read_excel(csvpath)

        print(df.columns)

        dellist = [""]
        ## imgfeapathtextfeapath
        # for p in df['imgfeapath'].tolist():
        for p in dellist:
            print(f'{p} not exists')
            ## 
            df = df[df['imgfeapath'] != p]

            # if not os.path.exists(p):
            #     print(f'{p} not exists')
            #     ## 
            #     df = df[df['imgfeapath'] != p]




            # try:
            #     fea = torch.load(p,map_location="cpu")
            # except:
            #     print(f'{p} not exists')
            #     ## 
            #     df = df[df['imgfeapath'] != p]

        # for p in df['textfeapath'].tolist():
        for p in dellist:
            print(f'{p} not exists')
            ## 
            df = df[df['textfeapath'] != p]
        # if not os.path.exists(p):
        #     print(f'{p} not exists')
        #     ## 
        #     df = df[df['textfeapath'] != p]

            # try:
            #     fea = torch.load(p,map_location="cpu")
            # except:
            #     print(f'{p} not exists')
            #     ## 
            #     df = df[df['textfeapath'] != p]

        self.pathdict = df.to_dict(orient="index")
        print(f'total img: {len(self.pathdict)}')

        if self.train and self.shuffle:
            keys = random.sample(list(self.pathdict.keys()),len(self.pathdict))
            self.pathdict = {k:self.pathdict[k] for k in keys}
        else:
            # 1024
            self.pathdict = {k:self.pathdict[k] for k in list(self.pathdict.keys())[:1024]}
        # exit()
        return
    
    def __len__(self):
        return len(self.pathdict)

    def __getitem__(self, idx):
        try:
            pathdict = self.pathdict[idx]
            imgfeapath = pathdict['imgfeapath']
            textfeapath = pathdict['textfeapath']
            name = pathdict['name']

            imgfea = self.load_imgfea(imgfeapath)
            textfea = self.load_textfea(textfeapath)
            # print(imgfea.shape,textfea.shape)
            # exit()
            return [imgfea, textfea, name]

        except Exception as e:
            print(e)
            print(f"error")
            # exit()
            return self.__getitem__(idx+1)




    def _read_h5(self, h5path, key):
        with h5py.File(h5path, 'r') as f:
            data = f[key][:]
        return data

    def _read_npy(self, npypath):
        data = np.load(npypath)
        return data

    def _read_nii(self,path):
        data = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(data)
        return data





class dataset_clip_twopatch_textRecon(nn.Module):
    def __init__(self, args=None, rootdir=None, csvpath=None, mode='train',debug=False,cut=True,text=True):
        super(dataset_clip_twopatch_textRecon, self).__init__()
        self.show = True
        self.cut = cut
        self.text = text

        if debug:
            self.imglist = [""]
            self.show_before_train(nums=1)
            exit()
        
        self.args = args
        if self.text:
            self.text_feature_pth = args.text_feature_file
            self.text_feature_dict = pickle.load(open(self.text_feature_pth, "rb"))
        
        self.imglist,self.label_list = self.read_csv(csvpath)
        if rootdir:
            self.imglist = [rootdir + img for img in self.imglist]
        # try:
        #     self.label_list = [int(label) for label in self.label_list]
        # except:
        #     self.label_list = None
        
        ###assert img exists
        # self.not_exists = []
        # for img in tqdm.tqdm(self.imglist):
        #     if not os.path.exists(img):
        #         self.not_exists.append(img)
        # if len(self.not_exists) > 0:
        #     print('not exists: ', self.not_exists)
        #     print('total not exists: ', len(self.not_exists))
        #     if_con = input('img not exists, continue? y/n')
        #     if if_con == 'y':
        #         self.imglist = [img for img in self.imglist if img not in self.not_exists]
        #     else:
        #         raise ValueError(f'{str(self.not_exists)} not exists')
        # else:
        #     print('all img exists')
        # print('total img: ', len(self.imglist))
        
        
        self.train = False
        if mode == 'train':
            self.train = True
    
    def load_text_feature(self,patid):
            text_feature = self.text_feature_dict[patid]
            return text_feature

    def img_aug(self,img):
        if random.random() < 0.5:
            img = random_augment(img)
        return img

    def show_before_train(self, nums=10):
        plt.figure(figsize=(20,20))
        for i in range(nums):
            img,imgpath = self.__getitem__(i)[0],self.__getitem__(i)[-1]
            plt.subplot(1,nums,i+1)
            try:
                plt.imshow(img)
            except:
                pass
            try:
                plt.imshow(img[0])
            except:
                plt.imshow(img[0,0,...])
                plt.title(imgpath)
        try:
            plt.savefig(self.args.save_path+'/before_train.png')
        except:
            plt.savefig('./before_train.png')
        return

    def read_csv(self, csvpath,ifcls=False):
        imglist = []
        label_list = []

        try:
            df = pd.read_csv(csvpath)
        except:
            try:
                df = pd.read_csv(csvpath,encoding='gbk')
            except:
                df = pd.read_excel(csvpath)

        ## df
        if 'path' in df.columns:
            pass
        else:
            df.columns = ['path','label']
            # raise Warning('csv should have path column,assert')
            # print(df.head())
            # # y/n
            # if_con = input('csv should have path column,assert,continue? y/n')
            # if if_con == 'y':
            #     pass
            # else:
            #     exit()
        
        imglist = df['path'].tolist()

        if 'label' in df.columns:
            label_list = df['label'].tolist()
        else:
            # raise Warning('csv should have label column,assert')
            # print(df.head())
            # # y/n
            # if_con = input('csv should have label column,assert,continue? y/n')
            # if if_con == 'y':
            #     pass
            # else:
            #     exit()
            label_list = [0]*len(imglist)
        
        dellist = []
        for i in range(len(label_list)):
            patid = imglist[i].split('/')[-2]
            if self.text:
                if patid not in self.text_feature_dict.keys():
                    # print(f'{patid} not in text feature dict')
                    dellist.append(i)
                    continue

            if os.path.exists(imglist[i]):
                pass
            else:
                imglist[i] = imglist[i].replace("")
                if os.path.exists(imglist[i]):
                    pass
                else:
                    print(f'{imglist[i]} not exists')
                    dellist.append(i)
        
        imglist = [imglist[i] for i in range(len(imglist)) if i not in dellist]
        label_list = [label_list[i] for i in range(len(label_list)) if i not in dellist]

        print(f'total img: {len(imglist)}')
        print(f'total label: {len(label_list)}')
        print(f'total not exists: {len(dellist)}')        
                
        return imglist,label_list
    

    # def read_csv(self, csvpath):
    #     imglist = []
    #     label_list = []
    #     with open(csvpath, 'r') as f:
    #         for line in f.readlines():
    #             if ',' in line:
    #                 imgname, label = line.strip().split(',')
    #             else:
    #                 imgname= line.strip()
    #                 label = None
    #             if 'shengjing' in imgname:
    #                 continue
    #             imgname = imgname#.replace("", '/a800/ssd/')
    #             pat_name = imgname.split('/')[-2]
    #             if pat_name not in self.text_feature_dict:
    #                 continue
    #             imglist.append(imgname)
    #             label_list.append(label)

    #     return imglist,label_list
    
    
    def __len__(self):
        return len(self.imglist)
    
    
    def read_img(self, imgpath):
        if imgpath.endswith('.h5'):
            try:
                img = self._read_h5(imgpath, 'image')
            except:
                img = self._read_h5(imgpath, 'data')
        elif imgpath.endswith('.npy'):
            img = self._read_npy(imgpath)
        elif imgpath.endswith('.nii.gz'):
            img = self._read_nii(imgpath)
        elif imgpath.endswith(".pt"):
            img = torch.load(imgpath)
        else:
            raise ValueError('imgpath should be h5 or npy')
        return img
    
    def standard_img(self, img, mean=None ,std=None):
        if mean and std:
            try:
                img = img.astype(np.float32)
                img = (img - mean) / (std + 1e-6)
                # img = np.expand_dims(img, axis=0)
            except:
                img = img.float()
                img = (img - mean) / (std + 1e-6)
                # img = img.unsqueeze(0)
        else:

            try:
                img = img.astype(np.float32)
                img = (img - img.mean()) / (img.std() + 1e-6)
                # img = np.expand_dims(img, axis=0)
            except:
                img = img.float()
                img = (img - img.mean()) / (img.std() + 1e-6)
                # img = img.unsqueeze(0)

        return img

    
    def random_transform_img(self, img):
        # print(img.shape)
        # img,posi = crop_rand_one_img(a=32,b=32,c=32,samples=img)
        # img,posi,img1,posi1,label = crop_rand_two_img(a=32,b=32,c=32,samples=img) # 96 96 96 patch
        # img,posi,img1,posi1,label = crop_rand_two_img(a=16,b=64,c=64,samples=img) # 48 256 256 patch
        oriimg,img,posi,img1,posi1,label = crop_rand_two_img_ZEROIMG(a=16,b=64,c=64,samples=img) # 48 256 256 patch

        return oriimg,img,posi,img1,posi1,label
    
    def __getitem__(self, idx):
        
        randint = random.randint(0,self.__len__()-1)
        idx = randint
        random.shuffle(self.imglist)
        imgpath = self.imglist[idx]#.replace("", '/a800/ssd/')
        pat_name = imgpath.split('/')[-2]

        try:
            if self.text:
                text_feature = self.load_text_feature(pat_name)

            try:
                img = self.read_img(imgpath)
                if not imgpath.endswith(".pt"):
                    img = torch.from_numpy(img)
                img = img.unsqueeze(0)

            except:
                print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
                return self.__getitem__(idx+1)
            
            img = img.float()
            




            if self.cut:
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                    
                    if self.train:
                        img = self.img_aug(img)

                
                img,img_crop,posi,img_crop1,posi1,posilabel = self.random_transform_img(img)
                posilabel = torch.tensor(posilabel,dtype=torch.float32)

                if self.train:
                    img_crop = self.img_aug(img_crop)
                    img_crop1 = self.img_aug(img_crop1)

            label = torch.tensor(0) # 

            
            
            
            img = self.standard_img(img)
            

            img_crop = self.standard_img(img_crop)
            img_crop1 = self.standard_img(img_crop1)

            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            if len(img_crop.shape) == 3:
                img_crop = img_crop.unsqueeze(0)
            if len(img_crop1.shape) == 3:
                img_crop1 = img_crop1.unsqueeze(0)
            
            # if self.show:
            #     img = img.numpy()
            #     img2save = img[0,...]
            #     # os.makedirs("/ssd/lcc/code/ubuntu/EGFR/clip/egfr_img",exist_ok=True)
            #     sitk.WriteImage(sitk.GetImageFromArray(img2save),f"{self.args.save_path}/{idx}.nii")
            #     self.show = False
            #     exit()
            
            # img2save = img.squeeze(0).squeeze(0).numpy()
            # img2savepath = "./log/CLIP/downstream_task/img.nii.gz"
            # sitk.WriteImage(sitk.GetImageFromArray(img2save),img2savepath)
            # exit()

            if not self.text:
                return [img, img_crop, posi,img_crop1, posi1,posilabel,label,imgpath]
            if not self.cut:
                return [img, text_feature, label, imgpath]
            else:
                return [img, img_crop, posi,img_crop1, posi1,posilabel,text_feature,label,imgpath]

        except Exception as e:
            print(e)
            print(f"error: {pat_name}")
            return self.__getitem__(idx+1)


    def _read_h5(self, h5path, key):
        with h5py.File(h5path, 'r') as f:
            data = f[key][:]
        return data

    def _read_npy(self, npypath):
        data = np.load(npypath)
        return data

    def _read_nii(self,path):
        data = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(data)
        return data




class dataset_clip_twopatch_textResize(nn.Module):
    def __init__(self, args=None, rootdir=None, csvpath=None, mode='train',debug=False,cut=True,text=True):
        super(dataset_clip_twopatch_textResize, self).__init__()
        self.show = True
        self.cut = cut
        self.text = text

        if debug:
            self.imglist = [""]
            self.show_before_train(nums=1)
            exit()
        
        self.args = args
        if self.text:
            self.text_feature_pth = args.text_feature_file
            self.text_feature_dict = pickle.load(open(self.text_feature_pth, "rb"))
        
        self.imglist,self.label_list = self.read_csv(csvpath)
        if rootdir:
            self.imglist = [rootdir + img for img in self.imglist]
        # try:
        #     self.label_list = [int(label) for label in self.label_list]
        # except:
        #     self.label_list = None
        
        ###assert img exists
        # self.not_exists = []
        # for img in tqdm.tqdm(self.imglist):
        #     if not os.path.exists(img):
        #         self.not_exists.append(img)
        # if len(self.not_exists) > 0:
        #     print('not exists: ', self.not_exists)
        #     print('total not exists: ', len(self.not_exists))
        #     if_con = input('img not exists, continue? y/n')
        #     if if_con == 'y':
        #         self.imglist = [img for img in self.imglist if img not in self.not_exists]
        #     else:
        #         raise ValueError(f'{str(self.not_exists)} not exists')
        # else:
        #     print('all img exists')
        # print('total img: ', len(self.imglist))
        
        
        self.train = False
        if mode == 'train':
            self.train = True
    
    def load_text_feature(self,patid):
            text_feature = self.text_feature_dict[patid]
            return text_feature

    def img_aug(self,img):
        if random.random() < 0.5:
            img = random_augment(img)
        return img

    def show_before_train(self, nums=10):
        plt.figure(figsize=(20,20))
        for i in range(nums):
            img,imgpath = self.__getitem__(i)[0],self.__getitem__(i)[-1]
            plt.subplot(1,nums,i+1)
            try:
                plt.imshow(img)
            except:
                pass
            try:
                plt.imshow(img[0])
            except:
                plt.imshow(img[0,0,...])
                plt.title(imgpath)
        try:
            plt.savefig(self.args.save_path+'/before_train.png')
        except:
            plt.savefig('./before_train.png')
        return

    def read_csv(self, csvpath,ifcls=False):
        imglist = []
        label_list = []

        try:
            df = pd.read_csv(csvpath)
        except:
            try:
                df = pd.read_csv(csvpath,encoding='gbk')
            except:
                df = pd.read_excel(csvpath)

        ## df
        if 'path' in df.columns:
            pass
        else:
            df.columns = ['path','label']
            # raise Warning('csv should have path column,assert')
            # print(df.head())
            # # y/n
            # if_con = input('csv should have path column,assert,continue? y/n')
            # if if_con == 'y':
            #     pass
            # else:
            #     exit()
        
        imglist = df['path'].tolist()

        if 'label' in df.columns:
            label_list = df['label'].tolist()
        else:
            # raise Warning('csv should have label column,assert')
            # print(df.head())
            # # y/n
            # if_con = input('csv should have label column,assert,continue? y/n')
            # if if_con == 'y':
            #     pass
            # else:
            #     exit()
            label_list = [0]*len(imglist)
        
        dellist = []
        for i in range(len(label_list)):
            patid = imglist[i].split('/')[-2]
            if self.text:
                if patid not in self.text_feature_dict.keys():
                    # print(f'{patid} not in text feature dict')
                    dellist.append(i)
                    continue

            if os.path.exists(imglist[i]):
                pass
            else:
                imglist[i] = imglist[i].replace("")
                if os.path.exists(imglist[i]):
                    pass
                else:
                    print(f'{imglist[i]} not exists')
                    dellist.append(i)
        
        imglist = [imglist[i] for i in range(len(imglist)) if i not in dellist]
        label_list = [label_list[i] for i in range(len(label_list)) if i not in dellist]

        print(f'total img: {len(imglist)}')
        print(f'total label: {len(label_list)}')
        print(f'total not exists: {len(dellist)}')        
                
        return imglist,label_list
    

    # def read_csv(self, csvpath):
    #     imglist = []
    #     label_list = []
    #     with open(csvpath, 'r') as f:
    #         for line in f.readlines():
    #             if ',' in line:
    #                 imgname, label = line.strip().split(',')
    #             else:
    #                 imgname= line.strip()
    #                 label = None
    #             if 'shengjing' in imgname:
    #                 continue
    #             imgname = imgname#.replace("", '/a800/ssd/')
    #             pat_name = imgname.split('/')[-2]
    #             if pat_name not in self.text_feature_dict:
    #                 continue
    #             imglist.append(imgname)
    #             label_list.append(label)

    #     return imglist,label_list
    
    
    def __len__(self):
        return len(self.imglist)
    
    
    def read_img(self, imgpath):
        if imgpath.endswith('.h5'):
            try:
                img = self._read_h5(imgpath, 'image')
            except:
                img = self._read_h5(imgpath, 'data')
        elif imgpath.endswith('.npy'):
            img = self._read_npy(imgpath)
        elif imgpath.endswith('.nii.gz'):
            img = self._read_nii(imgpath)
        elif imgpath.endswith(".pt"):
            img = torch.load(imgpath)
        else:
            raise ValueError('imgpath should be h5 or npy')
        return img
    
    def standard_img(self, img, mean=None ,std=None):
        if mean and std:
            try:
                img = img.astype(np.float32)
                img = (img - mean) / (std + 1e-6)
                # img = np.expand_dims(img, axis=0)
            except:
                img = img.float()
                img = (img - mean) / (std + 1e-6)
                # img = img.unsqueeze(0)
        else:

            try:
                img = img.astype(np.float32)
                img = (img - img.mean()) / (img.std() + 1e-6)
                # img = np.expand_dims(img, axis=0)
            except:
                img = img.float()
                img = (img - img.mean()) / (img.std() + 1e-6)
                # img = img.unsqueeze(0)

        return img
    
    def random_transform_img(self, img):
        # print(img.shape)
        # img,posi = crop_rand_one_img(a=32,b=32,c=32,samples=img)
        # img,posi,img1,posi1,label = crop_rand_two_img(a=32,b=32,c=32,samples=img) # 96 96 96 patch
        # img,posi,img1,posi1,label = crop_rand_two_img(a=16,b=64,c=64,samples=img) # 48 256 256 patch
        img,posi,img1,posi1,label = crop_rand_two_img(a=24,b=64,c=64,samples=img) # 96 256 256 patch

        return img,posi,img1,posi1,label
    

    def __getitem__(self, idx):

        
        imgpath_resizeori = self.imglist[idx]
        imgpath_resize = imgpath_resizeori.replace('resized','resized96256256').replace('.npy','.pt')
        
        # resized96256256
        if not os.path.exists(imgpath_resize):
            img2resize = self.read_img(imgpath_resizeori)
            if not img2resize.shape == (48,256,256):
                return self.__getitem__(idx+1)
            else:
                img_resized = cv2resize3d(img=img2resize,size=(96,256,256),interpolation=cv2.INTER_LINEAR)
                img_resizedpt = torch.from_numpy(img_resized)
                os.makedirs(os.path.dirname(imgpath_resize),exist_ok=True)
                torch.save(img_resizedpt,imgpath_resize)
            img = img_resizedpt
        else:
            img = self.read_img(imgpath_resize)
            
        if len(img.shape) == 3:
            img = img.unsqueeze(0)


        # randint = random.randint(0,self.__len__()-1)
        # idx = randint
        # random.shuffle(self.imglist)

        imgpath = self.imglist[idx]#.replace("", '/a800/ssd/')
        pat_name = imgpath.split('/')[-2]

        try:
            if self.text:
                text_feature = self.load_text_feature(pat_name)

            # try:
            #     img = self.read_img(imgpath)
            #     if not imgpath.endswith(".pt"):
            #         img = torch.from_numpy(img)
            #     img = img.unsqueeze(0)

            # except:
            #     print("eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
            #     return self.__getitem__(idx+1)
            
            img = img.float()
            

            if self.cut:
                if len(img.shape) == 3:
                    img = img.unsqueeze(0)
                img_crop,posi,img_crop1,posi1,posilabel = self.random_transform_img(img)
                posilabel = torch.tensor(posilabel,dtype=torch.float32)
            

            label = torch.tensor(0) # 

            
            if self.train:
                # print('img',img.shape)
                if self.cut:
                    img_crop = self.img_aug(img_crop)
                    img_crop1 = self.img_aug(img_crop1)
                    # img_crop1 = img_crop1.unsqueeze(0)
                    # img_crop = img_crop.unsqueeze(0)

                    # print('img',img.shape)
                # img = img.squeeze(0)
                img = self.img_aug(img)
                # img = img.unsqueeze(0)
            
            img = self.standard_img(img)
            
            img_mean = img.mean()
            img_std = img.std()
            img_crop = self.standard_img(img_crop,mean=img_mean,std=img_std)
            img_crop1 = self.standard_img(img_crop1,mean=img_mean,std=img_std)

            if len(img.shape) == 3:
                img = img.unsqueeze(0)
            if len(img_crop.shape) == 3:
                img_crop = img_crop.unsqueeze(0)
            if len(img_crop1.shape) == 3:
                img_crop1 = img_crop1.unsqueeze(0)
            # if self.show:
            #     img = img.numpy()
            #     img2save = img[0,...]
            #     # os.makedirs("/ssd/lcc/code/ubuntu/EGFR/clip/egfr_img",exist_ok=True)
            #     sitk.WriteImage(sitk.GetImageFromArray(img2save),f"{self.args.save_path}/{idx}.nii")
            #     self.show = False
            #     exit()
            
            if not self.text:
                return [img, img_crop, posi,img_crop1, posi1,posilabel,label,imgpath]
            if not self.cut:
                return [img, text_feature, label, imgpath]
            else:
                return [img, img_crop, posi,img_crop1, posi1,posilabel,text_feature,label,imgpath]

        except Exception as e:
            print(e)
            print(f"error: {pat_name}")
            return self.__getitem__(idx+1)


    def _read_h5(self, h5path, key):
        with h5py.File(h5path, 'r') as f:
            data = f[key][:]
        return data

    def _read_npy(self, npypath):
        data = np.load(npypath)
        return data

    def _read_nii(self,path):
        data = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(data)
        return data





class dataset_clip_cls(nn.Module):
    def __init__(self, args=None, rootdir=None, csvpath=None, mode='train',debug=False,num_classes=2):
        super(dataset_clip_cls, self).__init__()
        if debug:
            self.imglist = [""]
            self.show_before_train(nums=1)
            exit()
        
        self.args = args
        try:
            self.multi_label = args.multi_label
        except:
            self.multi_label = False



        self.mode = mode
        self.imglist,self.label_list = self.read_csv(csvpath)
        if rootdir or str(args.datadir).lower() != 'none':
            rootdir = rootdir if rootdir else args.datadir
            # self.imglist = [rootdir + img for img in self.imglist]
            self.imglist = [rootdir + img.split('/')[-1] for img in self.imglist]
        if not self.multi_label:
            self.label_list = [int(label) for label in self.label_list]    
        
        
        ###assert img exists
        self.not_exists = []
        # for img in tqdm.tqdm(self.imglist):
        #     ### imgstrPath
        #     img = Path(img)
        #     if not os.path.exists(img):
        #         self.not_exists.append(img)
        # if len(self.not_exists) > 0:
        #     print('not exists: ', self.not_exists)
        #     print('total not exists: ', len(self.not_exists))
        #     self.imglist = [img for img in self.imglist if img not in self.not_exists]
        #     # if_con = input('img not exists, continue? y/n')
        #     # if if_con == 'y':
        #     #     self.imglist = [img for img in self.imglist if img not in self.not_exists]
        #     # else:
        #     #     raise ValueError(f'{str(self.not_exists)} not exists')
        # else:
        #     print('all img exists')
        # print('total img: ', len(self.imglist))
        ### ，
        
        
        

        if mode == 'train' and args.transform == True:
            self.train = True
            self.transform = M3d_transform(mode='train')
            print('using train transform M3d_transform')
        else:
            self.train = False
            self.transform = M3d_transform(mode='test')
            print('using test transform M3d_transform')
        


        self.labeldict = {}
        for i in range(len(self.imglist)):
            self.labeldict[self.imglist[i]] = self.label_list[i]
        

        # print(self.labeldict)
        self.in_channels = self.args.in_channels
        # if self.args.model in ['ori','no_causal','total_3d_no_causal','total_3d_causal']:
        #     self.in_channels = 3
        #     print("channel: 3, 1")
        # else:
        #     self.in_channels = 1
        #     print("channel: 1, 33")
        if self.args.img_aug:
            self.transform = M3d_transform(mode=mode)


    
    def show_before_train(self, nums=10):
        plt.figure(figsize=(20,20))
        for i in range(nums):
            img,imgpath = self.__getitem__(i)[0],self.__getitem__(i)[-1]
            plt.subplot(1,nums,i+1)
            try:
                plt.imshow(img)
            except:
                pass
            try:
                plt.imshow(img[0])
            except:
                plt.imshow(img[0,0,...])
                plt.title(imgpath)
        try:
            plt.savefig(self.args.save_path+'/before_train.png')
        except:
            plt.savefig('./before_train.png')
        return


    def read_csv(self, csvpath,ifcls=False):
        imglist = []
        label_list = []
        try:
            df = pd.read_csv(csvpath)
        except:
            try:
                df = pd.read_csv(csvpath,encoding='gbk')
            except:
                df = pd.read_excel(csvpath,sheet_name='Sheet1')
        
        ### dfcolumnsstr
        df.columns = [str(col) for col in df.columns]

        # ## df
        # if 'path' in str(df.columns) or 'label' in str(df.columns):
        #     pass
        # else:
        #     df.columns = ['path','label']
        #     print("csv，【path,label】")
        #     # raise Warning('csv should have path column,assert')
        #     # print(df.head())
        #     # # y/n
        #     # if_con = input('csv should have path column,assert,continue? y/n')
        #     # if if_con == 'y':
        #     #     pass
        #     # else:
        #     #     exit()
        
            
        # df['path'] = df['path'].apply(lambda x: './data/EGFR/EGFR96_pt/huaxi/share+data+huaxiEGFR+'+x.split('.pt')[0].split('/')[-1]+'+label:'+str(int(df[df["path"]==x]["label"].values[0]))+'.pt')
        # print('****************\n969696EGFR')
    
        print(df.head(),"df.head()")
        print(df.columns,"df.columns")

        if self.mode == 'test':
            self.args.path_name = self.args.tepath_name
            self.args.label_name = self.args.telabel_name



        if self.args.label_name != 'label' or self.args.path_name != 'path':
            newdf = pd.DataFrame({'path': df[self.args.path_name].tolist(), 'label': df[self.args.label_name].tolist(),self.args.path_name: df[self.args.path_name].tolist(), self.args.label_name: df[self.args.label_name].tolist()})
        else:
            newdf = pd.DataFrame({'path': df['path'].tolist(), 'label': df['label'].tolist()})

        newdf = newdf[~newdf['label'].isnull()]
        newdf = newdf[~newdf[self.args.path_name].isnull()]

        ## label

        newdf.to_csv(self.args.save_path + f'/new_{self.mode}.csv', index=False)
        print(newdf.value_counts('label'),"original")

        if self.args.oversample and self.mode == 'train':
            newdf = _over_sample(newdf)
            newdf.to_csv(self.args.save_path + f'/new_{self.mode}_oversample.csv', index=False)
            print(newdf.value_counts('label'),"oversample")
        else:
            pass
    
        imglist = newdf[self.args.path_name].tolist()
        label_list = newdf[self.args.label_name].tolist()


        dellist = []
        # for i in range(len(label_list)):
        #     if os.path.exists(str(imglist[i])):
        #         pass
        #     else:
        #         # exit(f'{imglist[i]} not exists')
        #         imglist[i] = imglist[i].replace("")
        #         if os.path.exists(imglist[i]):
        #             pass
        #         else:
        #             print(f'{imglist[i]} not exists')
        #             dellist.append(i)
        ### ，
        
        imglist = [imglist[i] for i in range(len(imglist)) if i not in dellist]
        label_list = [label_list[i] for i in range(len(label_list)) if i not in dellist]

        print(f'total img: {len(imglist)}')
        print(f'total label: {len(label_list)}')
        print(f'total not exists: {len(dellist)}')        
                
        return imglist,label_list
    
    
    
    
    
    def cut_patches(self,img,a=16,b=64,c=64):
        return crop_img_2_patch(a=16,b=64,c=64,samples=img)

    
    def __len__(self):
        return len(self.imglist)
    
    
    def read_img(self, imgpath):
        if imgpath.endswith('.h5'):
            try:
                img = self._read_h5(imgpath, 'image')
            except:
                img = self._read_h5(imgpath, 'data')
        elif imgpath.endswith('.npy'):
            img = self._read_npy(imgpath)
        elif imgpath.endswith('.nii.gz'):
            img = self._read_nii(imgpath)
        elif imgpath.endswith('.nii'):
            img = self._read_nii(imgpath)
        elif imgpath.endswith(".pt"):
            img = torch.load(imgpath,weights_only=False)
            if isinstance(img,torch.Tensor):
                img = img.numpy()
        else:
            raise ValueError('imgpath should be h5 or npy')
        
        # print(img.shape,"img.shape")
        # exit()
        return img
    
    def standard_img(self, img, args=None):
        # img = img.astype(np.float32)
        if isinstance(img,np.ndarray):
            img = torch.from_numpy(img)
        img = stardardize(x=img,args=args)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        
        # ### ch1 >> ch3
        if self.in_channels == 3:
            # print(img.shape,"img.shape")
            # print(img)
            # img = img.repeat(3,1,1,1)
            # 1 48 256 256 >> 3 48 256 256
            img3 = torch.zeros((3,img.shape[1],img.shape[2],img.shape[3]))
            img3[0] = img[0]
            img3[1] = img[0]
            img3[2] = img[0]
            img = img3
            # img = img.repeat((3, 1, 1, 1))
        
        return img
    
    def random_transform_img(self, img):
        # print(img.shape)
        img,posi = crop_rand_one_img(a=32,b=32,c=32,samples=img)
        return img,posi
    
    def img_aug(self,img):
        if random.random() < 0.5:
            img = random_augment(img)
        return img


    def __getitem__(self, idx):

        imgpath = self.imglist[idx]
        if '.' not in imgpath:
            imgpathlist = [os.path.join(imgpath,i) for i in os.listdir(imgpath)]
            if len(imgpathlist) == 0:
                try:    
                    return self.__getitem__(idx+1)
                except:
                    return self.__getitem__(idx-1)
            if self.mode == 'train':
                random.shuffle(imgpathlist)
            imgpath2read = imgpathlist[0]
        else:
            imgpath2read = imgpath

        try:
            img = self.read_img(imgpath2read)
            
        except:
            return self.__getitem__(idx+1)

        # if self.train:
        # img_crop,posi = self.random_transform_img(img)

        # if self.args.cut_patches:
        #     img_crop = self.cut_patches(img,a=16,b=64,c=64)
        #     # print(img_crop)
        #     # img_crop = torch.tensor(img_crop)
        #     for i in range(len(img_crop)):
        #         img_crop[i,:,:,:,:] = self.standard_img(img_crop[i,:,:,:,:])
        #     img_crop = img_crop.to(torch.float32)
        # else:
        #     img_crop = 0

        posi = 0
        img_crop = 0

        

        
        ## lidcresize
        if "LIDC" in self.args.trainpath and "tumor" in self.args.trainpath:
            img = np.transpose(img,(2,0,1))
            assert img.shape[-1] >= img.shape[0], img.shape
            if img.shape != self.args.img_size:
                img = cv2resize3d(img,self.args.img_size,interpolation=cv2.INTER_CUBIC)
        if self.args.img_size:
            # print(self.args.img_size,"self.args.img_size")
            # print(self.args.img_size.type,"self.args.img_size.type")
            if img.shape != self.args.img_size:
                img = cv2resize3d(img,self.args.img_size,interpolation=cv2.INTER_CUBIC)

        
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)
        img = self.transform(img)
        if self.args.img_aug:
            img = self.transform(img)

        if self.multi_label:
            try:
                # label = self.label_list[idx_random]
                label = self.labeldict[imgpath]
                label = str(label)
                label = label.split(',')
                label = [int(l) for l in label]
                label = torch.tensor(label)
            except:
                return self.__getitem__(idx+1)

        else:
            try:
                # label = self.label_list[idx_random]
                label = self.labeldict[imgpath]
                label = int(label)
            except:
                return self.__getitem__(idx+1)

            if self.args.num_classes == 1:
                assert label in [0,1], f"label: {label} not in [0,1]"
            else:
                assert label in range(self.args.num_classes), f"label: {label} not in range {self.args.num_classes}"
            
            label = torch.tensor(label)
            
            
        img = self.standard_img(img,args=self.args)
        

        return [img, img_crop, posi,label,imgpath]

    def _read_h5(self, h5path, key):
        with h5py.File(h5path, 'r') as f:
            data = f[key][:]
        return data

    def _read_npy(self, npypath):
        data = np.load(npypath)
        return data

    def _read_nii(self,path):
        data = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(data)
        return data


class dataset_clip_cox(nn.Module):
    def __init__(self, args=None, rootdir=None, csvpath=None, mode='train',debug=False,num_classes=2):
        super(dataset_clip_cox, self).__init__()
        if debug:
            self.imglist = [""]
            self.show_before_train(nums=1)
            exit()
        
        self.args = args
        try:
            self.multi_label = args.multi_label
        except:
            self.multi_label = False
        self.mode = mode
        self.imglist,self.time_list,self.event_list = self.read_csv(csvpath)
        if rootdir or str(args.datadir).lower() != 'none':
            rootdir = rootdir if rootdir else args.datadir
            # self.imglist = [rootdir + img for img in self.imglist]
            self.imglist = [rootdir + img.split('/')[-1] for img in self.imglist]
        
        if mode == 'train':
            self.train = True
            self.transform = M3d_transform(mode='train')
            print('using train transform M3d_transform')
        else:
            self.train = False
            self.transform = M3d_transform(mode='test')
            print('using test transform M3d_transform')
            
        if self.train:
            idx_list = list(range(len(self.imglist)))
            np.random.shuffle(idx_list)
            self.imglist = [self.imglist[i] for i in idx_list]
            self.time_list = [self.time_list[i] for i in idx_list]
            self.event_list = [self.event_list[i] for i in idx_list]
            
        self.labeldict = {}
        for i in range(len(self.imglist)):
            self.labeldict[self.imglist[i]] = [self.time_list[i],self.event_list[i]]
        

        # print(self.labeldict)
        self.in_channels = self.args.in_channels
        # if self.args.model in ['ori','no_causal','total_3d_no_causal','total_3d_causal']:
        #     self.in_channels = 3
        #     print("channel: 3, 1")
        # else:
        #     self.in_channels = 1
        #     print("channel: 1, 33")
        if self.args.img_aug:
            self.transform = M3d_transform(mode=mode)


    
    def show_before_train(self, nums=10):
        plt.figure(figsize=(20,20))
        for i in range(nums):
            img,imgpath = self.__getitem__(i)[0],self.__getitem__(i)[-1]
            plt.subplot(1,nums,i+1)
            try:
                plt.imshow(img)
            except:
                pass
            try:
                plt.imshow(img[0])
            except:
                plt.imshow(img[0,0,...])
                plt.title(imgpath)
        try:
            plt.savefig(self.args.save_path+'/before_train.png')
        except:
            plt.savefig('./before_train.png')
        return


    def read_csv(self, csvpath, ifcls=False):
        imglist = []
        label_list = []
        try:
            df = pd.read_csv(csvpath)
        except:
            try:
                df = pd.read_csv(csvpath,encoding='gbk')
            except:
                df = pd.read_excel(csvpath,sheet_name='Sheet1')
        
        ### dfcolumnsstr
        df.columns = [str(col) for col in df.columns]

    
        print(df.head(),"df.head()")
        print(df.columns,"df.columns")

        if self.args.trlabel_name or self.args.telabel_name:
            if self.mode == 'train':
                self.args.label_name = self.args.trlabel_name
            if self.mode == 'test':
                self.args.label_name = self.args.telabel_name
        if self.args.trpath_name or self.args.tepath_name:
            if self.mode == 'train':
                self.args.path_name = self.args.trpath_name
            if self.mode == 'test':
                self.args.path_name = self.args.tepath_name


        time_name = self.args.label_name.split(',')[0]
        event_name = self.args.label_name.split(',')[1]
        newdf = pd.DataFrame({'path': df[self.args.path_name].tolist(), 'time': df[time_name].tolist(), 'event': df[event_name].tolist(), self.args.path_name: df[self.args.path_name].tolist()})

            
        newdf = newdf[~newdf['time'].isnull()]
        newdf = newdf[~newdf['event'].isnull()]
        
        if self.args.train_data_rate < 1. and self.mode == 'train':
            # newdf = newdf.sample(frac=self.args.train_data_rate, random_state=1)
            newdf = newdf.sample(frac=self.args.train_data_rate, random_state=self.args.seed)
        elif self.args.train_data_rate > 1. and self.mode == 'train':
            # newdf = newdf.sample(n=int(self.args.train_data_rate), random_state=1)
            newdf = newdf.sample(int(self.args.train_data_rate), random_state=self.args.seed)
        else:
            pass

        newdf.to_csv(self.args.save_path + f'/new_{self.mode}.csv', index=False)
        print(newdf.value_counts('event'),"newdf.value_counts('event')")

    
        imglist = newdf[self.args.path_name].tolist()
        time_list = newdf['time'].tolist()
        event_list = newdf['event'].tolist()    


        dellist = []
        for i in range(len(label_list)):
            if os.path.exists(str(imglist[i])):
                pass
            else:
                # exit(f'{imglist[i]} not exists')
                imglist[i] = imglist[i].replace("")
                if os.path.exists(imglist[i]):
                    pass
                else:
                    print(f'{imglist[i]} not exists')
                    dellist.append(i)
        
        imglist = [imglist[i] for i in range(len(imglist)) if i not in dellist]
        time_list = [time_list[i] for i in range(len(time_list)) if i not in dellist]
        event_list = [event_list[i] for i in range(len(event_list)) if i not in dellist]

        print(f'total img: {len(imglist)}')
        print(f'total time: {len(time_list)}')
        print(f'total event: {len(event_list)}')
        print(f'total not exists: {len(dellist)}')        
                
        return imglist,time_list,event_list
    
    
    
    
    
    def cut_patches(self,img,a=16,b=64,c=64):
        return crop_img_2_patch(a=16,b=64,c=64,samples=img)

    
    def __len__(self):
        return len(self.imglist)
    
    
    def read_img(self, imgpath):
        if imgpath.endswith('.h5'):
            try:
                img = self._read_h5(imgpath, 'image')
            except:
                img = self._read_h5(imgpath, 'data')
        elif imgpath.endswith('.npy'):
            img = self._read_npy(imgpath)
        elif imgpath.endswith('.nii.gz'):
            img = self._read_nii(imgpath)
        elif imgpath.endswith('.nii'):
            img = self._read_nii(imgpath)
        elif imgpath.endswith(".pt"):
            img = torch.load(imgpath,weights_only=False)
            if isinstance(img,torch.Tensor):
                img = img.numpy()
        else:
            raise ValueError('imgpath should be h5 or npy')
        
        # print(img.shape,"img.shape")
        # exit()
        return img
    
    def standard_img(self, img, args=None):
        # img = img.astype(np.float32)
        if isinstance(img,np.ndarray):
            img = torch.from_numpy(img)
        img = stardardize(x=img,args=args)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        
        # ### ch1 >> ch3
        if self.in_channels == 3:
            # print(img.shape,"img.shape")
            # print(img)
            # img = img.repeat(3,1,1,1)
            # 1 48 256 256 >> 3 48 256 256
            img3 = torch.zeros((3,img.shape[1],img.shape[2],img.shape[3]))
            img3[0] = img[0]
            img3[1] = img[0]
            img3[2] = img[0]
            img = img3
            # img = img.repeat((3, 1, 1, 1))
        
        return img
    
    def random_transform_img(self, img):
        # print(img.shape)
        img,posi = crop_rand_one_img(a=32,b=32,c=32,samples=img)
        return img,posi
    
    def img_aug(self,img):
        if random.random() < 0.5:
            img = random_augment(img)
        return img


    def __getitem__(self, idx):

        imgpath = self.imglist[idx]
        if '.' not in imgpath:
            imgpathlist = [os.path.join(imgpath,i) for i in os.listdir(imgpath)]
            if len(imgpathlist) == 0:
                try:    
                    return self.__getitem__(idx+1)
                except:
                    return self.__getitem__(idx-1)
            if self.mode == 'train':
                random.shuffle(imgpathlist)
            imgpath2read = imgpathlist[0]
        else:
            imgpath2read = imgpath

        try:
            img = self.read_img(imgpath2read)
            
        except:
            return self.__getitem__(idx+1)

        posi = 0
        img_crop = 0

        ## lidcresize
        if "LIDC" in self.args.trainpath and "tumor" in self.args.trainpath:
            img = np.transpose(img,(2,0,1))
            assert img.shape[-1] >= img.shape[0], img.shape
            if img.shape != self.args.img_size:
                img = cv2resize3d(img,self.args.img_size,interpolation=cv2.INTER_CUBIC)
        if self.args.img_size:
            # print(self.args.img_size,"self.args.img_size")
            # print(self.args.img_size.type,"self.args.img_size.type")
            if img.shape != self.args.img_size:
                img = cv2resize3d(img,self.args.img_size,interpolation=cv2.INTER_CUBIC)

        
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)
        img = self.transform(img)
        if self.args.img_aug:
            img = self.transform(img)


        try:
            
            time,event = self.labeldict[imgpath]
            time = float(time)
            event = int(event)
        except:
            return self.__getitem__(idx+1)

        assert event in [0,1]
        
        time = torch.tensor(time)
        event = torch.tensor(event)
        label = [time,event]
            
            
        img = self.standard_img(img,args=self.args)
        

        return [img, img_crop, posi,label,imgpath]

    def _read_h5(self, h5path, key):
        with h5py.File(h5path, 'r') as f:
            data = f[key][:]
        return data

    def _read_npy(self, npypath):
        data = np.load(npypath)
        return data

    def _read_nii(self,path):
        data = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(data)
        return data




class dataset_clip_seg(nn.Module):
    def __init__(self, args=None, rootdir=None, csvpath=None, mode='train',debug=False,num_classes=2):
        super(dataset_clip_seg, self).__init__()
        if debug:
            self.imglist = [""]
            self.show_before_train(nums=1)
            exit()
        
        self.args = args
        self.mode = mode
        self.imglist,self.label_list = self.read_csv(csvpath)
        if rootdir or str(args.datadir).lower() != 'none':
            rootdir = rootdir if rootdir else args.datadir
            # self.imglist = [rootdir + img for img in self.imglist]
            self.imglist = [rootdir + img.split('/')[-1] for img in self.imglist]
        
        
        ###assert img exists
        self.not_exists = []
        for img in tqdm.tqdm(self.imglist):
            ### imgstrPath
            img = Path(img)
            if not os.path.exists(img):
                self.not_exists.append(img)
        for label in tadm.tqdm(self.label_list):
            label= Path(label)
            if not os.path.exists(label):
                self.not_exists.append(label)
    
        if len(self.not_exists) > 0:
            print('not exists: ', self.not_exists)
            print('total not exists: ', len(self.not_exists))
            if_con = input('img not exists, continue? y/n')
            if if_con == 'y':
                self.imglist = [img for img in self.imglist if img not in self.not_exists]
            else:
                raise ValueError(f'{str(self.not_exists)} not exists')
        else:
            print('all img exists')
        print('total img: ', len(self.imglist))
        

        if mode == 'train':
            self.train = True
            self.transform = M3d_transform_seg(mode='train')
            print('using train transform M3d_transform')
        else:
            self.train = False
            self.transform = M3d_transform_seg(mode='test')
            print('using test transform M3d_transform')
            
        if self.train:
            idx_list = list(range(len(self.imglist)))
            np.random.shuffle(idx_list)
            self.imglist = [self.imglist[i] for i in idx_list]
            self.label_list = [self.label_list[i] for i in idx_list]
            
        self.labeldict = {}
        for i in range(len(self.imglist)):
            self.labeldict[self.imglist[i]] = self.label_list[i]
        
        self.in_channels = self.args.in_channels

        if self.args.img_aug:
            self.transform = M3d_transform(mode=mode)

    def show_before_train(self, nums=10):
        plt.figure(figsize=(20,20))
        for i in range(nums):
            img,imgpath = self.__getitem__(i)[0],self.__getitem__(i)[-1]
            plt.subplot(1,nums,i+1)
            try:
                plt.imshow(img)
            except:
                pass
            try:
                plt.imshow(img[0])
            except:
                plt.imshow(img[0,0,...])
                plt.title(imgpath)
        try:
            plt.savefig(self.args.save_path+'/before_train.png')
        except:
            plt.savefig('./before_train.png')
        return


    def read_csv(self, csvpath,ifcls=False):
        imglist = []
        label_list = []
        try:
            df = pd.read_csv(csvpath)
        except:
            ## excelsheet1
            print("excelsheet1")
            df = pd.read_excel(csvpath,sheet_name='Sheet1')

        ## df
        if 'path' in str(df.columns) or 'label' in str(df.columns):
            pass
        else:
            df.columns = ['path','label']
            print("csv，【path,label】")

    
        print(df.head(),"df.head()")
        print(df.columns,"df.columns")

        if self.args.trlabel_name or self.args.telabel_name:
            if self.mode == 'train':
                self.args.label_name = self.args.trlabel_name
            if self.mode == 'test':
                self.args.label_name = self.args.telabel_name
        if self.args.trpath_name or self.args.tepath_name:
            if self.mode == 'train':
                self.args.path_name = self.args.trpath_name
            if self.mode == 'test':
                self.args.path_name = self.args.tepath_name

        
        if self.args.label_name != 'label' or self.args.path_name != 'path':
            newdf = pd.DataFrame({'path': df[self.args.path_name].tolist(), 'label': df[self.args.label_name].tolist(),self.args.path_name: df[self.args.path_name].tolist(),self.args.label_name: df[self.args.label_name].tolist()})
        else:
            newdf = pd.DataFrame({'path': df['path'].tolist(), 'label': df['label'].tolist()})
            
        newdf = newdf[~newdf['label'].isnull()]
        newdf = newdf[~newdf[self.args.path_name].isnull()]
        if self.args.train_data_rate < 1. and self.mode == 'train':
            # newdf = newdf.sample(frac=self.args.train_data_rate, random_state=1)
            newdf = newdf.sample(frac=self.args.train_data_rate, random_state=self.args.seed)
        elif self.args.train_data_rate > 1. and self.mode == 'train':
            # newdf = newdf.sample(n=int(self.args.train_data_rate), random_state=1)
            newdf = newdf.sample(int(self.args.train_data_rate), random_state=self.args.seed)
        else:
            pass

        newdf.to_csv(self.args.save_path + f'/new_{self.mode}.csv', index=False)
        print(newdf.value_counts('label'),"original")

        if self.args.oversample and self.mode == 'train':
            newdf = _over_sample(newdf)
            newdf.to_csv(self.args.save_path + f'/new_{self.mode}_oversample.csv', index=False)
            print(newdf.value_counts('label'),"oversample")
        else:
            pass
    
        imglist = newdf[self.args.path_name].tolist()
        label_list = newdf[self.args.label_name].tolist()


        dellist = []
        for i in range(len(label_list)):
            if os.path.exists(str(imglist[i])):
                pass
            else:
                exit(f'{imglist[i]} not exists')

        
        imglist = [imglist[i] for i in range(len(imglist)) if i not in dellist]
        label_list = [label_list[i] for i in range(len(label_list)) if i not in dellist]

        print(f'total img: {len(imglist)}')
        print(f'total label: {len(label_list)}')
        print(f'total not exists: {len(dellist)}')        
                
        return imglist,label_list
    
    def cut_patches(self,img,a=16,b=64,c=64):
        return crop_img_2_patch(a=16,b=64,c=64,samples=img)

    
    def __len__(self):
        return len(self.imglist)
    
    
    def read_img(self, imgpath):
        if imgpath.endswith('.h5'):
            try:
                img = self._read_h5(imgpath, 'image')
            except:
                img = self._read_h5(imgpath, 'data')
        elif imgpath.endswith('.npy'):
            img = self._read_npy(imgpath)
        elif imgpath.endswith('.nii.gz'):
            img = self._read_nii(imgpath)
        elif imgpath.endswith(".pt"):
            img = torch.load(imgpath,weights_only=False)
            if isinstance(img,torch.Tensor):
                img = img.numpy()
        else:
            raise ValueError('imgpath should be h5 or npy')
        
        # print(img.shape,"img.shape")
        # exit()
        return img
    
    def standard_img(self, img,args=None):
        # img = img.astype(np.float32)
        if isinstance(img,np.ndarray):
            img = torch.from_numpy(img)
        img = stardardize(x=img,args=args)
        if len(img.shape) == 3:
            img = img.unsqueeze(0)
        
        # ### ch1 >> ch3
        if self.in_channels == 3:
            # print(img.shape,"img.shape")
            # print(img)
            # img = img.repeat(3,1,1,1)
            # 1 48 256 256 >> 3 48 256 256
            img3 = torch.zeros((3,img.shape[1],img.shape[2],img.shape[3]))
            img3[0] = img[0]
            img3[1] = img[0]
            img3[2] = img[0]
            img = img3
            # img = img.repeat((3, 1, 1, 1))
        
        return img
    
    def random_transform_img(self, img):
        # print(img.shape)
        img,posi = crop_rand_one_img(a=32,b=32,c=32,samples=img)
        return img,posi
    
    def img_aug(self,img):
        if random.random() < 0.5:
            img = random_augment(img)
        return img


    def __getitem__(self, idx):

        imgpath = self.imglist[idx]
        labelpath = self.label_list[idx]

        try:
            img = self.read_img(imgpath)
            label = self.read_img(labelpath)
            
        except:
            return self.__getitem__(idx+1)

        posi = 0
        img_crop = 0
        
        ## lidcresize
        if self.args.img_size:
            if img.shape != self.args.img_size:
                img = cv2resize3d(img,self.args.img_size,interpolation=cv2.INTER_CUBIC)
                label = cv2resize3d(label,self.args.img_size,interpolation=cv2.INTER_NEAREST)

        
        if len(img.shape) == 3:
            img = np.expand_dims(img, axis=0)
            label = np.expand_dims(label, axis=0)
        img = self.transform(img,label)
        
        
        if self.args.img_aug:
            img = self.transform(img,label)
        
        img = self.standard_img(img=img,args=self.args)
        label = self.standard_label(label)
        

        return [img, img_crop, posi,label,imgpath]

    def _read_h5(self, h5path, key):
        with h5py.File(h5path, 'r') as f:
            data = f[key][:]
        return data

    def _read_npy(self, npypath):
        data = np.load(npypath)
        return data

    def _read_nii(self,path):
        data = sitk.ReadImage(path)
        data = sitk.GetArrayFromImage(data)
        return data






def cv2resize3d(img, size, interpolation=cv2.INTER_CUBIC):
    """
    img :   x y z >> x1, y1, z1

    """
    img = img.astype(np.float32)
    x,y,z = img.shape
    pointx, pointy, pointz = size
    resized_img1 = np.zeros((pointx, pointy, z))
    for z in range(img.shape[2]):
        resized_img1[:, :, z] = cv2.resize(img[:, :, z], (size[1], size[0]), interpolation=interpolation)

    #   48 256 512
        
    #   48 256 256
    resized_img = np.zeros((pointx, pointy, pointz))

    for z in range(resized_img.shape[0]):
        resized_img[z, :, :] = cv2.resize(resized_img1[z, :, :], (size[2], size[1]), interpolation=interpolation)
    
    return resized_img


if __name__ == '__main__':
    # dataset = dataset_clip(args=None, rootdir=None, csvpath=None, mode='train',debug=True)
    # x = torch.ones((48,256,256))
    # x_aug = random_augment(x)
    # print(x_aug.shape)

    dictpath = "/ssd/henan/labelEncodeEnglish/shengjing&henanEng.pkl"
    dict1 = pickle.load(open(dictpath,"rb"))
    print(dict1.keys())
    print(dict1['bm_00060694'])