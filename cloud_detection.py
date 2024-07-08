import datasets_here.transform as tr
import cv2
import numpy as np
import os
from PIL import Image
import random
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import albumentations as A
import albumentations.pytorch
import torchvision.transforms.functional as F


class CloudDetection(Dataset):

    def __init__(self, root, mode, nb):
        super(CloudDetection, self).__init__()
        self.root = root
        self.mode = mode
        self.nb = nb

        B2X = np.expand_dims(np.memmap(os.path.join(self.root, self.mode, 'L2A_B2.dat'), dtype='int16', mode='r', shape=(self.nb, 512, 512)), 3)
        B3X = np.expand_dims(np.memmap(os.path.join(self.root, self.mode, 'L2A_B3.dat'), dtype='int16', mode='r', shape=(self.nb, 512, 512)), 3)
        B4X = np.expand_dims(np.memmap(os.path.join(self.root, self.mode, 'L2A_B4.dat'), dtype='int16', mode='r', shape=(self.nb, 512, 512)), 3)

#        B8X = np.expand_dims(np.memmap(os.path.join(self.root, self.mode, 'L2A_B8.dat'), dtype='int16', mode='r', shape=(self.nb, 512, 512)), 3)

#        self.y = np.expand_dims(np.memmap(os.path.join(self.root, self.mode, 'LABEL_manual_hq.dat'), dtype='int8', mode='r', shape=(self.nb, 512, 512)), 3)
        self.y = np.memmap(os.path.join(self.root, self.mode, 'LABEL_manual_hq.dat'), dtype='int8', mode='r', shape=(self.nb, 512, 512))
        print('AAAAAAAAAAAAAA')

        self.X = np.concatenate((B4X, B3X, B2X), 3)
        #self.X = self.X[0:100]
        #self.y = self.y[0:100]
        
        #self.ids = data_ids
        #random.shuffle(self.ids)

        #self.transform = transforms.Compose([
        #    tr.RandomFlipOrRotate(),
        #    tr.RandomHorizontalFlip(),
        #    tr.RandomVerticalFlip(),
        #    #tr.RandomFixRotate(),
        #    #tr.GBlur(),
        #    #tr.Sharp(),
        #    #tr.Contrast(),
        #    #tr.Nothing(),

        #])
##########################################################################################
        
        # Non destructive transformations - Dehidral group D4
        self.augms = A.OneOf([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Transpose(p=0.5)
        ], p=1)

        #weak_augmentation = A.Compose([
            #A.PadIfNeeded(min_height=512, min_width=512, p=1, always_apply=True),
        #    nodestructive_pipe,
        #    albumentations.pytorch.transforms.ToTensorV2()
        #])

        self.totensor = A.Compose([
            #A.PadIfNeeded(min_height=512, min_width=512, p=1, always_apply=True),
            albumentations.pytorch.transforms.ToTensorV2()
        ])
##############################################################################################        
        
#        self.normalize = transforms.Compose([
#            transforms.ToTensor(),
#            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
#        ])

    def rand_crop_CD(self, img1, label, size):

        h = img1.shape[0]
        w = img1.shape[1]
        c_h = size[0]
        c_w = size[1]

        if h < c_h or w < c_w:
            print("Cannot crop area {} from image with size ({}, {})"
                  .format(str(size), h, w))
        else:
            s_h = random.randint(0, h-c_h)
            e_h = s_h + c_h
            s_w = random.randint(0, w-c_w)
            e_w = s_w + c_w

            crop_im1 = img1[s_h:e_h, s_w:e_w, :]
            crop_label = label[s_h:e_h, s_w:e_w]
            return crop_im1, crop_label

        
    def random_augm(self, img):
        img = np.array(img, dtype=np.uint8)
        img = Image.fromarray(img)
        rand_choice = np.random.randint(0,5)
#        factor = np.random.uniform(0.7, 1.3)
        
        brightness, contrast, saturation, hue = 0.2, 0.2, 0.2, 0.2
        
        if rand_choice==0:
            factor = np.random.uniform(max(0, 1 - brightness), 1 + brightness)
            img = F.adjust_brightness(img, factor)
        elif rand_choice==1:
            factor = np.random.uniform(max(0, 1 - contrast), 1 + contrast)
            img = F.adjust_contrast(img, factor)
        elif rand_choice==2:
            factor = np.random.uniform(max(0, 1 - saturation), 1 + saturation)
            img = F.adjust_saturation(img, factor)
        elif rand_choice==3:
            factor = np.random.uniform(-hue,hue)
            img = F.adjust_hue(img, factor)
        else:    
            img = img
        img = np.array(img) 
        img = img.clip(0,255)

        return img

    
    def stretch_16bit(self, band, lower_percent=2, higher_percent=98):
        a = 0
        b = 65535
        real_values = band.flatten()
        real_values = real_values[real_values > 0]
        c = np.percentile(real_values, lower_percent)
        d = np.percentile(real_values, higher_percent)
        t = a + (band - c) * (b - a) / float(d - c)
        t[t<a] = a
        t[t>b] = b
        t = t/256.
        t = t.astype(np.uint8) #/255.
        return t
        


    def __getitem__(self, index):
        #id = self.ids[index]

        



#        def stretch_16bit(band, lower_percent=2, higher_percent=98):
#                perc_2 = np.percentile(band, lower_percent)
#                perc_98 = np.percentile(band, higher_percent)
#                band = (band - perc_2) / (perc_98 - perc_2)
#                band[band < 0] = 0.
#                band[band > 1] = 1.

#                return band
        
        

        Ximg = np.array(self.X[index].copy())
        yimg = np.array(self.y[index].copy()).squeeze()
#        idx2 = np.where(yimg==2)
#        idx3 = np.where(yimg==3)
#        yimg[idx2]=1
#        yimg[idx3]=0
        

        #print('enter', Ximg.shape, yimg.shape)
        
        
        X_stretched = np.zeros((Ximg.shape[0], Ximg.shape[1], Ximg.shape[2]))
        for ch in range(0, Ximg.shape[2]):
            #print('ch', ch)
            band_stretched = self.stretch_16bit(Ximg[:,:,ch])
            X_stretched[:,:,ch] = band_stretched

        #print('stretch', X_stretched.shape, yimg.shape)

        #X_stretched, yimg = self.rand_crop_CD(X_stretched, yimg, [256,256])
    
        if self.mode == 'train': 
            X_stretched, yimg = self.augms(image=X_stretched, mask=yimg).values()
            #X_stretched, yimg = self.rand_crop_CD(X_stretched, yimg, [384, 384]) 
            #X_stretched = self.random_augm(X_stretched) ###################################################@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
        X_stretched = X_stretched/255.
        X_stretched, yimg = X_stretched.astype(np.float32), yimg.astype(np.float32)   
        X_stretched = np.transpose(X_stretched, (2,0,1))

        #print('final', Ximg.shape, yimg.shape)

        
#        change = torch.from_numpy(change)

        return X_stretched, yimg

    def __len__(self):
        return len(self.y)
