import segmentation_models_pytorch as smp 
import os
import tools
import torch
import numpy as np
import rasterio as rio
import warnings
import cv2
from datasets_here import cloud_detection
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import shutil
import torchnet as tnt
from PIL import Image
from unet import unet_model
from recunet import *
import myUNF

def stretch_8bit(band, lower_percent=2, higher_percent=98):
    a = 0
    #b = 65535
    b = 255
    real_values = band.flatten()
    real_values = real_values[real_values > 0]
    c = np.percentile(real_values, lower_percent)
    d = np.percentile(real_values, higher_percent)
    t = a + (band - c) * (b - a) / float(d - c)
    t[t<a] = a
    t[t>b] = b
    return t.astype(np.uint8)/255.


device = 'cuda' if torch.cuda.is_available() else 'cpu'

#net = smp.Unet(
#        encoder_name="mobilenet_v2",
#        encoder_weights=None,
#        classes=4,
#        in_channels=3        
#)

#net = unet_model.UNet(3,4)
#net = U_Net(3,2)

net = myUNF.UNetFormer(num_classes=4)
net.load_state_dict(torch.load('./myUNF_RGB_nobrightaugm_SUCCESS1/net_27.pt'))
#net.load_state_dict(torch.load('./UNet_augm_saved_models/net_29.pt'))

net.to(device)
net.eval()


ids = os.listdir('./test_customer_imgs/')

#w, h = 512/10
w, h = 256, 256
w_s, h_s = 52, 52
w_div, h_div = 64, 64


def pad_left(arr, cnt, n=256): 
    deficit_x = (n - arr.shape[1] % n) 
    deficit_y = (n - arr.shape[2] % n) 
    if not (arr.shape[1] % n): 
        deficit_x = 0 
    if not (arr.shape[2] % n): 
        deficit_y = 0 
    arr = np.pad(arr, ((0, 0), (deficit_x, 0), (deficit_y, 0)), mode='reflect') 
    cv2.imwrite('./pads/padded/{}_.png'.format(cnt), np.transpose(arr[[2,1,0],:,:], (1,2,0))*256)
    return arr, deficit_x, deficit_y

def predict_clouds(img, cnt, scale_percent=10):
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    x_arr = img.copy()

    x_arr = cv2.resize(x_arr, dim, interpolation = cv2.INTER_AREA)
#    x_arr = x_arr[:,:,[2,1,0]]
    x_arr = np.transpose(x_arr, [2, 0, 1])
    im_test, def_x, def_y = pad_left(x_arr, cnt)
    #result = self.prediction_step(im_test)
    #result = result[def_x:, def_y:]
    
    #result = cv2.resize(result, (img.shape[1], img.shape[0]), interpolation = cv2.INTER_NEAREST)
    
    return im_test, def_x, def_y


def stretch_8bit(band, lower_percent=2, higher_percent=98):
    a = 0
    b = 255
    real_values = band.flatten()
    real_values = real_values[real_values > 0]
    c = np.percentile(real_values, lower_percent)
    d = np.percentile(real_values, higher_percent)
    t = a + (band - c) * (b - a) / float(d - c)
    t[t<a] = a
    t[t>b] = b
    return t.astype(np.uint8)/255.


for cnt, id in enumerate(tqdm(ids)):
    patch_div = np.zeros((w_div, h_div, 3))

    im = Image.open('./test_customer_imgs/{}'.format(id))
    im = np.array(im)/255.0  #(512, 512, 3)
    print('im', im.shape)   
    #im = stretch_8bit(im)
    
    im_zeros = np.zeros((im.shape[0], im.shape[1], im.shape[2]))
    for ch in range(0, im.shape[2]):
        stre = stretch_8bit(im[:,:,ch])
        im_zeros[:,:,ch] = stre    

    rim, def_x, def_y = predict_clouds(im,cnt)
    print(rim.shape) #(3, 256, 256)
    
#    rim = rim[:, def_x:, def_y:]
#    print('AAAAAAAAAAAAAAAA', rim.shape, im.shape) #(3, 51, 51) (512, 512, 3)
#    rim = np.transpose(rim, (1,2,0))
#    rim = cv2.resize(rim, (im.shape[1], im.shape[0]), interpolation = cv2.INTER_NEAREST)
#    print('rrrrrrrrr', rim.shape)
#    cv2.imwrite('./customer_results/{}'.format(id), rim*256)
    
    
    
    

    rim = torch.from_numpy(rim).unsqueeze(0).float().cuda()
    out = net(rim)
    #out = torch.softmax(out, 1)
    out = torch.argmax(out, 1)
    out = out.squeeze().data.float().cpu().numpy()
    cv2.imwrite('./customer_results/{}'.format(id), out*256)

    out = out[def_x:, def_y:]
    
    out = cv2.resize(out, (im.shape[1], im.shape[0]), interpolation = cv2.INTER_NEAREST)
    
    
    print('uni', np.unique(out))

#    cv2.imwrite('./customer_results/{}'.format(id), out*256)
    

