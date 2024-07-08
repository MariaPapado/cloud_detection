#import sys
#sys.path.append('/notebooks/GeoSeg/')
#import segmentation_models_pytorch as smp 
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
import argparse
#from networks.vit_seg_modeling import VisionTransformer as ViT_seg
#from networks.vit_seg_modeling import CONFIGS as CONFIGS_ViT_seg
#from networks.vit_seg_modeling_resnet_skip import StdConv2d
#from losses import *
#from MACUNet import *
#from marnet import marnet
#from fusion_unet import *
#from unet import unet_model
from recunet import *
import myUNF
#from geoseg.losses import *
#from catalyst.contrib.nn import Lookahead
#from catalyst import utils
print('ok')


parser = argparse.ArgumentParser()
#parser.add_argument("--data_root", type=str, default="/home/mariapap/DATASETS/second_dataset/SECOND_train_set/")
#parser.add_argument("--train_txt_file", type=str, default="/home/mariapap/DATASETS/second_dataset/SECOND_train_set/list/train.txt")
#parser.add_argument("--val_txt_file", type=str, default="/home/mariapap/DATASETS/second_dataset/SECOND_train_set/list/val.txt")
#parser.add_argument("--batch-size", type=int, default=6)
#parser.add_argument("--val-batch-size", type=int, default=8)
parser.add_argument('--vit_name', type=str,
                    default='R50-ViT-B_16', help='select one vit model')
parser.add_argument('--vit_patches_size', type=int,
                    default=16, help='vit_patches_size, default is 16')
parser.add_argument('--num_classes', type=int,
                    default=4, help='output channel of network')
parser.add_argument('--n_skip', type=int,
                    default=3, help='using number of skip-connect, default is num')
parser.add_argument('--img_size', type=int,
                    default=512, help='input patch size of network input')

args = parser.parse_args()





device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('ok')

#net = MACUNet(3, 4)


#config_vit = CONFIGS_ViT_seg[args.vit_name]
#config_vit.n_classes = args.num_classes
#config_vit.n_skip = args.n_skip

#config_vit.pretrained_path = './vit_weights/R50+ViT-B_16.npz'
#if args.vit_name.find('R50') != -1:
#    config_vit.patches.grid = (int(args.img_size / args.vit_patches_size), int(args.img_size / args.vit_patches_size))
#net = ViT_seg(config_vit, img_size=args.img_size, num_classes=config_vit.n_classes) #.to(device)
#net.load_from(weights=np.load(config_vit.pretrained_path))


#####################################################################################################################################
#conv_res_weight = net.transformer.embeddings.hybrid_model.root.conv.weight.clone()
#net.transformer.embeddings.hybrid_model.root.conv = torch.nn.Conv2d(6, 64, kernel_size=7, stride=2, bias=False, padding=3).to(device) #torch.nn.Conv2d(6, 64, kernel_size=7, stride=2, bias=False, padding=3).to(device)
#torch.nn.init.xavier_uniform_(net.transformer.embeddings.hybrid_model.root.conv.weight)
#with torch.no_grad():
#  net.transformer.embeddings.hybrid_model.root.conv.weight[:,0:3,:,:] = conv_res_weight
#net = R2AttU_Net(3,4,2)

#net.load_state_dict(torch.load('./saved_models/net_28.pt'))

#net = smp.Unet(
#        encoder_name="mobilenet_v2",
#        encoder_weights=None,
#        classes=4,
#        in_channels=3        
#)


#model_dict = {'MARNet':marnet.MARNet}
#net = model_dict['MARNet'](downsample=1, objective='dmp+amp')
#net = AttU_Net(3,2)
#net = FresUNet(3,4)

#net = unet_model.UNet(3,4)
###############################################################################################################

net = myUNF.UNetFormer(num_classes=4)
#for c in net.state_dict():
#    print(c)
#net.backbone.conv1= torch.nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
#trained_dict = torch.load('/notebooks/train_scratch/cloudnet_saved_model/net_49.pt')

#for c in trained_dict.keys():
#    if 'decoder' not in c:
#        net.state_dict()[c] = trained_dict[c]


#print('weights loaded')
###################################################################################################

#net.load_state_dict(checkdict)
#print(net.Conv_1x1)
#net.Conv_1x1 = torch.nn.Conv2d(64,4,kernel_size=1,stride=1,padding=0)
#print(net)

net.load_state_dict(torch.load('./saved_models/net_23.pt'))
net.to(device)
#print(net)


w_tensor=torch.FloatTensor(4)
w_tensor[0]= 0.4
w_tensor[1]= 0.7
w_tensor[2]= 0.99
w_tensor[3]= 0.99

w_tensor = w_tensor.to(device)

criterion = torch.nn.CrossEntropyLoss(w_tensor).to(device)
#criterion = torch.nn.CrossEntropyLoss().to(device)


base_lr = 0.00000005 #0.0005
base_wd = 0.01

#layerwise_params = {"backbone.*": dict(lr=base_lr, weight_decay=base_wd)}
#net_params = utils.process_model_params(net, layerwise_params=layerwise_params)
#base_optimizer = torch.optim.AdamW(net_params, lr=base_lr, weight_decay=base_wd)
#optimizer = Lookahead(base_optimizer)
#lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)


optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)
#optimizer = torch.optim.AdamW(net.parameters(), lr=base_lr, weight_decay=base_wd)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode="min", factor=0.1, patience=2, verbose=True)

batch_size = 4
val_batch_size=2

epochs = 50
num_classes = 4

print('loaders')

trainset = cloud_detection.CloudDetection('/notebooks/CloudSEN12-high/', 'train', 8490)
print('0')
valset = cloud_detection.CloudDetection('/notebooks/CloudSEN12-high/', 'val', 535)
print('1')
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True,
                                      pin_memory=False, drop_last=False)
print('2')
valloader = DataLoader(valset, batch_size=val_batch_size, shuffle=False,
                                    pin_memory=True, drop_last=False)
print('3')

print('calculating alpha..')
#alpha = get_alpha(trainloader)
# alpha-0 (no-cloud)=1216263864, alpha-1 (thick-cloud)=596674914, alpha-2 (thin-cloud)=215404610, alpha-3 (shadow)=197259172
alpha = [1216263864,596674914,215404610,197259172]
##alpha = [, , , ]
##print('perc ch', alpha[1]/alpha[0])
print(f"alpha-0 (no-cloud)={alpha[0]}, alpha-1 (thick-cloud)={alpha[1]}, alpha-2 (thin-cloud)={alpha[2]}, alpha-3 (shadow)={alpha[3]}")
#criterion = FocalLoss(apply_nonlin = softmax_helper, alpha = alpha, gamma = 2, smooth = 1e-5).to(device)


print('opttttttttttt', optimizer.param_groups[0]['lr'])

total_iters = len(trainloader) * epochs
print('totaliters', total_iters)
save_folder = 'saved_models' #where to save the models and training progress
#if os.path.exists(save_folder):
#    shutil.rmtree(save_folder)
#os.mkdir(save_folder)
ff=open('./' + save_folder + '/progress.txt','w')
iter_ = 0

iters = len(trainloader)
for epoch in range(24, epochs+1):

    net.train()
    train_losses = []
    confusion_matrix = tnt.meter.ConfusionMeter(4, normalized=True)
    for i, batch in enumerate(tqdm(trainloader)):
        Ximg, y = batch
        #print(Ximg.shape, y.shape)
       # print('uni', np.unique(y.data.cpu().numpy()))

        imgs, labels = batch
#################################################################################################################################

        #img_save = imgs[0].permute(1,2,0).data.cpu().numpy()
        #lbl_save = labels[0].squeeze().data.cpu().numpy()
        
        #cv2.imwrite('./checks/imgs/img_{}.png'.format(i), img_save[:,:,[2,1,0]]*256)
        #cv2.imwrite('./checks/labels/lbl_{}.png'.format(i), lbl_save*256)

#################################################################################################################################


        imgs, labels = imgs.float().to(device), labels.long().to(device)
        #print(imgs)
        optimizer.zero_grad()
     #   preds, d0, d1, d2, d3, d4, amp41, amp31, amp21, amp11, amp01 = net(imgs)
        preds, _ = net(imgs)
        #print('bbbbbb', imgs.shape, labels.shape, preds.shape)

    
        label_conf, pred_conf = labels.flatten(), torch.argmax(preds,1).flatten()
        
        confusion_matrix.add(label_conf, pred_conf)
        
        loss = criterion(preds, labels)
        train_losses.append(loss.item())

        loss.backward()
        optimizer.step()
        iter_ += 1
        #lr_scheduler.step(epoch + i / iters)
        #lr_ = base_lr * (1.0 - iter_ / total_iters) ** 0.9
        #for param_group in optimizer.param_groups:
        #    param_group['lr'] = lr_


            
        if iter_ % 20 == 0:
            pred = preds[0]
            pred = torch.softmax(pred, 0)
            pred = np.argmax(pred.data.cpu().numpy(), axis=0)
            gt = labels.data.cpu().numpy()[0]
            print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss_CH: {:.6f}\tAccuracy: {}'.format(
                      epoch, epochs, i, len(trainloader),100.*i/len(trainloader), loss.item(), tools.accuracy(pred, gt)))
        
        



    train_acc=(np.trace(confusion_matrix.conf)/float(np.ndarray.sum(confusion_matrix.conf))) *100
    print('TRAIN_LOSS: ', '%.3f' % np.mean(train_losses), 'TRAIN_ACC: ', '%.3f' % train_acc)
    prec, rec, f1 = tools.metrics(confusion_matrix.conf)
    print ('Precision: {}\nRecall: {}\nF1: {}'.format(prec, rec, f1)) 
    
    confusion_matrix = tnt.meter.ConfusionMeter(4, normalized=True)
    with torch.no_grad():
        net.eval()
        val_losses = []

        for i, batch in enumerate(tqdm(valloader)):
            imgs, labels = batch
            imgs, labels = imgs.float().to(device), labels.long().to(device)
#            preds, d0, d1, d2, d3, d4, amp41, amp31, amp21, amp11, amp01 = net(imgs)            
            preds = net(imgs)
            label_conf, pred_conf = labels.flatten(), torch.argmax(torch.softmax(preds, 1),1).flatten()
            confusion_matrix.add(label_conf, pred_conf)


            loss = criterion(preds, labels)
            val_losses.append(loss.item())


        scheduler.step(np.mean(val_losses))    
        test_acc=(np.trace(confusion_matrix.conf)/float(np.ndarray.sum(confusion_matrix.conf))) *100
        print('VAL_LOSS: ', '%.3f' % np.mean(val_losses), 'VAL_ACC: ', '%.3f' % test_acc)
        print('VALIDATION CONFUSION MATRIX')    
        print(confusion_matrix.conf)
        prec, rec, f1 = tools.metrics(confusion_matrix.conf)
        print ('Precision: {}\nRecall: {}\nF1: {}'.format(prec, rec, f1)) 
    

    tools.write_results(ff, save_folder, epoch, train_acc, test_acc, np.mean(train_losses), np.mean(val_losses), confusion_matrix.conf, prec, rec, f1, optimizer.param_groups[0]['lr'])

    #save model in every epoch
    torch.save(net.state_dict(), './' + save_folder + '/net_{}.pt'.format(epoch))



