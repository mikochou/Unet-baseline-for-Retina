from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score
import torch
from tqdm import tqdm
from torchvision.utils import save_image
import os
from model import UNet

from torch.utils.data import DataLoader
import Dataset
import torch.nn as nn
import torch.nn.functional as F


def val(val_loader,net,loss_func,device,bz,epoch,max_epoch, img_save_path):
    
    net.eval()
    avg_loss = 0.0
    save_path = img_save_path + '/val/'
    with torch.no_grad():
        for inputs, labels in tqdm(val_loader, desc="Epoch{}".format(max_epoch),
                                       ascii=True, total=len(val_loader)):
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = net(inputs)
                loss = loss_func(outputs, labels)
                ori_image = inputs[0]
                pred = outputs[0]
                pred = torch.sigmoid(pred)
                y = labels[0]
                pred = pred.repeat(3, 1, 1)
                y = y.repeat(3, 1, 1)
                img = torch.stack([ori_image, pred, y], 0)
                save_image(img.cpu(), os.path.join(save_path, "{}.png".format(epoch)))
                
                avg_loss = avg_loss + loss
        
    return avg_loss/bz

class DiceLoss(nn.Module):
	def __init__(self):
		super(DiceLoss, self).__init__()
 
	def	forward(self, input, target):
		N = target.size(0)
		smooth = 1
 
		input_flat = input.view(N, -1)
		target_flat = target.view(N, -1)
 
		intersection = input_flat * target_flat
 
		loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
		loss = 1 - loss.sum() / N
 
		return loss