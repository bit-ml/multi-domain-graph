from __future__ import division
import torch
from torch.autograd import Variable
from torch.utils import data

import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.utils.model_zoo as model_zoo
from torchvision import models

# general libs
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import math
import time
import os
import copy
import cv2
import tqdm

### My libs
from dataset import Generic_Test
from model import STM


def Run_video(Fs, Ms, num_frames, num_objects, model, Mem_every=None, Mem_number=None):
    # initialize storage tensors
    if Mem_every:
        to_memorize = [int(i) for i in np.arange(0, num_frames, step=Mem_every)]
    elif Mem_number:
        to_memorize = [int(round(i)) for i in np.linspace(0, num_frames, num=Mem_number+2)[:-1]]
    else:
        raise NotImplementedError

    Es = torch.zeros_like(Ms)
    Es[:,:,0] = Ms[:,:,0]

    for t in tqdm.tqdm(range(1, num_frames)):
        # memorize
        with torch.no_grad():
            prev_key, prev_value = model(Fs[:,:,t-1], Es[:,:,t-1], torch.tensor([num_objects])) 

        if t-1 == 0: # 
            this_keys, this_values = prev_key, prev_value # only prev memory
        else:
            this_keys = torch.cat([keys, prev_key], dim=3)
            this_values = torch.cat([values, prev_value], dim=3)
        
        # segment
        with torch.no_grad():
            logit = model(Fs[:,:,t], this_keys, this_values, torch.tensor([num_objects]))
        Es[:,:,t] = F.softmax(logit, dim=1)
        
        # update
        if t-1 in to_memorize:
            keys, values = this_keys, this_values
        
    pred = np.argmax(Es[0].cpu().numpy(), axis=0).astype(np.uint8)
    return pred, Es

class STMTest:

    def __init__(self, model_path):

        model = nn.DataParallel(STM())
        model.cuda()
        model.eval()
        model.load_state_dict(torch.load(model_path))

        self.model = model

    def apply(self, frames, first_frame_annotation, single_object):
        testset = Generic_Test(frames, first_frame_annotation, single_object)
        testloader = data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

        for seq, V in enumerate(testloader):
            Fs, Ms, num_objects, info = V
            seq_name = info['name'][0]
            num_frames = info['num_frames'][0].item()
            #print('[{}]: num_frames: {}, num_objects: {}'.format(seq_name, num_frames, num_objects[0][0]))
    
            pred, Es = Run_video(Fs, Ms, num_frames, num_objects, self.model, Mem_every=5, Mem_number=None)
            
            predictions = []
            for f in range(num_frames):
                c_pred = pred[f]
                f_pred = np.zeros((c_pred.shape[0], c_pred.shape[1], num_objects+1))
                for idx in range(num_objects+1):
                    if idx == 0:
                        continue
                    [ys, xs] = np.where(c_pred == idx)
                    f_pred[ys, xs, idx] = 1
                #predictions.append(pred[f])
                predictions.append(f_pred)
            return predictions

if __name__=="__main__":

    frames_path = r'/datsets_2/DAVIS/samples/semi_supervised_task/trainval/JPEGImages/480p/soapbox'
    annotation_path = r'/datsets_2/DAVIS/samples/semi_supervised_task/trainval/Annotations/480p/soapbox/00000.png'

    mask = cv2.imread(annotation_path)
    unique_colors = np.unique(mask.reshape(-1, mask.shape[-1]), axis=0, return_counts = True)[0]
    n_unique_colors = unique_colors.shape[0]
    obj_masks = np.zeros((mask.shape[0], mask.shape[1], n_unique_colors))
    for color_idx in range(n_unique_colors):
        positions = np.where((mask[:,:,0] == unique_colors[color_idx, 0]) & (mask[:,:,1] == unique_colors[color_idx, 1]) & (mask[:,:,2] == unique_colors[color_idx, 2]))
        obj_masks[positions[0], positions[1], color_idx] = 1
        #cv2.imwrite('test_%d.png'%color_idx, np.uint8(obj_masks[:,:,color_idx]*255))
    imgs = os.listdir(frames_path)
    imgs.sort()
    frames = []
    for img in imgs:
        frames.append(cv2.imread(os.path.join(frames_path, img)))

    #import pdb 
    #pdb.set_trace()
    stmtest = STMTest('STM_weights.pth')

    predictions = stmtest.apply(frames, obj_masks, 0)

    import pdb 
    pdb.set_trace()
    for idx in range(len(predictions)):
        img = np.zeros((predictions[idx].shape[0], predictions[idx].shape[1], 3))
        for obj_idx in range(n_unique_colors):
            obj_map = predictions[idx][:,:,obj_idx]
            obj_map = obj_map[:,:,None]
            obj_map = np.repeat(obj_map, 3, 2)
            color_values = unique_colors[obj_idx, :][None, None, :]
            img = img + obj_map * color_values
        cv2.imwrite('test_%d.png'%idx, np.uint8(img))
