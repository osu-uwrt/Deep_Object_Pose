#!/usr/bin/env python3

# Copyright (c) 2018 NVIDIA Corporation. All rights reserved.
# This work is licensed under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License.
# https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode

from __future__ import print_function

######################################################
"""
REQUIREMENTS:
simplejson==3.16.0
numpy==1.14.1
opencv_python==3.4.3.18
horovod==0.13.5
photutils==0.5
scipy==1.1.0
torch==0.4.0
pyquaternion==0.9.2
tqdm==4.25.0
pyrr==0.9.2
Pillow==5.2.0
torchvision==0.2.1
PyYAML==3.13
"""

######################################################
"""
HOW TO TRAIN DOPE

This is the DOPE training code.  
It is provided as a convenience for researchers, but it is otherwise unsupported.

Please refer to `python3 train.py --help` for specific details about the 
training code. 

If you download the FAT dataset 
(https://research.nvidia.com/publication/2018-06_Falling-Things)
you can train a YCB object DOPE detector as follows: 

```
python3 train.py --data path/to/FAT --object soup --outf soup 
--gpuids 0 1 2 3 4 5 6 7 
```

This will create a folder called `train_soup` where the weights will be saved 
after each epoch. It will use the 8 gpus using pytorch data parallel. 
"""

import argparse
import configparser
import random
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.utils.data as data
import torchvision.models as models
from torch.cuda import amp
import datetime
import json
import glob
import os

from PIL import Image
from PIL import ImageDraw

from math import acos
from math import sqrt
from math import pi    

from os.path import exists

import cv2
import colorsys,math
from tqdm import tqdm
import time
import albumentations as A

import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"]="0,1,2,3,4,5,6,7"


##################################################
# NEURAL NETWORK MODEL
##################################################

class DopeNetwork(nn.Module):
    def __init__(
            self,
            pretrained=False,
            numBeliefMap=9,
            numAffinity=16,
            stop_at_stage=6  # number of stages to process (if less than total number of stages)
        ):
        super(DopeNetwork, self).__init__()

        self.stop_at_stage = stop_at_stage
        
        if pretrained is False:
            print("Training network without imagenet weights.")
        else:
            print("Training network pretrained on imagenet.")

        vgg_full = models.vgg19(pretrained=pretrained).features
        self.vgg = nn.Sequential()
        for i_layer in range(24):
            self.vgg.add_module(str(i_layer), vgg_full[i_layer])

        # Add some layers
        i_layer = 23
        self.vgg.add_module(str(i_layer), nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer+1), nn.ReLU(inplace=True))
        self.vgg.add_module(str(i_layer+2), nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1))
        self.vgg.add_module(str(i_layer+3), nn.ReLU(inplace=True))

        # print('---Belief------------------------------------------------')
        # _2 are the belief map stages
        self.m1_2 = DopeNetwork.create_stage(128, numBeliefMap, True)
        self.m2_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m3_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m4_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m5_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)
        self.m6_2 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numBeliefMap, False)

        # print('---Affinity----------------------------------------------')
        # _1 are the affinity map stages
        self.m1_1 = DopeNetwork.create_stage(128, numAffinity, True)
        self.m2_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m3_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m4_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m5_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)
        self.m6_1 = DopeNetwork.create_stage(128 + numBeliefMap + numAffinity,
                                             numAffinity, False)


    def forward(self, x):
        '''Runs inference on the neural network'''

        out1 = self.vgg(x)

        out1_2 = self.m1_2(out1)
        out1_1 = self.m1_1(out1)

        if self.stop_at_stage == 1:
            return [out1_2],\
                   [out1_1]

        out2 = torch.cat([out1_2, out1_1, out1], 1)
        out2_2 = self.m2_2(out2)
        out2_1 = self.m2_1(out2)

        if self.stop_at_stage == 2:
            return [out1_2, out2_2],\
                   [out1_1, out2_1]

        out3 = torch.cat([out2_2, out2_1, out1], 1)
        out3_2 = self.m3_2(out3)
        out3_1 = self.m3_1(out3)

        if self.stop_at_stage == 3:
            return [out1_2, out2_2, out3_2],\
                   [out1_1, out2_1, out3_1]

        out4 = torch.cat([out3_2, out3_1, out1], 1)
        out4_2 = self.m4_2(out4)
        out4_1 = self.m4_1(out4)

        if self.stop_at_stage == 4:
            return [out1_2, out2_2, out3_2, out4_2],\
                   [out1_1, out2_1, out3_1, out4_1]

        out5 = torch.cat([out4_2, out4_1, out1], 1)
        out5_2 = self.m5_2(out5)
        out5_1 = self.m5_1(out5)

        if self.stop_at_stage == 5:
            return [out1_2, out2_2, out3_2, out4_2, out5_2],\
                   [out1_1, out2_1, out3_1, out4_1, out5_1]

        out6 = torch.cat([out5_2, out5_1, out1], 1)
        out6_2 = self.m6_2(out6)
        out6_1 = self.m6_1(out6)

        return [out1_2, out2_2, out3_2, out4_2, out5_2, out6_2],\
               [out1_1, out2_1, out3_1, out4_1, out5_1, out6_1]
                        
    @staticmethod
    def create_stage(in_channels, out_channels, first=False):
        '''Create the neural network layers for a single stage.'''

        model = nn.Sequential()
        mid_channels = 128
        if first:
            padding = 1
            kernel = 3
            count = 6
            final_channels = 512
        else:
            padding = 3
            kernel = 7
            count = 10
            final_channels = mid_channels

        # First convolution
        model.add_module("0",
                         nn.Conv2d(
                             in_channels,
                             mid_channels,
                             kernel_size=kernel,
                             stride=1,
                             padding=padding)
                        )

        # Middle convolutions
        i = 1
        while i < count - 1:
            model.add_module(str(i), nn.ReLU(inplace=True))
            i += 1
            model.add_module(str(i),
                             nn.Conv2d(
                                 mid_channels,
                                 mid_channels,
                                 kernel_size=kernel,
                                 stride=1,
                                 padding=padding))
            i += 1

        # Penultimate convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(mid_channels, final_channels, kernel_size=1, stride=1))
        i += 1

        # Last convolution
        model.add_module(str(i), nn.ReLU(inplace=True))
        i += 1
        model.add_module(str(i), nn.Conv2d(final_channels, out_channels, kernel_size=1, stride=1))
        i += 1

        return model



##################################################
# UTILS CODE FOR LOADING THE DATA
##################################################

def default_loader(path):
    return Image.open(path).convert('RGB')          

def loadjson(path, objectsofinterest, img):
    """
    Loads the data from a json file. 
    If there are no objects of interest, then load all the objects. 
    """
    with open(path) as data_file:    
        data = json.load(data_file)
    # print (path)
    pointsBelief = []
    points_keypoints_2d = []
    pointsBoxes = []
    centroids = []

    translations = []
    rotations = []
    points = []

    for i_line in range(len(data['objects'])):
        info = data['objects'][i_line]
        if not objectsofinterest is None and \
           not objectsofinterest in info['class'].lower():
            continue 
        
        box = info['bounding_box']
        boxToAdd = []

        boxToAdd.append(float(box['top_left'][0]))
        boxToAdd.append(float(box['top_left'][1]))
        boxToAdd.append(float(box["bottom_right"][0]))
        boxToAdd.append(float(box['bottom_right'][1]))

        boxpoint = [(boxToAdd[0],boxToAdd[1]),(boxToAdd[0],boxToAdd[3]),
                    (boxToAdd[2],boxToAdd[1]),(boxToAdd[2],boxToAdd[3])]

        pointsBoxes.append(boxpoint)
        
        # 3dbbox with belief maps
        points3d = []
        
        pointdata = info['projected_cuboid']
        for p in pointdata:
            points3d.append((p[0],p[1]))

        # Get the centroids
        pcenter = info['projected_cuboid_centroid']

        points3d.append ((pcenter[0],pcenter[1]))
        pointsBelief.append(points3d)
        points.append (points3d + [(pcenter[0],pcenter[1])])
        centroids.append((pcenter[0],pcenter[1]))

        # load translations
        location = info['location']
        translations.append([location[0],location[1],location[2]])

        # quaternion
        rot = info["quaternion_xyzw"]
        rotations.append(rot)

    return {
        "pointsBelief":pointsBelief, 
        "rotations":rotations,
        "translations":translations,
        "centroids":centroids,
        "points":points,
        "keypoints_2d":points_keypoints_2d,
        }

def loadimages(root):
    """
    Find all the images in the path and folders, return them in imgs. 
    """
    imgs = []

    def add_json_files(path,):
        for imgpath in glob.glob(path+"/*.png"):
            if exists(imgpath) and exists(imgpath.replace('png',"json")):
                imgs.append((imgpath,imgpath.replace(path,"").replace("/",""),
                    imgpath.replace('png',"json")))
        for imgpath in glob.glob(path+"/*.jpg"):
            if exists(imgpath) and exists(imgpath.replace('jpg',"json")):
                imgs.append((imgpath,imgpath.replace(path,"").replace("/",""),
                    imgpath.replace('jpg',"json")))

    def explore(path):
        if not os.path.isdir(path):
            return
        folders = [os.path.join(path, o) for o in os.listdir(path) 
                        if os.path.isdir(os.path.join(path,o))]
        if len(folders)>0:
            for path_entry in folders:                
                explore(path_entry)
        else:
            add_json_files(path)

    explore(root)

    return imgs

class MultipleVertexJson(data.Dataset):
    """
    Dataloader for the data generated by NDDS (https://github.com/NVIDIA/Dataset_Synthesizer). 
    This is the same data as the data used in FAT.
    """
    def __init__(self, root, preprocessing_transform, transform=None,
            normal = None, test=False,
            loader = default_loader, 
            objectsofinterest = "",
            save = False,  
            data_size = None,
            sigma = 16
            ):
        ###################
        self.objectsofinterest = objectsofinterest
        self.loader = loader
        self.transform = transform
        self.root = root
        self.imgs = []
        self.test = test
        self.normal = normal
        self.save = save 
        self.data_size = data_size
        self.sigma = sigma
        self.preprocessing_transform = preprocessing_transform

        def load_data(path):
            '''Recursively load the data.  This is useful to load all of the FAT dataset.'''
            imgs = loadimages(path)

            # Check all the folders in path 
            for name in os.listdir(str(path)):
                imgs += loadimages(path +"/"+name)
            return imgs


        self.imgs = load_data(root)

        # Shuffle the data, this is useful when we want to use a subset. 
        np.random.shuffle(self.imgs)

    def __len__(self):
        # When limiting the number of data
        if not self.data_size is None:
            return int(self.data_size)

        return len(self.imgs)   

    def __getitem__(self, index):
        """
        Depending on how the data loader is configured,
        this will return the debug info with the cuboid drawn on it, 
        this happens when self.save is set to true. 
        Otherwise, during training this function returns the 
        belief maps and affinity fields and image as tensors.  
        """
        path, name, txt = self.imgs[index]
        img = self.loader(path)

        loader = loadjson
        
        data = loader(txt, self.objectsofinterest,img)

        pointsBelief        =   data['pointsBelief'] 
        objects_centroid    =   data['centroids']
        points_keypoints    =   data['keypoints_2d']

        # Note:  All point coordinates are in the image space, e.g., pixel value.
        # This is used when we do saving --- helpful for debugging
        if self.test:   
            # Use the save to debug the data
            draw = ImageDraw.Draw(img)
            
            # PIL drawing functions, here for sharing draw
            def DrawKeypoints(points):
                for key in points:
                    DrawDot(key,(12, 115, 170),7) 
                                       
            def DrawLine(point1, point2, lineColor, lineWidth):
                if not point1 is None and not point2 is None:
                    draw.line([point1,point2],fill=lineColor,width=lineWidth)

            def DrawDot(point, pointColor, pointRadius):
                if not point is None:
                    xy = [point[0]-pointRadius, point[1]-pointRadius, point[0]+pointRadius, point[1]+pointRadius]
                    draw.ellipse(xy, fill=pointColor, outline=pointColor)

            def DrawCube(points, color = None):
                '''Draw cube with a thick solid line across the front top edge.'''
                lineWidthForDrawing = 2
                lineColor = (255, 215, 0)  # yellow-ish

                if not color is None:
                    lineColor = color        

                # draw front
                DrawLine(points[0], points[1], lineColor, 8) #lineWidthForDrawing)
                DrawLine(points[1], points[2], lineColor, lineWidthForDrawing)
                DrawLine(points[3], points[2], lineColor, lineWidthForDrawing)
                DrawLine(points[3], points[0], lineColor, lineWidthForDrawing)
                
                # draw back
                DrawLine(points[4], points[5], lineColor, lineWidthForDrawing)
                DrawLine(points[6], points[5], lineColor, lineWidthForDrawing)
                DrawLine(points[6], points[7], lineColor, lineWidthForDrawing)
                DrawLine(points[4], points[7], lineColor, lineWidthForDrawing)
                
                # draw sides
                DrawLine(points[0], points[4], lineColor, lineWidthForDrawing)
                DrawLine(points[7], points[3], lineColor, lineWidthForDrawing)
                DrawLine(points[5], points[1], lineColor, lineWidthForDrawing)
                DrawLine(points[2], points[6], lineColor, lineWidthForDrawing)

                # draw dots
                DrawDot(points[0], pointColor=(255,255,255), pointRadius = 3)
                DrawDot(points[1], pointColor=(0,0,0), pointRadius = 3)

            # Draw all the found objects. 
            for points_belief_objects in pointsBelief:
                DrawCube(points_belief_objects)
            for keypoint in points_keypoints:
                DrawKeypoints(keypoint)

            img = Image.fromarray(np.array(img))

        transform = self.preprocessing_transform
        if self.transform is not None:
            transform = A.Compose([self.transform, self.preprocessing_transform])

        keypoints = list(map(tuple, np.array(pointsBelief).reshape(-1, 2)))
        centroids = list(map(tuple, objects_centroid))
        transformed = transform(image=np.array(img), keypoints = keypoints, centroids=centroids)
        img = transformed['image']
        keypoints = np.array(transformed['keypoints']).reshape(np.array(pointsBelief).shape)
        centroids = transformed['centroids']

        beliefs = CreateBeliefMap(
            img, 
            pointsBelief=keypoints,
            nbpoints = 9,
            sigma = self.sigma)

        beliefs = transforms.Resize((img.shape[2] // 8, img.shape[1] // 8))(beliefs)
        

        affinities = GenerateMapAffinity(img,8,keypoints,centroids,8)

        return {
            'image': img,
            'beliefs': beliefs,
            'affinities': affinities,
        }

"""
Some simple vector math functions to find the angle
between two points, used by affinity fields. 
"""
def length(v):
    return sqrt(v[0]**2+v[1]**2)

def dot_product(v,w):
   return v[0]*w[0]+v[1]*w[1]

def normalize(v):
    norm=np.linalg.norm(v, ord=1)
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm

def determinant(v,w):
   return v[0]*w[1]-v[1]*w[0]

def inner_angle(v,w):
   cosx=dot_product(v,w)/(length(v)*length(w))
   rad=acos(cosx) # in radians
   return rad*180/pi # returns degrees

def py_ang(A, B=(1,0)):
    inner=inner_angle(A,B)
    det = determinant(A,B)
    if det<0: #this is a property of the det. If the det < 0 then B is clockwise of A
        return inner
    else: # if the det > 0 then A is immediately clockwise of B
        return 360-inner

def GenerateMapAffinity(img,nb_vertex,pointsInterest,objects_centroid,scale):
    """
    Function to create the affinity maps, 
    e.g., vector maps pointing toward the object center. 

    Args:
        img: PIL image
        nb_vertex: (int) number of points 
        pointsInterest: list of points 
        objects_centroid: (x,y) centroids for the obects
        scale: (float) by how much you need to scale down the image 
    return: 
        return a list of tensors for each point except centroid point      
    """

    # Apply the downscale right now, so the vectors are correct. 
    img_affinity = Image.new('RGB', (int(img.shape[0]/scale),int(img.shape[1]/scale)), "black")

    affinities = []
    for i_points in range(nb_vertex):
        affinities.append(torch.zeros(2,int(img.shape[1]/scale),int(img.shape[2]/scale)))
    
    for i_pointsImage in range(len(pointsInterest)):    
        pointsImage = pointsInterest[i_pointsImage]
        center = objects_centroid[i_pointsImage]
        for i_points in range(nb_vertex):
            point = pointsImage[i_points]
            affinity_pair, img_affinity = getAffinityCenter(int(img.shape[2]/scale),
                int(img.shape[1]/scale),
                tuple((np.array(pointsImage[i_points])/scale).tolist()),
                tuple((np.array(center)/scale).tolist()), 
                img_affinity = img_affinity, radius=1)

            affinities[i_points] = (affinities[i_points] + affinity_pair)/2


            # Normalizing
            v = affinities[i_points].numpy()                    
            
            xvec = v[0]
            yvec = v[1]

            norms = np.sqrt(xvec * xvec + yvec * yvec)
            nonzero = norms > 0

            xvec[nonzero]/=norms[nonzero]
            yvec[nonzero]/=norms[nonzero]

            affinities[i_points] = torch.from_numpy(np.concatenate([[xvec],[yvec]]))
    affinities = torch.cat(affinities,0)

    return affinities

def getAffinityCenter(width, height, point, center, radius=7, img_affinity=None):
    """
    Function to create the affinity maps, 
    e.g., vector maps pointing toward the object center. 

    Args:
        width: image wight
        height: image height
        point: (x,y) 
        center: (x,y)
        radius: pixel radius
        img_affinity: tensor to add to 
    return: 
        return a tensor
    """
    tensor = torch.zeros(2,height,width).float()

    # Create the canvas for the afinity output
    imgAffinity = Image.new("RGB", (width,height), "black")
    
    draw = ImageDraw.Draw(imgAffinity)    
    r1 = radius
    p = point
    draw.ellipse((p[0]-r1,p[1]-r1,p[0]+r1,p[1]+r1),(255,255,255))

    del draw

    # Compute the array to add the afinity
    array = (np.array(imgAffinity)/255)[:,:,0]

    angle_vector = np.array(center) - np.array(point)
    angle_vector = normalize(angle_vector)
    affinity = np.concatenate([[array*angle_vector[0]],[array*angle_vector[1]]])

    # print (tensor)
    if not img_affinity is None:
        # Find the angle vector
        # print (angle_vector)
        if length(angle_vector) >0:
            angle=py_ang(angle_vector)
        else:
            angle = 0
        # print(angle)
        c = np.array(colorsys.hsv_to_rgb(angle/360,1,1)) * 255
        draw = ImageDraw.Draw(img_affinity)    
        draw.ellipse((p[0]-r1,p[1]-r1,p[0]+r1,p[1]+r1),fill=(int(c[0]),int(c[1]),int(c[2])))
        del draw
    re = torch.from_numpy(affinity).float() + tensor
    return re, img_affinity

       
def CreateBeliefMap(img,pointsBelief,nbpoints,sigma=16):
    """
    Args: 
        img: image
        pointsBelief: list of points in the form of 
                      [nb object, nb points, 2 (x,y)] 
        nbpoints: (int) number of points, DOPE uses 8 points here
        sigma: (int) size of the belief map point
    return: 
        return an array of PIL black and white images representing the 
        belief maps         
    """
    beliefsImg = []
    sigma = int(sigma)
    for numb_point in range(nbpoints):    
        array = np.zeros((img.shape[2], img.shape[1]))

        for point in pointsBelief:
            p = point[numb_point]
            w = int(sigma*2)
            if p[0]-w>=0 and p[0]+w<img.shape[2] and p[1]-w>=0 and p[1]+w<img.shape[1]:
                for x in range(int(p[0])-w, int(p[0])+w):
                    for y in range(int(p[1])-w, int(p[1])+w):
                        array[y,x] = np.exp(-(((x - p[0])**2 + (y - p[1])**2)/(2*(sigma**2))))

        beliefsImg.append(array)
    return torch.from_numpy(np.array(beliefsImg))


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, img_range=None, scale_each=False, pad_value=0):
    """
    Make a grid of images.
    
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The Final grid size is (B / nrow, nrow). Default is 8.
        padding (int, optional): amount of padding. Default is 2.
        normalize (bool, optional): If True, shift the image to the img_range (0, 1),
            by subtracting the minimum and dividing by the maximum pixel value.
        img_range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If True, scale each image in the batch of
            images separately rather than the (min, max) over all images.
        pad_value (float, optional): Value for the padded pixels.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.view(1, tensor.size(0), tensor.size(1))
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.view(1, tensor.size(0), tensor.size(1), tensor.size(2))

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize == True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if img_range is not None:
            assert isinstance(img_range, tuple), \
                "img_range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, img_range):
            if img_range is not None:
                norm_ip(t, img_range[0], img_range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each == True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, img_range)
        else:
            norm_range(tensor, img_range)

    if tensor.size(0) == 1:
        return tensor.squeeze()

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    grid = tensor.new(3, height * ymaps + padding, width * xmaps + padding).fill_(pad_value)
    k = 0
    for y in range(ymaps):
        for x in range(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def save_image(tensor, filename, nrow=4, padding=2,mean=None, std=None):
    """
    Saves a given Tensor into an image file.
    If given a mini-batch tensor, will save the tensor as a grid of images.
    """
    from PIL import Image
    
    tensor = tensor.cpu()
    grid = make_grid(tensor, nrow=nrow, padding=10,pad_value=1)
    if mean is None:
        ndarr = grid.mul(0.5).add(0.5).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
    else:
        ndarr = grid.mul(std).add(mean).mul(255).byte().transpose(0,2).transpose(0,1).numpy()
      
        
    im = Image.fromarray(ndarr)
    im.save(filename)




##################################################
# TRAINING CODE MAIN STARTING HERE
##################################################

print ("start:" , datetime.datetime.now().time())

conf_parser = argparse.ArgumentParser(
    description=__doc__, # printed with -h/--help
    # Don't mess with format of description
    formatter_class=argparse.RawDescriptionHelpFormatter,
    # Turn off help, so we print all options in response to -h
    add_help=False
    )
conf_parser.add_argument("-c", "--config",
                        help="Specify config file", metavar="FILE")

parser = argparse.ArgumentParser()

parser.add_argument('--data',  
    default = "/mnt/Data/visii_data/cutie/cutie_training", 
    help='path to training data')

parser.add_argument('--datatest', 
    default="/mnt/Data/visii_data/cutie/cutie_test", 
    help='path to data testing set')

parser.add_argument('--object', 
    default="cutie", 
    help='In the dataset which object of interest')

parser.add_argument('--workers', 
    type=int, 
    default=10,
    help='number of data loading workers')

parser.add_argument('--batchsize', 
    type=int, 
    default=128, 
    help='input batch size')

parser.add_argument('--subbatchsize', 
    type=int, 
    default=20, 
    help='input batch size')

parser.add_argument('--imagesize', 
    type=int, 
    default=400, 
    help='the height / width of the input image to network')

parser.add_argument('--lr', 
    type=float, 
    default=0.0001,
    help='learning rate, default=0.0001')

parser.add_argument('--noise', 
    type=float, 
    default=0.7, 
    help='gaussian noise added to the image')

parser.add_argument('--net', 
    default='', 
    help="path to net (to continue training)")

parser.add_argument('--namefile', 
    default='cutie', 
    help="name to put on the file of the save weights")

parser.add_argument('--manualseed', 
    type=int, 
    help='manual seed')

parser.add_argument('--epochs', 
    type=int, 
    default=120,
    help="number of epochs to train")

parser.add_argument('--loginterval', 
    type=int, 
    default=100)

parser.add_argument('--gpuids',
    nargs='+', 
    type=int, 
    default=[0], 
    help='GPUs to use')

parser.add_argument('--outf', 
    default='cutie', 
    help='folder to output images and model checkpoints, it will \
    add a train_ in front of the name')

parser.add_argument('--sigma', 
    default=4, 
    help='keypoint creation size for sigma')

parser.add_argument('--save', 
    action="store_true", 
    help='save a visual batch and quit, this is for\
    debugging purposes')

parser.add_argument("--pretrained",
    default=True,
    help='do you want to use vgg imagenet pretrained weights')

parser.add_argument('--nbupdates', 
    default=None, 
    help='nb max update to network, overwrites the epoch number\
    otherwise uses the number of epochs')

parser.add_argument('--datasize', 
    default=None, 
    help='randomly sample that number of entries in the dataset folder') 

# Read the config but do not overwrite the args written 
args, remaining_argv = conf_parser.parse_known_args()
defaults = { "option":"default" }

if args.config:
    config = configparser.SafeConfigParser()
    config.read([args.config])
    defaults.update(dict(config.items("defaults")))

parser.set_defaults(**defaults)
parser.add_argument("--option")
opt = parser.parse_args(remaining_argv)

if opt.pretrained in ['false', 'False']:
	opt.pretrained = False

if not "/" in opt.outf:
    opt.outf = "/mnt/Data/DOPE_trainings/train_{}".format(opt.outf)
    timestr = time.strftime("_%m_%d_%Y")
    opt.outf = opt.outf + timestr

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualseed is None:
    opt.manualseed = random.randint(1, 10000)

# save the hyper parameters passed
with open (opt.outf+'/header.txt','w') as file: 
    file.write(str(opt)+"\n")

with open (opt.outf+'/header.txt','w') as file: 
    file.write(str(opt))
    file.write("seed: "+ str(opt.manualseed)+'\n')
    with open (opt.outf+'/test_metric.csv','w') as file:
        file.write("epoch, passed,total \n")

# set the manual seed. 
random.seed(opt.manualseed)
torch.manual_seed(opt.manualseed)
torch.cuda.manual_seed_all(opt.manualseed)

additional_targets = {
    'centroids': 'keypoints'
}

def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')

def scale_down(x, **kwargs):
    return cv2.resize(x, (x.shape[0] // 8, x.shape[1] // 8))


img_size = (400,400)

mean = [0.45, 0.45, 0.45]
std = [0.25, 0.25, 0.25]
transform = A.Compose([
        A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.1, p=1, border_mode=0),

        A.RandomCrop(height=img_size[0], width=img_size[1]),

        A.IAAAdditiveGaussianNoise(p=0.2),
        A.IAAPerspective(p=0.5),

        A.OneOf(
            [
                A.CLAHE(p=1),
                A.RandomBrightness(p=1),
                A.RandomGamma(p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.IAASharpen(p=1),
                A.Blur(blur_limit=3, p=1),
                A.MotionBlur(blur_limit=3, p=1),
            ],
            p=0.9,
        ),

        A.OneOf(
            [
                A.RandomContrast(p=1),
                A.HueSaturationValue(p=1),
            ],
            p=0.9,
        )],
    additional_targets=additional_targets,
    keypoint_params=A.KeypointParams("xy", remove_invisible=False))

preprocessing_transform = A.Compose([
        A.Normalize(mean=mean, std=std),
        A.Lambda(mask=scale_down),
        A.Lambda(image=to_tensor, mask=to_tensor)],
    additional_targets=additional_targets,
    keypoint_params=A.KeypointParams("xy", remove_invisible=False))



#load the dataset using the loader in utils_pose
trainingdata = None
if not opt.data == "":
    train_dataset = MultipleVertexJson(
        root = opt.data,
        preprocessing_transform=preprocessing_transform,
        objectsofinterest=opt.object,
        sigma = opt.sigma,
        data_size = opt.datasize,
        save = opt.save,
        transform = transform
    )

    trainingdata = torch.utils.data.DataLoader(train_dataset,
        batch_size = opt.subbatchsize, 
        shuffle = True,
        num_workers = opt.workers, 
        pin_memory = True,
        drop_last=True
    )

    train_dataset.test = True
    for i in range(2):
        images = iter(trainingdata).next()

        save_image(images['image'],'{}/train_{}.png'.format( opt.outf,str(i).zfill(5)),mean=mean[0],std=std[0])
        print ("Saving batch %d" % i) 
    train_dataset.test = False

    if opt.save:
        print ('things are saved in {}'.format(opt.outf))
        quit()


testingdata = None
if not opt.datatest == "": 
    test_dataset = MultipleVertexJson(
            root = opt.datatest,
            preprocessing_transform=preprocessing_transform,
            objectsofinterest=opt.object,
            sigma = opt.sigma,
            data_size = opt.datasize,
            save = opt.save
            )

    testingdata = torch.utils.data.DataLoader(
        test_dataset,
        batch_size = opt.subbatchsize // 2, 
        shuffle = True,
        num_workers = opt.workers, 
        pin_memory = True,
        drop_last=True)


if not trainingdata is None:
    print('training data: {} batches'.format(len(trainingdata)))
if not testingdata is None:
    print ("testing data: {} batches".format(len(testingdata)))

net = DopeNetwork(pretrained=opt.pretrained).cuda()
# net = torch.nn.DataParallel(net,device_ids=opt.gpuids).cuda()

if opt.net != '':
    net.load_state_dict(torch.load(opt.net))

parameters = filter(lambda p: p.requires_grad, net.parameters())
optimizer = optim.Adam(parameters,lr=opt.lr)

with open (opt.outf+'/loss_train.csv','w') as file: 
    file.write('epoch,batchid,loss\n')

with open (opt.outf+'/loss_test.csv','w') as file: 
    file.write('epoch,batchid,loss\n')

nb_update_network = 0

def _runnetwork(epoch, loader, train=True, scaler=None, pbar=None):
    global nb_update_network
    # net
    if train:
        net.train()
    else:
        net.eval()

    if train:
        optimizer.zero_grad()
    for batch_idx, targets in enumerate(loader):

        data = Variable(targets['image'].cuda())
        
        with amp.autocast():
            output_belief, output_affinities = net(data)

            target_belief = Variable(targets['beliefs'].cuda())        
            target_affinity = Variable(targets['affinities'].cuda())

            loss = None
            
            # Belief maps loss
            for l in output_belief: #output, each belief map layers. 
                if loss is None:
                    loss = ((l - target_belief) * (l-target_belief)).mean()
                else:
                    loss_tmp = ((l - target_belief) * (l-target_belief)).mean()
                    loss += loss_tmp
            
            # Affinities loss
            for l in output_affinities: #output, each belief map layers. 
                loss_tmp = ((l - target_affinity) * (l-target_affinity)).mean()
                loss += loss_tmp 

        if train:
            scaler.scale(loss).backward()
            if batch_idx % (opt.batchsize // opt.subbatchsize) == 0:
                if train:
                    scaler.step(optimizer)
                    scaler.update()
                    nb_update_network+=1
                    optimizer.zero_grad()

        if train:
            namefile = '/loss_train.csv'
        else:
            namefile = '/loss_test.csv'

        with open (opt.outf+namefile,'a') as file:
            s = '{}, {},{:.15f}\n'.format(
                epoch,batch_idx,loss.data.item()) 
            # print (s)
            file.write(s)

        # break
        if not opt.nbupdates is None and nb_update_network > int(opt.nbupdates):
            torch.save(net.state_dict(), '{}/net_{}.pth'.format(opt.outf, opt.namefile))
            break

        if train:
            if pbar is not None:
                pbar.set_description("Training loss: %0.4f (%d/%d)" % (loss.data.item(), batch_idx, len(loader)))
        else:
            if pbar is not None:
                pbar.set_description("Testing loss: %0.4f (%d/%d)" % (loss.data.item(), batch_idx, len(loader)))
    if train:
        optimizer.zero_grad()

scaler = amp.GradScaler()
torch.backends.cudnn.benchmark = True
pbar = tqdm(range(1, opt.epochs + 1))

for epoch in pbar:

    if not trainingdata is None:
        _runnetwork(epoch,trainingdata, scaler=scaler, pbar=pbar)

    if not opt.datatest == "":
        _runnetwork(epoch,testingdata, train=False, pbar=pbar)
        if opt.data == "":
            break # lets get out of this if we are only testing
    try:
        torch.save(net.state_dict(), '{}/net_{}_{}.pth'.format(opt.outf, opt.namefile ,epoch))
    except:
        pass

    if not opt.nbupdates is None and nb_update_network > int(opt.nbupdates):
        break

print ("end:" , datetime.datetime.now().time())
