import os 
import cv2 
import math
import h5py
import argparse
import numpy as np
from PIL import Image 
import scipy.io as sco

from util import processDepthImage

def saveColor(data, out_dir):
    colors = np.array(data['images'])
    for i in range(len(colors)): 
        color = colors[i]
        color = np.stack((color[2].T, color[1].T, color[0].T), axis=-1)
        cv2.imwrite(os.path.join(out_dir, str(i)+'.png'), color)

def saveDepth(data, out_dir):
    depths = np.array(data['depths'])
    for i in range(len(depths)): 
        depth = np.uint16(depths[i].T * 1000)
        cv2.imwrite(os.path.join(out_dir, str(i)+'.png'), depth)

def saveLabel(labels, out_dir):
    for i in range(labels.shape[-1]): 
        label = labels[:, :, i]
        cv2.imwrite(os.path.join(out_dir, str(i)+'.png'), label)

def camera_matrix(): 
    fx_rgb = 5.1885790117450188e+02
    fy_rgb = 5.1946961112127485e+02
    cx_rgb = 3.2558244941119034e+02
    cy_rgb = 2.5373616633400465e+02
    C = np.array([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])
    return C 

# This function is borrowed from https://github.com/charlesCXK/Depth2HHA-python
def saveHHA(data, out_dir): 
    C = camera_matrix()
    depths = np.array(data['depths'])
    for i in range(1): 
        D = RD = depths[i].T
        missingMask = (RD == 0)
        pc, N, yDir, h, pcRot, NRot = processDepthImage(D * 100, missingMask, C)
    
        tmp = np.multiply(N, yDir)
        acosValue = np.minimum(1,np.maximum(-1,np.sum(tmp, axis=2)))
        angle = np.array([math.degrees(math.acos(x)) for x in acosValue.flatten()])
        angle = np.reshape(angle, h.shape)
    
        '''
        Must convert nan to 180 as the MATLAB program actually does. 
        Or we will get a HHA image whose border region is different
        with that of MATLAB program's output.
        '''
        angle[np.isnan(angle)] = 180        
    
    
        pc[:,:,2] = np.maximum(pc[:,:,2], 100)
        I = np.zeros(pc.shape)
    
        # opencv-python save the picture in BGR order.
        I[:,:,2] = 31000/pc[:,:,2]
        I[:,:,1] = h
        I[:,:,0] = (angle + 128-90)
    
        # print(np.isnan(angle))
    
        '''
        np.uint8 seems to use 'floor', but in matlab, it seems to use 'round'.
        So I convert it to integer myself.
        '''
        I = np.rint(I)
    
        # np.uint8: 256->1, but in MATLAB, uint8: 256->255
        I[I>255] = 255
        HHA = I.astype(np.uint8)
        cv2.imwrite(os.path.join(out_dir, 'HHA', str(i)+'.png'), HHA)

def parse_args(): 
    parser = argparse.ArgumentParser(description='DetSeg extract nyuv2 dataset')
    parser.add_argument('matfile', help='nyuv2_depth_v2_labeled.mat')
    # download labels40.mat from https://github.com/ankurhanda/nyuv2-meta-data
    parser.add_argument('labelfile', help='labels40.mat')
    parser.add_argument('--out_dir', default='data/nyuv2', help='data dir')
    args = parser.parse_args()
    return args 

if __name__ == '__main__': 
    args = parse_args()
    data = h5py.File(args.matfile)
    labels = sco.loadmat(args.labelfile)['labels40']
    if not os.path.exists(args.out_dir): 
        for t in ['image', 'depth', 'label', 'HHA']: 
            os.makedirs(os.path.join(args.out_dir, t))
    
    saveColor(data, os.path.join(args.out_dir, 'image'))
    saveDepth(data, os.path.join(args.out_dir, 'depth'))
    saveLabel(labels, os.path.join(args.out_dir, 'label'))
    saveHHA(data, args.out_dir)
    
