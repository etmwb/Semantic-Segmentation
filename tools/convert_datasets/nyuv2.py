import os 
import cv2 
import math
import h5py
import argparse
import numpy as np
from PIL import Image 

from util import processDepthImage

def saveColor(data, out_dir):
    colors = np.array(data['images'])
    for i in range(len(colors)): 
        color = colors[i]
        r, g, b = Image.fromarray(color[0]).convert('L'), Image.fromarray(color[1]).convert('L'), Image.fromarray(color[2]).convert('L')
        color = Image.merge("RGB", (r, g, b)).transpose(Image.ROTATE_270)
        color.save(os.path.join(out_dir, str(i)+'.png'))

def saveDepth(data, out_dir):
    depths = np.array(data['depths'])
    for i in range(len(depths)): 
        depth = np.uint16(depths[i].T * 1000)
        cv2.imwrite(os.path.join(out_dir, str(i)+'.png'), depth)

def saveLabel(data, out_dir):
    labels = np.array(data['labels'])
    for i in range(len(labels)): 
        label = labels[i]
        label = Image.fromarray(np.uint8(label)).transpose(Image.ROTATE_270)
        label.save(os.path.join(out_dir, str(i)+'.png'))

def camera_matrix(): 
    fx_rgb = 5.1885790117450188e+02
    fy_rgb = 5.1946961112127485e+02
    cx_rgb = 3.2558244941119034e+02
    cy_rgb = 2.5373616633400465e+02
    C = np.array([[fx_rgb, 0, cx_rgb], [0, fy_rgb, cy_rgb], [0, 0, 1]])
    return C 

# This function is borrowed from https://github.com/charlesCXK/Depth2HHA-python
def saveHHA(files, out_dir): 
    C = camera_matrix()
    for i in range(len(files)): 
        D = RD = cv2.imread(os.path.join(out_dir, 'depth', files[i]), cv2.COLOR_BGR2GRAY)/1000
        missingMask = (RD == 0);
        pc, N, yDir, h, pcRot, NRot = processDepthImage(D * 100, missingMask, C);
    
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
    parser.add_argument('--out_dir', default='data/nyuv2', help='data dir')
    args = parser.parse_args()
    return args 

if __name__ == '__main__': 
    args = parse_args()
    data = h5py.File(args.matfile)
    if not os.path.exists(args.out_dir): 
        for t in ['image', 'depth', 'label', 'HHA']: 
            os.makedirs(os.path.join(args.out_dir, t))
    
    saveColor(data, os.path.join(args.out_dir, 'image'))
    saveDepth(data, os.path.join(args.out_dir, 'depth'))
    saveLabel(data, os.path.join(args.out_dir, 'label'))
    saveHHA(os.listdir(os.path.join(args.out_dir, 'image')), args.out_dir)
    
