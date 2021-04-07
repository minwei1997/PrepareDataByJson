import cv2
import numpy as np 
import os
import os.path as osp
import sys

def CalcMean(ImgDir, imgs):
    _img = cv2.imread(osp.join(ImgDir, imgs[0]))
    pixel_val = np.zeros(_img.shape)
    for i, f in enumerate(imgs):
        img = cv2.imread(osp.join(ImgDir, f))
        pixel_val += np.array(img)
        sys.stdout.write('processed image : {}/{} \r' \
                           .format(i+1, len(imgs)))
    pixel_val /= len(imgs)
    pixel_mean = np.mean(pixel_val, axis=(0,1))
    return pixel_mean

if __name__ == '__main__':
    ImgDir = r'.\data\Defect_Img'
    ImgExt = '.jpg'
    AllImgFile = [files for files in os.listdir(ImgDir) if files.endswith(ImgExt)]       # xxx.jpg

    pixel_mean = CalcMean(ImgDir, AllImgFile)
    print('pixel_mean : ', pixel_mean)