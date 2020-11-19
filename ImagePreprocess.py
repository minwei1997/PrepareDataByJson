import cv2, os 
import numpy as np
import os.path as osp

from utils.StringSortByDigit import natural_keys
from argparse import ArgumentParser

class ImgPreprocessor():
    def __init__(self, ext='.jpg'):
        super().__init__()
        self.ImgPath = r'.\JPEGImages'
        self.AllImg = [files for files in os.listdir(self.ImgPath) if files.endswith(ext)]    # Get all image endwith .png
        self.AllImg.sort(key=natural_keys)    # Sort by file's index

        # check if path exist if not then create 
        self.SavePath = r'.\temp'
        if not osp.isdir(self.SavePath):
            os.mkdir(self.SavePath)
        
    def DoContrastProcess(self, alpha, beta):
        '''Given alpha(contrast coe.) and beta(brigtness coe.)

            then convert img by g(x) = αf(x) + β '''
        print('DoContrastProcess...')
        for idx, im in enumerate(self.AllImg):
            img = cv2.imread(osp.join(self.ImgPath, im))
            img = np.uint8(np.clip((alpha * img + beta), 0, 255))   # Contrast and Brightness process
            cv2.imwrite(osp.join(self.SavePath, im), img)
            print('processed image save to', osp.join(self.SavePath, im))

    def DoColor2Gray(self):
        ''' convert color image to gray scale(3 channels) image'''

        print('DoColor2Gray...')
        for idx, im in enumerate(self.AllImg):
            img = cv2.imread(osp.join(self.ImgPath, im), 0)
            img = np.tile(img[:, :, np.newaxis], (1, 1, 3))
            cv2.imwrite(osp.join(self.SavePath, im), img)
            print('processed image save to : ', osp.join(self.SavePath, im))

if __name__ == '__main__':
    # args
    parser = ArgumentParser()
    parser.add_argument("-a", "--alpha", dest="alpha", type=float, default=2.0)   # alpha (contrast, 0 ~ 3)
    parser.add_argument("-b", "--beta", dest="beta", type=int, default=20)     # beta (brigtness, 0 ~ 255)
    parser.add_argument("-m", "--mode", dest="mode", type=str, default='contrast')     # mode can be contrast or gray
    args = parser.parse_args()
    # get parameter from args
    alpha = args.alpha if (args.alpha>=0 and args.alpha<=3) else -1
    beta = args.beta if (args.alpha>=0 and args.alpha<=255) else -1
    mode = args.mode if (args.mode=='contrast' or args.mode=='gray') else -1
    if alpha == -1 or beta == -1 or mode == -1:
        raise ValueError('some args are wrong!')

    # Do image preprocess
    Processor = ImgPreprocessor()
    if mode == 'contrast':
        Processor.DoContrastProcess(alpha, beta)
    elif mode == 'gray':
        Processor.DoColor2Gray()
