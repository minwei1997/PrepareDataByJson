import pickle
import os
import os.path as osp
import numpy as np 
import cv2
from utils.data_extract import Data_extractor
from utils.DataAug_Rot_funciton import Rot_img_bbox


class DataAugmentation():
    def __init__(self, img, roidb):
        super().__init__()
        # define all cls type
        self.classes = ('__background__',  # always index 0
                    'hole')

        self.Data_extractor = Data_extractor(self.classes)
        self.img = img
        self.bbox =  roidb['boxes']
        self.cls = roidb['gt_classes']
        self.num_objs = self.bbox.shape[0]

    def horizontal_flip(self):
        ''' flipping img '''
        img_ctr = np.array(self.img.shape[:2])[::-1]/2        # center coord (ctr_w, ctr_h)
        img =  self.img[:,::-1,:]     # horizontal flip (flip w's dim)

        New_boxes = np.zeros((self.num_objs, 4), dtype=np.uint16)
        for ix in range(self.num_objs):
            ''' about generate new bbox '''
            box_ctr_w = round(np.sum(self.bbox[ix, [0,2]]) / 2)
            box_ctr_h = round(np.sum(self.bbox[ix, [1,3]]) / 2)
            box_w = round(self.bbox[ix, 2] - self.bbox[ix, 0])
            box_h = round(self.bbox[ix, 3] - self.bbox[ix, 1])

            new_x1, new_y1 = round(2*img_ctr[0] - box_ctr_w - (box_w/2)), self.bbox[ix, 1].astype(np.int)
            new_x2, new_y2 = round(2*img_ctr[0] - box_ctr_w + (box_w/2)), self.bbox[ix, 3].astype(np.int)
            New_boxes[ix, :] = np.array([new_x1, new_y1, new_x2, new_y2])
            

        ''' about generate new dataset '''
        New_roidb = self.Data_extractor.roidb_generate(num_objs=self.num_objs, aug_mode=True, clss=self.cls, bbox=New_boxes)
        
        return img, New_roidb

    def vertical_flip(self):
        ''' flipping img '''
        img_ctr = np.array(self.img.shape[:2])[::-1]/2        # center coord (ctr_w, ctr_h)
        img =  self.img[::-1,:,:]     # vertical flip (flip h's dim)

        New_boxes = np.zeros((self.num_objs, 4), dtype=np.uint16)
        for ix in range(self.num_objs):
            ''' about generate new bbox '''
            box_ctr_w = round(np.sum(self.bbox[ix, [0,2]]) / 2)
            box_ctr_h = round(np.sum(self.bbox[ix, [1,3]]) / 2)
            box_w = round(self.bbox[ix, 2] - self.bbox[ix, 0])
            box_h = round(self.bbox[ix, 3] - self.bbox[ix, 1])

            new_x1, new_y1 = self.bbox[ix, 0].astype(np.int), round(2*img_ctr[1] - box_ctr_h - (box_h/2))
            new_x2, new_y2 = self.bbox[ix, 2].astype(np.int), round(2*img_ctr[1] - box_ctr_h + (box_h/2))
            New_boxes[ix, :] = np.array([new_x1, new_y1, new_x2, new_y2])
            

        ''' about generate new dataset '''
        New_roidb = self.Data_extractor.roidb_generate(num_objs=self.num_objs, aug_mode=True, clss=self.cls, bbox=New_boxes)
        
        return img, New_roidb

    def RotateByTheta(self, theta):
        ''' theta is define positive for counter clockwise 
        '''
        img, New_boxes = Rot_img_bbox(theta, self.img, self.bbox)
        New_roidb = self.Data_extractor.roidb_generate(num_objs=self.num_objs, aug_mode=True, clss=self.cls, bbox=New_boxes)

        return img, New_roidb
        

    def rescale_img(self):
        ''' use for visualize (because the original img is too big)
        '''
        pass

if __name__ == '__main__':
    path = '.\data\\training_pickle'
    file = 'training_data_pkl.pkl'
    with open(osp.join(path, file), 'rb') as f:
        data = pickle.load(f)
        
    roidb = data[35]    #35
    img = cv2.imread('.\data\Defect_Img\HoleImage36.png')    # (h, w, channel)  #36

    DataAug = DataAugmentation(img, roidb)
    mode = 'Vertical_flip'
    if mode == 'Horizontal_flip':
        img, New_roidb = DataAug.horizontal_flip()
    elif mode == 'Vertical_flip':
        img, New_roidb = DataAug.vertical_flip()
    elif mode == 'rotate': 
        img, New_roidb = DataAug.RotateByTheta(30)
    else:
        ValueError('mode = {} (typo?)'.format(mode))


    target_size = 800
    im_scale = target_size / img.shape[1]
    
    
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)
    # # for ix in range(len(New_roidb)):
    bbox = New_roidb['boxes'] * im_scale
    for i in range(bbox.shape[0]):
        x1, y1, x2, y2 = bbox[i, :].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        x1, y1, x2, y2 = bbox[i, :].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


