import pickle
import os
import os.path as osp
import numpy as np 
import cv2
from utils.data_extract import Data_extractor
from utils.DataAug_Rot_funciton import Rot_img_bbox
from utils.StringSortByDigit import natural_keys

''' generate augmented roidb '''
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
        img, New_boxes, mask = Rot_img_bbox(theta, self.img, self.bbox)
        num_objs = np.sum(mask)     # calc. the boxes number that need to reserve
        cls = self.cls[mask]
        New_boxes = New_boxes[mask]
        if num_objs == 0:
            fail = True
            return None, None, fail

        New_roidb = self.Data_extractor.roidb_generate(num_objs=num_objs, aug_mode=True, clss=cls, bbox=New_boxes)
        fail = False
        return img, New_roidb, fail
        

    def Resize_and_Vis_Img(self, img, im_scale, roidb):
        ''' resize and visualize (because the original img is too big)
        '''
        img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)
        bbox = roidb['boxes'] * im_scale
        for i in range(bbox.shape[0]):
            x1, y1, x2, y2 = bbox[i, :].astype(int)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        cv2.imshow('window', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    vis_mode = True     # True for visualize result. False for Data Augmentation
    original_num = 53
    
    if vis_mode:
        # load roidb 
        path = '.\data\\training_pickle'
        file = 'Training_Data_Agmented.pkl'
        with open(osp.join(path, file), 'rb') as f:
            roidb = pickle.load(f)

        # get all img
        img_fileDir = './data/Defect_Img/'
        img_fileExt = '.png'
        all_img = [files for files in os.listdir(img_fileDir) if files.endswith(img_fileExt)]    
        all_img.sort(key=natural_keys)    # Sort by file's index
        
        DataAug = DataAugmentation(None, roidb[0])
        for k in range(len(all_img)):
            print('img :' ,all_img[k])
            img = cv2.imread('.\data\Defect_Img\\' + all_img[k])

            target_size = 800
            im_scale = target_size / img.shape[1]
            DataAug.Resize_and_Vis_Img(img, im_scale, roidb[k])
    else:
        # load roidb 
        path = '.\data\\training_pickle'
        file = 'training_data_pkl.pkl'
        with open(osp.join(path, file), 'rb') as f:
            roidb = pickle.load(f)

        # get all img
        img_fileDir = './data/Defect_Img/'
        img_fileExt = '.png'
        all_img = [files for files in os.listdir(img_fileDir) if files.endswith(img_fileExt)]    
        all_img.sort(key=natural_keys)    # Sort by file's index

        img_name = natural_keys(all_img[-1])[0]         # HoleImage
        img_idx = str(natural_keys(all_img[-1])[1] + 1)      # idx
        if int(img_idx) > original_num:
            img_idx = str(original_num + 1)
        print(img_idx)
        # Data Augmentation
        for i in range(len(roidb)):
            img = cv2.imread('.\data\Defect_Img\\' + all_img[i])
            _roidb = roidb[i]
            DataAug = DataAugmentation(img, _roidb)

            # Augment process
            ''' horizontal flip '''
            img1, New_roidb_1 = DataAug.horizontal_flip()
            roidb.append(New_roidb_1)
            cv2.imwrite((img_fileDir + img_name + img_idx + img_fileExt), img1)
            img_idx = str(int(img_idx) + 1)
            ''' vertical flip '''
            img2, New_roidb_2 = DataAug.vertical_flip()
            roidb.append(New_roidb_2)
            cv2.imwrite((img_fileDir + img_name + img_idx + img_fileExt), img2)
            img_idx = str(int(img_idx) + 1)
            ''' rotation (from -30 to 30 deg. with each step=10) '''
            for j in range(-30, 31, 10):
                if j == 0:
                    continue
                else:
                    img3, New_roidb_3, fail = DataAug.RotateByTheta(j)
                    if fail == True:
                        raise ValueError('number of obj is 0 after augment process !') 
                    roidb.append(New_roidb_3)
                    cv2.imwrite((img_fileDir + img_name + img_idx + img_fileExt), img3)
                    img_idx = str(int(img_idx) + 1)

        # save pkl file
        pkl_path = './data/training_pickle/Training_Data_Agmented.pkl'
        with open(pkl_path, 'wb') as fid:
            pickle.dump(roidb, fid, pickle.HIGHEST_PROTOCOL)
            print('Wrote gt roidb to {}'.format(pkl_path))

        print('\nProcess Done.\n')



    
    



