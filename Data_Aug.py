import pickle
import os
import os.path as osp
import numpy as np 
import sys
import cv2
import xml.etree.ElementTree as ET

from utils.data_extract import Data_extractor
from utils.DataAug_Rot_funciton import Rot_img_bbox
from utils.StringSortByDigit import natural_keys

""" 此code之擴增資料為 水平翻轉 + 垂直翻轉 後再針對所有圖片進行Rotate"""

''' generate augmented xml '''
class DataAugmentation():
    def __init__(self, ImgPath, OldImgName, XmlPath, OldXmlName):
        super().__init__()

        self.ImgPath = ImgPath
        self.XmlPath = XmlPath
        self.img = cv2.imread(osp.join(self.ImgPath, OldImgName))
        self.num_objs, self.bbox, self.cls = self.get_xml_info(OldXmlName)
        self._Data_extractor = Data_extractor(self.XmlPath, self.ImgPath)

    def get_xml_info(self, xml_name):
        config = {'use_diff': False}

        filename = os.path.join(self.XmlPath, xml_name)
        tree = ET.parse(filename)
        objs = tree.findall('object')       # find all objs
        if not config['use_diff']:
            # Exclude the samples labeled as difficult (extract all non difficult obj in xml)
            non_diff_objs = [
                obj for obj in objs if int(obj.find('difficult').text) == 0]    # find 'difficult' text with specific obj in xml 

            objs = non_diff_objs

        num_objs = len(objs)

        boxes = []
        cls = []
        for ix, obj in enumerate(objs):
            # Load object bounding boxes 
            bndbox = obj.find('bndbox')   
            old_x1 = float(bndbox.find('xmin').text)
            old_y1 = float(bndbox.find('ymin').text)
            old_x2 = float(bndbox.find('xmax').text)
            old_y2 = float(bndbox.find('ymax').text)
            box_coord = np.asarray([[old_x1, old_y1, old_x2, old_y2]])
            boxes = box_coord if len(boxes) == 0 else np.vstack((boxes, box_coord))

            # Load object label 
            clss = obj.find('name').text.lower().strip()
            cls.append(clss)

        return num_objs, boxes, cls

    def horizontal_flip(self, NewImgName):
        ''' flipping img '''
        [h, w] = np.array(self.img.shape[:2]) 
        img =  self.img[:,::-1,:]     # horizontal flip (flip w's dim)

        New_boxes = np.zeros((self.num_objs, 4), dtype=np.uint16)
        for ix in range(self.num_objs):
            ''' about generate new bbox '''
            box_ctr_w = round(np.sum(self.bbox[ix, [0,2]]) / 2)
            box_ctr_h = round(np.sum(self.bbox[ix, [1,3]]) / 2)
            box_w = round(self.bbox[ix, 2] - self.bbox[ix, 0])
            box_h = round(self.bbox[ix, 3] - self.bbox[ix, 1])

            new_x1, new_y1 = round(w - box_ctr_w - (box_w/2)), self.bbox[ix, 1].astype(np.int)
            new_x2, new_y2 = round(w - box_ctr_w + (box_w/2)), self.bbox[ix, 3].astype(np.int)
            New_boxes[ix, :] = np.array([new_x1, new_y1, new_x2, new_y2])
            
        ''' about generate new xml '''
        self._Data_extractor.xml_generate(num_objs=self.num_objs, AugMode=True, img=self.img, clss=self.cls, bbox=New_boxes, NewImgName=NewImgName)
        
        ''' about saving augmented image '''
        cv2.imwrite(osp.join(self.ImgPath, NewImgName), img)

    def vertical_flip(self, NewImgName):
        ''' flipping img '''
        [h, w] = np.array(self.img.shape[:2])
        img =  self.img[::-1,:,:]     # vertical flip (flip h's dim)

        New_boxes = np.zeros((self.num_objs, 4), dtype=np.uint16)
        for ix in range(self.num_objs):
            ''' about generate new bbox '''
            box_ctr_w = round(np.sum(self.bbox[ix, [0,2]]) / 2)
            box_ctr_h = round(np.sum(self.bbox[ix, [1,3]]) / 2)
            box_w = round(self.bbox[ix, 2] - self.bbox[ix, 0])
            box_h = round(self.bbox[ix, 3] - self.bbox[ix, 1])

            new_x1, new_y1 = self.bbox[ix, 0].astype(np.int), round(h - box_ctr_h - (box_h/2))
            new_x2, new_y2 = self.bbox[ix, 2].astype(np.int), round(h - box_ctr_h + (box_h/2))
            New_boxes[ix, :] = np.array([new_x1, new_y1, new_x2, new_y2])
            

        ''' about generate new xml '''
        self._Data_extractor.xml_generate(num_objs=self.num_objs, AugMode=True, img=self.img, clss=self.cls, bbox=New_boxes, NewImgName=NewImgName)
        
        ''' about saving augmented image '''
        cv2.imwrite(osp.join(self.ImgPath, NewImgName), img)

    def RotateByTheta(self, theta):
        ''' theta is define positive for counter clockwise '''
        fail = False

        img, New_boxes, mask = Rot_img_bbox(theta, self.img, self.bbox)
        num_objs = np.sum(mask)     # calc. the boxes number that need to reserve
        cls = [i for idx,i in enumerate(self.cls) if mask[idx] == True]
        New_boxes = New_boxes[mask]
        if num_objs == 0:
            fail = True
            return None, None, fail

        ''' about generate new xml '''
        self._Data_extractor.xml_generate(num_objs=self.num_objs, AugMode=True, img=self.img, clss=self.cls, bbox=New_boxes, NewImgName=NewImgName)

        ''' about saving augmented image '''
        cv2.imwrite(osp.join(self.ImgPath, NewImgName), img)

        return fail
        

def Resize_and_Vis_Img(img, im_scale, xml_file):
    ''' resize and visualize (because the original img is too big)
    '''
    tree = ET.parse(xml_file)
    objs = tree.findall('object')

    bbox = []
    for ix, obj in enumerate(objs):
        bndbox = obj.find('bndbox')   
        x1 = float(bndbox.find('xmin').text)
        y1 = float(bndbox.find('ymin').text)
        x2 = float(bndbox.find('xmax').text)
        y2 = float(bndbox.find('ymax').text)
        box_coord = np.asarray([[x1, y1, x2, y2]])
        bbox = box_coord if len(bbox) == 0 else np.vstack((bbox, box_coord))

    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)

    bbox = bbox * im_scale
    for i in range(bbox.shape[0]):
        x1, y1, x2, y2 = bbox[i, :].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    def digit_count(idx):
        count = 0
        n = int(idx)
        while(n > 0):
            count += 1
            n = n // 10
        return count

    vis_mode = int(input('Visualize Mode. (0)False, (1)True :'))     
    if vis_mode > 1:
        raise ValueError('Mode Error. Retry again')
    else:
        vis_mode = bool(vis_mode)

    if vis_mode:
        # get all img
        ImgDir = r'.\data\Defect_Img'
        ImgExt = '.jpg'
        AllImg = [files for files in os.listdir(ImgDir) if files.endswith(ImgExt)]    
        AllImg.sort(key=natural_keys)    # Sort by file's index
        
        # get all xml file
        XmlDir = r'.\data\xml_file'
        XmlExt = '.xml'
        all_xml = [files for files in os.listdir(XmlDir) if files.endswith(XmlExt)]    
        all_xml.sort(key=natural_keys)    

        for k in range(len(AllImg)):
            print('img :' ,AllImg[k])
            img = cv2.imread(osp.join(ImgDir, AllImg[k]))

            target_size = 800
            im_scale = target_size / img.shape[1]
            _xml_path = osp.join(XmlDir, all_xml[k])
            Resize_and_Vis_Img(img, im_scale, _xml_path)
    else:
        original_num = int(input('number of original image : '))

        # get all image
        ImgDir = r'.\data\Defect_Img'
        ImgExt = '.jpg'
        AllImg = [files for files in os.listdir(ImgDir) if files.endswith(ImgExt)]      # xxx.jpg  
        AllImg.sort(key=natural_keys)    # Sort by file's index

        img_idx = str(int(AllImg[-1].strip('.jpg')) + 1)      # img idx
        if int(img_idx) > original_num:
            img_idx = str(original_num + 1)
            AllImg = AllImg[:int(img_idx)-1]
        print('image index : ', img_idx)

        XmlPath = r'.\data\xml_file'
        _name = '000000'

        ''' Data Augmentation (Do horizontal and vertical flip) '''
        print('horizontal and vertical flip processing...')
        for i in range(len(AllImg)):
            OldImgName = AllImg[i]
            OldXmlName = AllImg[i].strip('.jpg') + '.xml'    
            DataAug = DataAugmentation(ImgDir, OldImgName, XmlPath, OldXmlName)

            # Augment process
            ''' horizontal flip '''
            d_count = digit_count(int(img_idx))
            NewName = _name[:-d_count] + img_idx
            NewImgName = NewName + '.jpg'
            NewXmlName = NewName
            DataAug.horizontal_flip(NewImgName)
            img_idx = str(int(img_idx) + 1)

            ''' vertical flip '''
            d_count = digit_count(int(img_idx))
            NewName = _name[:-d_count] + img_idx
            NewImgName = NewName + '.jpg'
            NewXmlName = NewName
            DataAug.vertical_flip(NewImgName)
            img_idx = str(int(img_idx) + 1)
            
            sys.stdout.write('Fliping : {}/{} \r' \
                           .format(i+1, len(AllImg)))

        print('Horizontal and Vertical flip are done ! ')


        ''' Data Augmentation (Rotate)  '''
        # get all img
        AllImg = [files for files in os.listdir(ImgDir) if files.endswith(ImgExt)]    
        AllImg.sort(key=natural_keys)    # Sort by file's index
        img_idx = str(int(AllImg[-1].strip('.jpg')) + 1)      # img idx

        print('Rotate processing...')
        for i in range(len(AllImg)):
            ''' rotation (from -30 to 30 deg. with each step=10) '''
            OldImgName = AllImg[i]
            OldXmlName = AllImg[i].strip('.jpg') + '.xml'    
            DataAug = DataAugmentation(ImgDir, OldImgName, XmlPath, OldXmlName)

            # for j in range(-30, 31, 5):
            for j in range(-5, 6, 5):
                if j == 0:
                    continue
                else:
                    d_count = digit_count(int(img_idx))
                    NewName = _name[:-d_count] + img_idx
                    NewImgName = NewName + '.jpg'
                    NewXmlName = NewName
                    fail = DataAug.RotateByTheta(j)
                    if fail == True:
                        raise ValueError('number of obj is 0 after augment process !') 
                    img_idx = str(int(img_idx) + 1)

            sys.stdout.write('Rotating : {}/{} \r' \
                .format(i+1, len(AllImg)))

        print('\nProcess Done.\n')



    
    



