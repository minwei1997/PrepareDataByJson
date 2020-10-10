from scipy import sparse
import numpy as np 
import codecs
import cv2
import os.path as osp
import json 

class Data_extractor():
    ''' use to generate xml file '''

    # self.xml_path, img_path, old_img_name
    def __init__(self, xml_save_path, img_path, img_name):
        super().__init__()
        self.xml_save_path = xml_save_path
        self.img_path = img_path
        self.img_name = img_name
        

    def xml_generate(self, num_objs, json_file=None, aug_mode=False, clss=None, bbox=None, new_img_name=None):
        ''' generate xml file '''

        img = cv2.imread(self.img_path + '\\' + self.img_name)
        h, w, depth = img.shape

        height = h
        width = w
        depth = depth 

        _img_name = self.img_name if not aug_mode else new_img_name
        with codecs.open(self.xml_save_path + '\\' + _img_name.split('.')[0] + '.xml', 'w', 'utf-8') as xml:
            xml.write('<annotation>\n')
            xml.write('\t<folder>' + 'VOC2007' + '</folder>\n')
            xml.write('\t<filename>' + _img_name + '</filename>\n')
            xml.write('\t<source>\n')
            xml.write('\t\t<database>The Defect Img Database</database>\n')
            xml.write('\t\t<annotation>Defect Img</annotation>\n')
            xml.write('\t\t<image>NULL</image>\n')
            xml.write('\t\t<flickrid>NULL</flickrid>\n')
            xml.write('\t</source>\n')
            xml.write('\t<owner>\n')
            xml.write('\t\t<flickrid>NULL</flickrid>\n')
            xml.write('\t\t<name>MinWei</name>\n')
            xml.write('\t</owner>\n')
            xml.write('\t<size>\n')
            xml.write('\t\t<width>'+ str(width) + '</width>\n')
            xml.write('\t\t<height>'+ str(height) + '</height>\n')
            xml.write('\t\t<depth>' + str(depth) + '</depth>\n')
            xml.write('\t</size>\n')
            xml.write('\t<segmented>0</segmented>\n')
            for j in range(num_objs):
                # get data information
                x1 = int((json_file['shapes'][j]['points'])[0][0]) if not aug_mode else int(bbox[j, 0])
                y1 = int((json_file['shapes'][j]['points'])[0][1]) if not aug_mode else int(bbox[j, 1])
                x2 = int((json_file['shapes'][j]['points'])[1][0]) if not aug_mode else int(bbox[j, 2])
                y2 = int((json_file['shapes'][j]['points'])[1][1]) if not aug_mode else int(bbox[j, 3])
                _cls = json_file['shapes'][j]['label'] if not aug_mode else clss[j]

                xml.write('\t<object>\n')
                xml.write('\t\t<name>' + str(_cls) + '</name>\n')     # cls
                xml.write('\t\t<pose>Unspecified</pose>\n')
                xml.write('\t\t<truncated>0</truncated>\n')
                xml.write('\t\t<difficult>0</difficult>\n')
                xml.write('\t\t<bndbox>\n')
                xml.write('\t\t\t<xmin>' + str(x1) + '</xmin>\n')    # x1
                xml.write('\t\t\t<ymin>' + str(y1) + '</ymin>\n')    # y1
                xml.write('\t\t\t<xmax>' + str(x2) + '</xmax>\n')    # x2
                xml.write('\t\t\t<ymax>' + str(y2) + '</ymax>\n')    # y2
                xml.write('\t\t</bndbox>\n')
                xml.write('\t</object>\n')
            xml.write('</annotation>')

        # '''---------------------'''
        # for ix in range(num_objs):
        #     if not aug_mode:    # get data from json
        #         cls = json_file['shapes'][ix]['label']              # get the cls label 
        #         bbox = json_file['shapes'][ix]['points']      # get the bbox coord
        #         # Make pixel indexes 0-based (get data form list obj)
        #         x1 = bbox[0][0] - 1
        #         y1 = bbox[0][1] - 1
        #         x2 = bbox[1][0] - 1
        #         y2 = bbox[1][1] - 1
        #         cls = self._class_to_ind[cls.lower().strip()]    # conver cls to corresponding index   
        #     else:               # get data from original img when Data Aug. mode
        #         cls = clss[ix]
        #         # get data from numpy array
        #         x1 = bbox[ix, 0] - 1
        #         y1 = bbox[ix, 1] - 1
        #         x2 = bbox[ix, 2] - 1
        #         y2 = bbox[ix, 3] - 1
            
        #     boxes[ix, :] = [x1, y1, x2, y2]
        #     gt_classes[ix] = cls
        #     overlaps[ix, cls] = 1.0
        #     seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        # overlaps = sparse.csr_matrix(overlaps)
        # return {'boxes': boxes,
        #         'gt_classes': gt_classes,
        #         'gt_overlaps': overlaps,
        #         'flipped': False,
        #         'seg_areas': seg_areas}