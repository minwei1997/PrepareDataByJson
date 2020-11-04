from scipy import sparse
import numpy as np 
import codecs
import cv2
import os.path as osp
import json 

class Data_extractor():
    ''' use to generate xml file '''

    def __init__(self, XmlSavePath, ImgSavePath):
        super().__init__()
        self.XmlSavePath = XmlSavePath
        self.ImgSavePath = ImgSavePath

    def xml_generate(self, num_objs, OldImgName=None, JsFile=None, AugMode=False, img=None, clss=None, bbox=None, NewImgName=None):
        ''' generate xml file '''

        img = cv2.imread(osp.join(self.ImgSavePath, OldImgName)) if not AugMode else img   # read image
        h, w, depth = img.shape     # get image shape

        height = h
        width = w
        depth = depth 

        _img_name = OldImgName if not AugMode else NewImgName
        with codecs.open(osp.join(self.XmlSavePath, _img_name.split('.')[0]+'.xml'), 'w', 'utf-8') as xml:
        # with codecs.open(self.XmlSavePath + '\\' + _img_name.split('.')[0] + '.xml', 'w', 'utf-8') as xml:
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
                x1 = int((JsFile['shapes'][j]['points'])[0][0]) if not AugMode else int(bbox[j, 0])
                y1 = int((JsFile['shapes'][j]['points'])[0][1]) if not AugMode else int(bbox[j, 1])
                x2 = int((JsFile['shapes'][j]['points'])[1][0]) if not AugMode else int(bbox[j, 2])
                y2 = int((JsFile['shapes'][j]['points'])[1][1]) if not AugMode else int(bbox[j, 3])
                _cls = JsFile['shapes'][j]['label'] if not AugMode else clss[j]

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

