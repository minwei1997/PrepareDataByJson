import json
import os
import os.path as osp
import pickle 
from utils.generate_json_output import js_to_output
from utils.data_extract_v2 import Data_extractor
from utils.StringSortByDigit import natural_keys

''' Prepare xml file(for Faster RCNN use) from json file's data
'''
if __name__ == '__main__':
    print('\n')
    print('Start Processing....')
    print('-'*30)

    # get all json file under the path
    JsonDir = r'.\data\js_data\js_file'
    JsonExt = '.json'
    AllJsonFile = [files for files in os.listdir(JsonDir) if files.endswith(JsonExt)]       # xxx.json
    AllJsonFile.sort(key=natural_keys)    # Sort by file's index
    #  get all image file under the path
    ImgDir = r'.\data\Defect_Img'
    ImgExt = '.jpg'
    AllImgFile = [files for files in os.listdir(ImgDir) if files.endswith(ImgExt)]          # xxx.jpg
    AllImgFile.sort(key=natural_keys)    # Sort by file's index

    if len(AllJsonFile) != len(AllImgFile):     # check if number is the same
        raise ValueError('Number of AllJsonFile is not the same as AllImgFile.')

    XmlSavePath = osp.abspath(r'.\data\xml_file')    # path to save xml file
    ImgSavePath = osp.abspath(r'.\data\Defect_Img')   # path to save img
    _Data_extractor = Data_extractor(XmlSavePath, ImgSavePath)
    for ix, files in enumerate(AllJsonFile):
        ''' about generate xml file '''
        with open(osp.join(JsonDir, files) , 'r') as reader:
            jf = json.loads(reader.read())
            num_objs = len(jf['shapes'])

            # To generate xml file
            _Data_extractor.xml_generate(num_objs, OldImgName=AllImgFile[ix], JsFile=jf, AugMode=False)   

        ''' about generate json's output (5 files) '''
        js_to_output(files)
        print('{} output done!'.format(files))
        print('-'*30)

    print('\nProcess Done.\n')