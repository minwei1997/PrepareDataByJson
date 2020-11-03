import json
import os
import os.path as osp
import pickle 
from utils.generate_json_output import js_to_output
from utils.data_extract import Data_extractor
from utils.StringSortByDigit import natural_keys

''' Prepare xml file(for Faster RCNN use) from json file's data
'''
if __name__ == '__main__':
    print('\n')
    print('Start Processing....')
    print('-'*30)

    # get all json file under the path
    fileDir = './data/js_data/js_file/'
    fileExt = '.json'
    all_json_file = [files for files in os.listdir(fileDir) if files.endswith(fileExt)]    
    all_json_file.sort(key=natural_keys)    # Sort by file's index 
    all_img_name = [all_json_file[i].split('.')[0] + '.jpg' for i in range(len(all_json_file))]

    save_path = osp.abspath(r'.\data\xml_file')    # path to save xml file
    img_path = osp.abspath(r'.\data\Defect_Img')   # img's path
    for ix, files in enumerate(all_json_file):
        ''' about generate xml file '''
        _Data_extractor = Data_extractor(save_path, img_path, all_img_name[ix])
        with open(osp.join(fileDir, files) , 'r') as reader:
            jf = json.loads(reader.read())
            num_objs = len(jf['shapes'])
            # To generate xml file
            _Data_extractor.xml_generate(num_objs, json_file=jf, aug_mode=False)   


        ''' about generate json's output (5 files) '''
        js_to_output(files)
        print('{} output done!'.format(files))
        print('-'*30)

    print('\nProcess Done.\n')

    # # save data to pkl file
    # pkl_path = './data/training_pickle/training_data_pkl.pkl'
    # with open(pkl_path, 'wb') as fid:
    #     pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
    #     print('Wrote gt roidb to {}'.format(pkl_path))

    # print('\nProcess Done.\n')