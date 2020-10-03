import json
import os
import os.path as osp
import pickle 
from utils.generate_json_output import js_to_output
from utils.data_extract import Data_extractor


if __name__ == '__main__':
    print('\n\n')
    print('Start Processing....')
    print('-'*30)

    # get all json file under the path
    fileDir = './data/js_data/js_file/'
    fileExt = '.json'
    all_json_file = [files for files in os.listdir(fileDir) if files.endswith(fileExt)]    

    # define all cls type
    classes = ('__background__',  # always index 0
                'hole')

    gt_roidb = []
    Data_extractor = Data_extractor(classes)

    for ix, files in enumerate(all_json_file):
        ''' about generate roidb '''
        with open(osp.join(fileDir, files) , 'r') as reader:
            jf = json.loads(reader.read())
            # get dict data of {boxes(N x 4), gt_classes(N, ), gt_overlap(N x num_cls), flipped(bool), seg_area(N, )}
            dict_data = Data_extractor.data_prepare(jf)     
            # append data to gt_roidb
            gt_roidb.append(dict_data)

        ''' about generate json's output (5 files) '''
        js_to_output(files)
        print('{} output done!'.format(files))
        print('-'*30)

    # save data to pkl file
    pkl_path = './data/training_pickle/training_data_pkl.pkl'
    with open(pkl_path, 'wb') as fid:
        pickle.dump(gt_roidb, fid, pickle.HIGHEST_PROTOCOL)
        print('Wrote gt roidb to {}'.format(pkl_path))

    print('\nProcess Done.\n')