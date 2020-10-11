import numpy as np 
import os
import os.path as osp

img_path = './data/Defect_Img/'
all_img_file = [files for files in os.listdir(img_path) if files.endswith('.jpg')]      # get all .jpg files
num_img = len(all_img_file)
all_img_file = [all_img_file[i].strip('.jpg') for i in range(num_img)]        # remove suffix


trainval_percent = 0.7
test_percent = 1 - trainval_percent

# random permutation 
perm_idx = np.random.permutation(np.arange(num_img))
# choose trainval and test data by rand permutated index
trainval_data = np.asarray(all_img_file, dtype=np.str)[perm_idx[:(int(num_img * trainval_percent))]]
test_data = np.asarray(all_img_file)[perm_idx[(int(num_img * trainval_percent)):]]

train_percent = 0.5
val_percent = 1 - train_percent

num_trainval = len(trainval_data)
# random permutation 
perm_idx = np.random.permutation(np.arange(num_trainval))
# choose trainval and test data by rand permutated index
train_data = np.asarray(trainval_data)[perm_idx[:(int(num_trainval * train_percent))]]
val_data = np.asarray(trainval_data)[perm_idx[(int(num_trainval * train_percent)):]]

print('total number of data : ', len(all_img_file))
print('num trainval_data : ', len(trainval_data))
print('num test_data : ', len(test_data))
print('num train_data : ', len(train_data))
print('num val_data : ', len(val_data))


# save to txt file
txt_save_path = './trainval_set'

np.savetxt(osp.join(txt_save_path, 'trainval.txt'), trainval_data, fmt='%s')
np.savetxt(osp.join(txt_save_path, 'test.txt'), test_data, fmt='%s')
np.savetxt(osp.join(txt_save_path, 'train.txt'), train_data, fmt='%s')
np.savetxt(osp.join(txt_save_path, 'val.txt'), val_data, fmt='%s')