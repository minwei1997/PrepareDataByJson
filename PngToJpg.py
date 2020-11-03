from PIL import Image
import os 

''' Chageable Variables Setting '''
img_path = r'E:\MinWei\AI code\PrepareDataByJson\data\Defect_Img\\'
old_ext = '.png'
new_ext = '.jpg'

all_img = [files for files in os.listdir(img_path) if files.endswith(old_ext)]    # get all img under the path
all_img_name = [all_img[i].split('.')[0] for i in range(len(all_img))]            # remove surfix
num_img = len(all_img_name)

for i in range(num_img):
    print('processing image : ', all_img_name[i] + old_ext)
    im1 = Image.open(img_path + all_img_name[i] + old_ext)      # open image
    im1.save('.\\temp\\' + all_img_name[i] + new_ext)           # save with new sub filename