from PIL import Image
import os 

from utils.StringSortByDigit import natural_keys

path = r'D:\user\Desktop\Defect_Img\\'

# Get all image endwith .png
img_fileExt = '.png'
all_img = [files for files in os.listdir(path) if files.endswith(img_fileExt)]    
all_img.sort(key=natural_keys)    # Sort by file's index

new_name = '000000'
ext = '.png'
idx = '1'

Img_Num = 94
for i in range(Img_Num):
    # Count the Number of Digits in a Number 
    count = 0
    n = int(idx)
    while(n > 0):
        count += 1
        n = n // 10

    os.rename((path + all_img[i]), (path + new_name[:-count] + idx + ext))
    idx = str(int(idx) + 1)

print('Done')
