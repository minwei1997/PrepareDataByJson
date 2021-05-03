from PIL import Image
import os 

from utils.StringSortByDigit import natural_keys

''' Chageable Variables Setting '''
path = r'D:\user\Desktop\temp\\'
ext = '.jpg'
all_img = [files for files in os.listdir(path) if files.endswith(ext)]    # Get all image endwith .png
all_img.sort(key=natural_keys)    # Sort by file's index
new_name = '000000'

idx = '1'
Img_Num = len(all_img)
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
