from PIL import Image
import os 

path = r'.\data\js_data\js_file\\'

old_name = 'HoleImage'
new_name = '000000'
ext = '.json'
idx = '1'

for i in range(477):
    # Count the Number of Digits in a Number 
    count = 0
    n = int(idx)
    while(n > 0):
        count += 1
        n = n // 10

    os.rename((path + old_name + idx + ext), (path + new_name[:-count] + idx + ext))
    idx = str(int(idx) + 1)

print('Done')
