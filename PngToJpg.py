from PIL import Image

img_path = '.\data\Defect_Img'
name = 'HoleImage'
idx = '1'
num_img = 477
old_ext = '.png'
new_ext = '.jpg'

for i in range(num_img):
    print('process ing : ', i)
    im1 = Image.open(img_path + '\\' + name + idx + old_ext)
    im1.save('.\\temp\\' + name + idx + new_ext)
    idx = str(int(idx)+1)