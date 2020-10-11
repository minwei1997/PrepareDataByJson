import numpy as np 
import cv2 

img = '.\data\Defect_Img\\000168.jpg'
img = cv2.imread(img)
target_size = 800
im_scale = target_size / img.shape[1]
img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)
bbox = np.array([[2637, 1372, 2889, 1552]]) * im_scale

for i in range(bbox.shape[0]):
    x1, y1, x2, y2 = bbox[i, :].astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    x1, y1, x2, y2 = bbox[i, :].astype(int)
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow('window', img)
cv2.waitKey(0)
cv2.destroyAllWindows()