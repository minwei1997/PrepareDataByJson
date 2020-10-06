import numpy as np 
from numpy import random
import cv2

def rotate_im(image, angle):
    """Rotate the image.
    
    Rotate the image such that the rotated image is enclosed inside the tightest
    rectangle. The area not occupied by the pixels of the original image is colored
    black. 
    
    Parameters
    ----------
    
    image : numpy.ndarray
        numpy image
    
    angle : float
        angle by which the image is to be rotated
    
    Returns
    -------
    
    numpy.ndarray
        Rotated Image
    
    """

    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    image = cv2.warpAffine(image, M, (nW, nH))

    return image

def get_corners(bboxes):
    """Get 4 corners of bounding boxes
    
    Parameters
    ----------
    
    bboxes: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    
    returns
    -------
    
    numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates :
        (x1 y1) -------- (x2 y2)       
            |                |
            |                |
        (x3 y3) -------- (x4 y4)
    """
    width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
    height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
    
    x1 = bboxes[:,0].reshape(-1,1)
    y1 = bboxes[:,1].reshape(-1,1)
    
    x2 = x1 + width
    y2 = y1 
    
    x3 = x1
    y3 = y1 + height
    
    x4 = bboxes[:,2].reshape(-1,1)
    y4 = bboxes[:,3].reshape(-1,1)
    
    corners = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))
    
    return corners

def rotate_box(corners, angle, cx, cy, h, w):   
    """Rotate the bounding box.
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    
    angle : float
        angle by which the image is to be rotated
        
    cx : int
        x coordinate of the center of image (about which the box will be rotated)
        
    cy : int
        y coordinate of the center of image (about which the box will be rotated)
        
    h : int 
        height of the image
        
    w : int 
        width of the image
    
    Returns
    -------
    
    numpy.ndarray
        Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
    """

    # shape of corner -> (4n x 2)
    corners = corners.reshape(-1,2)
    corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
    
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))
    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cx
    M[1, 2] += (nH / 2) - cy
    # Prepare the vector to be transformed
    calculated = np.dot(M,corners.T).T
    
    calculated = calculated.reshape(-1,8)
    
    return calculated

def get_enclosing_box(corners):
    """Get an enclosing box for rotated corners of a bounding box
    
    Parameters
    ----------
    
    corners : numpy.ndarray
        Numpy array of shape `N x 8` containing N bounding boxes each described by their 
        corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
    
    Returns 
    -------
    
    numpy.ndarray
        Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
        
    """
    x_ = corners[:, [0,2,4,6]]
    y_ = corners[:, [1,3,5,7]]
    
    xmin = np.min(x_,1).reshape(-1,1)
    ymin = np.min(y_,1).reshape(-1,1)
    xmax = np.max(x_,1).reshape(-1,1)
    ymax = np.max(y_,1).reshape(-1,1)
    
    final = np.hstack((xmin, ymin, xmax, ymax))
    
    return final

def bbox_area(bbox):
    """Caluclulate bounding boxes' area
    
    Parameters
    ----------
    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`

    Returns
    -------
    
    numpy.ndarray
        Numpy array containing all bounding boxes's area of shape `N` where N is the 
        number of bounding boxes 
        format `area1 area2 ...` 
    """

    w = bbox[:, 2] - bbox[:, 0]
    h = bbox[:, 3] - bbox[:, 1]
    area = w * h

    return area

def clip_box(bbox, clip_box, alpha):
    """Clip the bounding boxes to the borders of an image
    
    Parameters
    ----------
    
    bbox: numpy.ndarray
        Numpy array containing bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes and the bounding boxes are represented in the
        format `x1 y1 x2 y2`
    
    clip_box: numpy.ndarray
        An array of shape (4,) specifying the diagonal co-ordinates of the image
        The coordinates are represented in the format `x1 y1 x2 y2`
        
    alpha: float
        If the fraction of a bounding box left in the image after being clipped is 
        less than `alpha` the bounding box is dropped. 
    
    Returns
    -------
    
    numpy.ndarray
        Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the 
        number of bounding boxes left are being clipped and the bounding boxes are represented in the
        format `x1 y1 x2 y2` 
    
    """
    area = bbox_area(bbox)
    x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
    y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
    x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
    y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)
    bbox = np.hstack((x_min, y_min, x_max, y_max))

    # fraction of area between new boxes and original boxes 
    delta_area = bbox_area(bbox)/area

    # to record the box that need to reserve
    mask = np.ones((bbox.shape[0], ), dtype=np.bool)
    for i in range(bbox.shape[0]):
        if bbox[i, 0] == bbox[i, 2]:    # bounding box's 2 x coord are excced img
            mask[i] = False
        elif bbox[i, 1] == bbox[i, 3]:  # bounding box's 2 y coord are excced img
            mask[i] = False
        elif delta_area[i] < alpha:     # new box's area is less than threshold
            mask[i] = False

    return bbox, mask

def Rot_img_bbox(angle, img, bboxes):
    w, h = img.shape[1], img.shape[0]
    cx, cy = w//2, h//2

    img = rotate_im(img, angle)     # rotate img
    corners = get_corners(bboxes)   # get (4 corner's coord (total number = 8))
    corners[:,:8] = rotate_box(corners[:,:8], angle, cx, cy, h, w)  # rotate the corner coord

    new_bbox = get_enclosing_box(corners)   # get the minimum enclose box

    # img will enlarge when rotate. So we need to resize img to original size  
    # and box need to resize too
    scale_factor_x = img.shape[1] / w
    scale_factor_y = img.shape[0] / h
    img = cv2.resize(img, (w,h))
    new_bbox[:,:4] = new_bbox[:,:4] / [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 

    bboxes  = new_bbox
    
    bboxes, mask = clip_box(bboxes, [0,0,w, h], 0.25)

    return img, bboxes, mask


if __name__ == '__main__':
    img = cv2.imread('..\data\Defect_Img\HoleImage36.png')
    box = np.array([[ 511, 2077,  658, 2221], [3881,  465, 4111,  665]], dtype=np.uint16)
    img, bboxes, mask = Rot_img_bbox(30, img, box)
    print(mask)
    bboxes = bboxes[mask]

    target_size = 800
    im_scale = target_size / img.shape[1]
    img = cv2.resize(img, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_CUBIC)

    bboxes = bboxes * im_scale
    for i in range(bboxes.shape[0]):
        x1, y1, x2, y2 = bboxes[i, :].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        x1, y1, x2, y2 = bboxes[i, :].astype(int)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    cv2.imshow('window', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()