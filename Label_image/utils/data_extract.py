from scipy import sparse
import numpy as np 


class Data_extractor():
    ''' use to generate the gt_roidb
         roidb is a dict with 5 keys  
            -> {boxes(N x 4), gt_classes(N, ), gt_overlap(N x num_cls), flipped(bool), seg_area(N, )}
    '''

    def __init__(self, classes):
        super().__init__()
        self.classes = classes              # all cls type
        self.num_classes = len(classes)     # number of cls
        self._class_to_ind = dict(list(zip(self.classes, list(range(self.num_classes)))))   # function that covert cls to corresponding idx
        
            
    def data_prepare(self, json_file):
        ''' generate roidb '''

        num_objs = len(json_file['shapes'])    # num of objs in img

        # initialize params
        boxes = np.zeros((num_objs, 4), dtype=np.uint16)
        gt_classes = np.zeros((num_objs), dtype=np.int32)
        overlaps = np.zeros((num_objs, self.num_classes), dtype=np.float32)      # (N x num_cls) only the overlaped gt cls loaction have value(1)
        seg_areas = np.zeros((num_objs), dtype=np.float32)


        for ix in range(num_objs):
            cls = json_file['shapes'][ix]['label']              # get the cls label 
            bbox = json_file['shapes'][ix]['points']      # get the bbox coord

            # Make pixel indexes 0-based
            x1 = bbox[0][0] - 1
            y1 = bbox[0][1] - 1
            x2 = bbox[1][0] - 1
            y2 = bbox[1][1] - 1

            cls = self._class_to_ind[cls.lower().strip()]    # conver cls to corresponding index   
            boxes[ix, :] = [x1, y1, x2, y2]
            gt_classes[ix] = cls
            overlaps[ix, cls] = 1.0
            seg_areas[ix] = (x2 - x1 + 1) * (y2 - y1 + 1)

        overlaps = sparse.csr_matrix(overlaps)

        return {'boxes': boxes,
                'gt_classes': gt_classes,
                'gt_overlaps': overlaps,
                'flipped': False,
                'seg_areas': seg_areas}