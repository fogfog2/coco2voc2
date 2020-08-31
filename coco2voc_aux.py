from pycocotools import mask as maskUtils
import numpy as np


def annsToSeg(anns, coco_instance):
    '''
    converts COCO-format annotations of a given image to a PASCAL-VOC segmentation style label
     !!!No guarantees where segmentations overlap - might lead to loss of objects!!!
    :param anns: COCO annotations as returned by 'coco.loadAnns'
    :param coco_instance: an instance of the COCO class from pycocotools
    :return: three 2D numpy arrays where the value of each pixel is the class id, instance number, and instance id.
    '''
    image_details = coco_instance.loadImgs(anns[0]['image_id'])[0]

    h = image_details['height']
    w = image_details['width']

    class_seg = np.zeros((h, w))
    instance_seg = np.zeros((h, w))
    id_seg = np.zeros((h, w))
    masks, anns = annsToMask(anns, h, w)

    parsing = [0] * 92
    parsing[1] = 15
    parsing[2] = 2
    parsing[3] = 7
    parsing[4] = 14
    parsing[5] = 1
    parsing[6] = 6
    parsing[7] = 19
    parsing[9] = 4
    parsing[16] = 3
    parsing[17] = 8
    parsing[18] = 12
    parsing[19] = 13
    parsing[20] = 17
    parsing[21] = 10
    parsing[44] = 5
    parsing[62] = 9
    parsing[63] = 18
    parsing[64] = 16
    parsing[67] = 11
    parsing[72] = 20
    summ = 0

    #print(parsing)
    for i, mask in enumerate(masks):
       # print(anns[i]['category_id'])
        t = anns[i]['category_id']
        cat = parsing[anns[i]['category_id']]
       # print(t,",", cat)
        summ = summ + cat
        class_seg = np.where(class_seg>0, class_seg, mask*cat)
        instance_seg = np.where(instance_seg>0, instance_seg, mask*(i+1))
        id_seg = np.where(id_seg > 0, id_seg, mask * anns[i]['id'])
    
    if summ > 0 :
        iscontainsvoc = True
    else : 
        iscontainsvoc = False

    return class_seg, instance_seg, id_seg.astype(np.int64), iscontainsvoc


def annToRLE(ann, h, w):
    """
    Convert annotation which can be polygons, uncompressed RLE to RLE.
    :return: binary mask (numpy 2D array)
    """
    segm = ann['segmentation']
    if type(segm) == list:
        # polygon -- a single object might consist of multiple parts
        # we merge all parts into one mask rle code
        rles = maskUtils.frPyObjects(segm, h, w)
        rle = maskUtils.merge(rles)
    elif type(segm['counts']) == list:
        # uncompressed RLE
        rle = maskUtils.frPyObjects(segm, h, w)
    else:
        # rle
        rle = ann['segmentation']
    return rle


def annsToMask(anns, h, w):
    """
    Convert annotations which can be polygons, uncompressed RLE, or RLE to binary masks.
    :return: a list of binary masks (each a numpy 2D array) of all the annotations in anns
    """
    masks = []
    anns = sorted(anns, key=lambda x: x['area'])  # Smaller items first, so they are not covered by overlapping segs

    def is_uppersize(ann):
        return ann['area']>1000
    anns = list(filter(is_uppersize, anns))
    for ann in anns:
        rle = annToRLE(ann, h, w)
        m = maskUtils.decode(rle)
        masks.append(m)
    return masks, anns
