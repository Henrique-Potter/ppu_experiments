from pycocotools.coco import COCO
import numpy as np

import matplotlib.pyplot as plt
import pylab


dataDir = '..'
#data_type = 'stuff_train'
#data_type = 'captions_train'
data_type = 'instances_train'
#data_type = 'person_keypoints'

ins_annFile = './coco_annotations_db/instances_train2017.json'
key_points_annFile = './coco_annotations_db/person_keypoints2017.json'

ins_coco = COCO(ins_annFile)
keyp_coco = COCO(ins_annFile)

cats = ins_coco.loadCats(ins_coco.getCatIds())
nms1 = [cat['name'] for cat in cats]
print('COCO categories: \n{}\n'.format(' '.join(nms1)))

nms2 = set([cat['supercategory'] for cat in cats])
print('COCO supercategories: \n{}'.format(' '.join(nms2)))

catIds = ins_coco.getCatIds(catNms=['person'])

imgIds = ins_coco.getImgIds(catIds=catIds)

annIds = keyp_coco.getAnnIds(imgIds=imgIds[0], catIds=catIds, iscrowd=None)
anns = keyp_coco.loadAnns(annIds)

print(catIds)