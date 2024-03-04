import cv2
import os, sys
import numpy as np
from typing import Tuple

dirs = sys.argv[1:]

files = open('gt.txt', 'r').readlines()
files = [x.strip() for x in files]
print(files)

for file in files:
    all_img = []
    x1, x2, y1, y2 = None, None, None, None
    for curr_dir in dirs[:-1]:
        img = cv2.imread(os.path.join(curr_dir,file), cv2.IMREAD_UNCHANGED)
        if(len(all_img)==0):
            x, y = np.where(img[:,:,0]>0)
            x1, x2 = min(x), max(x)
            y1, y2 = min(y), max(y)
            marginx, marginy = min(50,int((x2-x1)*0.1)), min(50,int((y2-y1)*0.1))
            x1, x2 = max(0, x1-marginx), min(x2+marginx, img.shape[0])
            y1, y2 = max(0, y1-marginy), min(y2+marginy, img.shape[1])
        if(curr_dir.split('/')[-1]=='image'):
            cv2.imwrite(os.path.join(curr_dir, 'inc-'+file[:-4]+'.jpg'), img)
            img = cv2.resize(img, (y2-y1, x2-x1))
        else:
            img = img[x1:x2,y1:y2,:]
            cv2.imwrite(os.path.join(curr_dir, 'inc-'+file[:-4]+'.jpg'), img)
        all_img.append(img)
    all_img = np.concatenate(all_img, axis=1)
    cv2.imwrite(os.path.join(dirs[-1], file), all_img)


