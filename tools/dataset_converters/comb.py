import cv2
import os, sys
import numpy as np
from typing import Tuple

dirs = sys.argv[1:]

files = os.listdir(dirs[0])
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
            x1, x2 = max(0, x1-100), min(x2+100, img.shape[0])
            y1, y2 = max(0, y1-100), min(y2+100, img.shape[1])
        if(curr_dir.split('/')[-1]=='image'):
            img = cv2.resize(img, (y2-y1, x2-x1))
        else:
            img = img[x1:x2,y1:y2,:]
        cv2.imwrite(os.path.join(curr_dir, 'inc-'+file), img)
        all_img.append(img)
    all_img = np.concatenate(all_img, axis=1)
    cv2.imwrite(os.path.join(dirs[-1], file), all_img)



