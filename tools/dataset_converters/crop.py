import sys
import cv2
fname = sys.argv[1]
fnameout = sys.argv[2]
img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
img = img[275:975,300:1200,:]
cv2.imwrite(fnameout, img)
