import sys
import cv2
fname = sys.argv[1]
fnameout = sys.argv[2]
img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
#img = img[275:975,300:1200,:] fps samples
img = img[230:930,660:1130,:]
#img = img[420:650,750:1150,:]
cv2.imwrite(fnameout, img)
