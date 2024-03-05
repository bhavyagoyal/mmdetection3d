import sys
import cv2
fname = sys.argv[1]
fnameout = sys.argv[2]
img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
#img = img[275:975,300:1200,:] #fps samples
#img = img[230:930,660:1130,:]

# exp 14
#img = img[350:700,750:1100,:]
#img = img[300:700,700:1150,:]


# exp 14
img = img[400:650,750:1100,:] # top # front


cv2.imwrite(fnameout, img)
