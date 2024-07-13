import sys
import cv2
fname = sys.argv[1]
fnameout = sys.argv[2]
img = cv2.imread(fname, cv2.IMREAD_UNCHANGED)
#img = img[275:975,300:1200,:] #fps sample using 000001
#img = img[275:975,300:1100,:] # visualizing 000003

# day 3 exp 5
#img = img[400:750,700:1200,:] # front


# day 4 exp 14
#img = img[450:680,780:1100,:] # front
#img = img[375:650,700:1150,:] # top


# day 4 exp 19
#img = img[400:650,750:1100,:] # front
#img = img[375:650,800:1050,:] # top

# day 5 exp 18
#img = img[400:700,800:1110,:]  # front
#img = img[400:680,825:1060,:] # top


# day 5 exp 6
#img = img[400:700,750:1110,:] # top # front


# day 5 exp 11
#img = img[400:700,800:1110,:] # top # front

# day 13 confroom 
#img = img[300:800,700:1200,:] # top # front

# day 10
img = img[300:800,700:1200,:] # top # front

cv2.imwrite(fnameout, img)
