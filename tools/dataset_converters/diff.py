import cv2
import matplotlib.pyplot as plt

im1 = cv2.imread('005051.png', cv2.IMREAD_UNCHANGED)
im2 = cv2.imread('depth.png', cv2.IMREAD_UNCHANGED)
diff = im2 - im1
print(im2)

plt.ylim([0,70000])
plt.xlim([0,60000])
plt.hist(im1.reshape(-1), 100)
plt.savefig('fig1.png')

plt.clf()
plt.ylim([0,70000])
plt.xlim([0,60000])
plt.hist(im2.reshape(-1), 100)
plt.savefig('fig2.png')


#print(im1.shape, im2.shape)

#print(im1[200:220, 200:220])
#print(im2[200:220, 200:220])
#print(diff[200:220, 200:220])
from PIL import Image
i1 = Image.open('005051.png')
i2 = Image.open('depth.png')
print(i1.mode, i2.mode)



