
#Morphological Operations.ipynb

import cv2
import numpy
from google.colab.patches import cv2_imshow
image = cv2.imread('Image.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2_imshow(image)

"""Erosion"""
eroded = cv2.erode(gray.copy(), None, iterations=i + 1)
cv2_imshow(eroded)	

"""Dilation"""
dilated = cv2.dilate(gray.copy(), None, iterations=i + 1)
cv2_imshow(dilated)
	

"""Opening"""

op_image = cv2.imread('Open_Image.jpg')
kernel = np.ones((5,5),np.uint8)
opening = cv2.morphologyEx(op_image, cv2.MORPH_OPEN, kernel)
cv2_imshow(opening)

"""Closing"""
cl_image = cv2.imread('Close_Image.jpg')
closing = cv2.morphologyEx(cl_image, cv2.MORPH_CLOSE, kernel)
cv2_imshow(closing)


"""Skeleton"""
size = np.size(gray)
skel = np.zeros(gray.shape,np.uint8) 
ret,img = cv2.threshold(gray,127,255,0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
done = False 
while( not done):
    eroded = cv2.erode(img,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(img,temp)
    skel = cv2.bitwise_or(skel,temp)
    img = eroded.copy() 
    zeros = size - cv2.countNonZero(img)
    if zeros==size:
        done = True
 
cv2_imshow(skel)


"""Hit or Miss"""
kernel = np.ones((5,5),np.uint8)
hm_image = cv2.morphologyEx(gray, cv2.MORPH_HITMISS, kernel)
cv2_imshow(hm_image)