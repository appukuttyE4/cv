
#Feature Detection
#Harris corner Detection

import numpy as np
import cv2
from google.colab.patches import cv2_imshow
img = cv2.imread('Shapes.jpg')
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)
dst = cv2.cornerHarris(gray,2,3,0.04)
#result is dilated for marking the corners, not important
dst = cv2.dilate(dst,None)
# Threshold for an optimal value, it may vary depending on the image.
img[dst>0.01*dst.max()]=[0,0,255]
cv2_imshow(img)

"""SIFT"""
image8bit = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype('uint8') 
sift = cv2.xfeatures2d.SIFT_create()
kp = sift.detect(image8bit, None)
# Marking the keypoint on the image using circles
img=cv2.drawKeypoints(image8bit ,kp ,img)
cv2_imshow(img)

"""Shi - Tomasi"""

corners = cv2.goodFeaturesToTrack(gray,25,0.01,10)
corners = np.int0(corners)
for i in corners:
    x,y = i.ravel()
    cv2.circle(img,(x,y),3,255,-1)
cv2_imshow(img)

"""Fast Feature detector"""

# Initiate FAST object with default values
fast = cv2.FastFeatureDetector_create()
# find and draw the keypoints
kp = fast.detect(img,None)
img2 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
# Print all default params
print( "Threshold: {}".format(fast.getThreshold()) )
print( "nonmaxSuppression:{}".format(fast.getNonmaxSuppression()) )
print( "neighborhood: {}".format(fast.getType()) )
print( "\nTotal Keypoints with nonmaxSuppression: {}".format(len(kp)) )
cv2_imshow(img2)
# Disable nonmaxSuppression
fast.setNonmaxSuppression(0)
kp = fast.detect(img, None)
print( "\nTotal Keypoints without nonmaxSuppression: {}".format(len(kp)) )
img3 = cv2.drawKeypoints(img, kp, None, color=(255,0,0))
cv2_imshow(img3)