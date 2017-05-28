import cv2 
import numpy as np    

img=cv2.imread("debug_get_dim_l_input.jpg")

padding = 50

shape=img.shape
w=shape[1]
h=shape[0]

base_size=h+padding,w+padding,3
#make a 3 channel image for base which is slightly larger than target img
base=np.zeros(base_size,dtype=np.uint8)
cv2.rectangle(base,(0,0),(w+padding,h+padding),(255,0,0),padding)#really thick white rectangle
base[padding/2:h+padding/2,padding/2:w+padding/2]=img #this works

cv2.imshow('output',base)
cv2.waitKey(0)