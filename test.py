import cv2
import numpy as np

# img = cv2.imread('bugfix_images/office.jpg')
img = cv2.imread('bugfix_images/test2.png')
# img = cv2.resize(img,(800,400))

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.medianBlur(gray,3)

ret,thresh = cv2.threshold(gray,1,255,0)
contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    rect = cv2.boundingRect(c)
    print rect
    if rect[2] < 10 or rect[3] < 10: continue
    print cv2.contourArea(c)
    x,y,w,h = rect
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,255,255),-1)
    # cv2.putText(img,'Moth Detected',(x+w+10,y+h),0,0.3,(0,255,0))
cv2.imshow("Show",img)
cv2.waitKey()  
cv2.destroyAllWindows()