import cv2
import numpy as np

cap = cv2.VideoCapture(0)
i = 270 
classify = 1 
labels =[] 
while True:
    i += 1
    ret  , frame = cap.read()
    gray = cv2.cvtColor(frame , cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (15,15), 0)
    cv2.imshow('feed' , gray)
    
    if cv2.waitKey(1) & 0xFF == ord('q') :
        break 
    z = input()
    if z == 'i' :
        classify = 0
    elif z == 'e' :
        break
    cv2.imwrite("picture" + str(i) + ".jpg" , gray)
    print (classify)
    labels.append(classify)
    print (i)
    
np.array(labels).tofile('labels.dat')
print ("go")
cap.release()
cv2.destroyAllWindows()
