import cv2
import imutils

cam=cv2.VideoCapture(0)
FirstFrame=None
ImageSize=500
txt="Normal"
while True:
    _,img=cam.read()
    img=imutils.resize(img,width=1000)
    grsimg= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussianimg=cv2.GaussianBlur(grsimg,(21,21),0)
    if FirstFrame is None:
        FirstFrame=gaussianimg
        continue
    imgdiff = cv2.absdiff(FirstFrame,gaussianimg)
    thresh =cv2.threshold(imgdiff,25,255,cv2.THRESH_BINARY)[1]
    thresh =cv2.dilate(thresh,None,iterations=2)
    conts=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    conts=imutils.grab_contours(conts)
    for c in conts:
        if cv2.contourArea(c)<ImageSize:
            continue
        x,y,w,h=  cv2.boundingRect(c)
        cv2.rectangle(img,(x,y,x+w,y+h),(0,255,0),2)
        txt="Moving object"
    cv2.putText(img,txt,(20,10),cv2.FONT_HERSHEY_DUPLEX,0.5,(0,0,255),2)
    print(txt)
    cv2.imshow("Camera Feed",img)
    key=cv2.waitKey(10)
    if key==ord("q"):
        break
cam.release()
cv2.destroyAllWindows()




