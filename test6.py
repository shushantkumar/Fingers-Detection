import numpy as np
import cv2
lo = np.array([0, 130, 101])
hi = np.array([198, 155, 148])
kernel2 = np.ones((3, 3), np.uint8)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
kodi = np.ones((480, 640, 3), np.uint8)
found = 0
#def putText(cv2,frame,txt):
 #   font = cv2.FONT_HERSHEY_SIMPLEX
 #   cv2.putText(frame,'Fingers:'+txt,(0,70), font, 1,(255,255,255),2,cv2.CV_AA)
#------------video capture from external camera(pass 1) or pass 0 for inbuilt camera-----------#
cap = cv2.VideoCapture(0)
while(1):
    ret, frame = cap.read()
    # ------ YCrCb Color space part--------#
    yc = cv2.cvtColor(frame,cv2.COLOR_BGR2YCR_CB)
    mask = cv2.inRange(yc,lo,hi)
    mask = cv2.GaussianBlur(mask,(0, 0), 2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel2, iterations=2)
    mask3 = cv2.bitwise_and(frame,frame, mask=mask)
    #------- Threshold and Contour extraction -----------------#
    ret,th = cv2.threshold(mask, 60, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    thh = th.copy()
    _,cnt, hie = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cov = len(cnt)
    numb = np.array([])
    if(cov>0):
        for i in range(cov):
            covt = cnt[i]
            ar = cv2.contourArea(covt)
            if(ar>230):
                M = cv2.moments(covt)
                x1 = int(M['m10']/M['m00'])
                y1 = int(M['m01']/M['m00'])
                centroid = (x1,y1)
                numb = np.append(numb,x1)
                cv2.circle(frame,(x1,y1),3,(0,24,206),3)
                x,y,w,h = cv2.boundingRect(covt)
#------------------------ hull and defects -----------------#
                try:
                    hull = cv2.convexHull(covt,returnPoints = False)
                    defects = cv2.convexityDefects(covt,hull)
                    for i in range(defects.shape[0]):
                        s,e,f,d = defects[i,0]
                        start = tuple(covt[s][0])
                        end = tuple(covt[e][0])
                        far = tuple(covt[f][0])
                except:
                    continue
#------------------------ hull and defects END-----------------#
        cv2.drawContours(frame,cnt,-1,(34,223,109),3)
  #  putText(cv2,frame,str(numb.shape[0]))
    cv2.imshow('kovi',mask3)
    cv2.imshow('Hand detec', frame)
    if(cv2.waitKey(20) == 27):
        break
cap.release()
cv2.destroyAllWindows()







