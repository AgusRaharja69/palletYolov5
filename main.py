from yoloDetection import run
import cv2
import numpy as np
import json
# import platform # When using Linux

weights_name = 'yolov5s-100'
cv_font = cv2.FONT_HERSHEY_SIMPLEX
jsonLidarPath = '../lidarJson.json'
angle, lidarDist = [], []

''' Euclidean '''
def compute_euclidean(x, y):
    return np.sqrt(np.sum((x-y)**2))

''' Find focal length of camera using Triangle Similiarity Principle on Pinhole Camera Model '''
def focal_length(xy,confidence):       
    # RESULT (yolom = 952.69, yolos = 989.057, yolon = 975.829)
    W = 50 # width of knowing object
    D = 100 # distance from camera
    P = int(xy[2] - xy[0])
    F = round(P*D/W, 3)
    fUse = 0
    if confidence > 0.5:
        fUse = F
        # print("Focal Length: ", fUse)
    return F, fUse

''' Find distance using Triangle Similiarity Principle on Pinhole Camera Model '''
def dist_pinhole(xy,X,Y,im0_shape,confidence):
    W = 50
    F = 952.69 if weights_name == 'yolov5m-100' else 989.057 if weights_name == 'yolov5s-100' else 975.829
    D = 0

    if confidence > 0.4:
        # print("xy: ",xy)
        # print("X: ", X)
        # print("Y: ", Y)
        # print("imShape: ",im0_shape)
        # print("====================================") 
        P = int(xy[2] - xy[0])
        CenterD = W * F / P  # center distance
        dA = np.array((X, Y))
        dB = np.array((int(im0_shape / 2), Y))
        wA = np.array(p1)
        wB = np.array((int(xy[2]), int(xy[1])))
        SideD = (compute_euclidean(dA, dB) / compute_euclidean(wA, wB)) * W  # Find the distance of center object to center frame
        D = round(np.hypot(CenterD, SideD), 3)
        # print("Distance: ",D)
    return D

''' Find distance using 2D LiDAR camera fusion with angle similiarity '''
def dist_lidar(xy,X,Y,im0_shape,confidence):
    ''' Opening LiDAR data '''  
    with open(jsonLidarPath) as f:
        try :
            lidar_data = json.load(f)
        except:
            lidar_data = {"data": [[0,0,0]]}

    # LiDAR data
    dataLidar = np.array(lidar_data['data'])
    angleRawLidar = np.radians(dataLidar[:, 1]) # Convert angle to radians
    rangeLidar = dataLidar[:, 2]
    angleLidar = np.pi - angleRawLidar #mirror
    
    # Angle by camera
    FoV = 55
    dispPix = int(im0_shape)

    detectWidth = round(int(xy[2] - xy[0])/5)
    AnglePix = [int(xy[0]),(int(xy[0])+detectWidth),(int(xy[2])-detectWidth),int(xy[2])]
    angleCamDet = [x*(FoV/dispPix) for x in AnglePix]
    angleCam = np.radians([90 + (FoV/2) - x for x in angleCamDet])

    # Scan dist
    angleArrayId = [angleLidar.index(x) for x in angleLidar if (x>=angleCam[0] and x<=angleCam[1]) or (x>=angleCam[2] and x<=angleCam[3])]
    D = np.median([rangeLidar[x] for x in angleArrayId])    
    # D = lidarDist
    return D

def projection(X,Y):
    lidarPixelX = X
    lidarPixelY = Y
    return lidarPixelX, lidarPixelY

''' Main Program '''
for im0, det in run():    
    ''' Write results '''
    for *xyxy, conf, cls in reversed(det):
        lw = 1
        txt_color = (255, 255, 255)
        color = (10, 15, 255)
        label = f'{"Wooden-Pallet"} {conf:.2f}'

        p1, p2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3]))
        cv2.rectangle(im0, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(im0, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(im0,
                        label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                        0,
                        lw / 3,
                        txt_color,
                        thickness=tf,
                        lineType=cv2.LINE_AA)
        x = int((xyxy[0] + xyxy[2]) / 2)
        y = int((xyxy[1] + xyxy[3]) / 2)

        ''' Edit Display Camera '''
        # coordinate_text = "(" + str(round(x)) + "," + str(round(y)) + ")" # pixel coordinate
        # cv2.putText(im0, text=coordinate_text, org=(x, y), # center coordinate pixel value
        # fontFace = cv_font, fontScale = 0.7, color=(255,255,255), thickness=1, lineType = cv2.LINE_AA)
        cv2.circle(im0, (x, y), 5, (0, 255, 0), -1)  # center coordinate
        # cv2.circle(im0,(int(im0.shape[1]/2),int(im0.shape[0]/2)), 5, (0,255,255), -1) # center frame
        # cv2.line(im0, (x, 0), (x, imgsz[1]), (0, 100, 0), thickness=2) # vertical line
        # cv2.circle(im0,(int(xyxy[0]), int(xyxy[1])), 5, (0,255,0), -1) # start coordinate
        # cv2.circle(im0,(int(xyxy[2]), int(xyxy[3])), 5, (255,0,0), -1) # end coordinate
        cv2.putText(im0, text="weights model : " + weights_name, org=(15, 20),  # weights names
        fontFace=cv_font, fontScale=0.6, color=(255, 55, 25), thickness=1, lineType=cv2.LINE_AA)

        ''' Find Object Distance '''
        dist = dist_pinhole(xyxy,x,y,im0.shape[1],conf)
        print('Distance: ',dist)

    ''' Stream results '''
    im0 = np.asarray(im0)
        # if platform.system() == 'Linux' and p not in windows:
        #     windows.append(p)
        #     cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
        #     cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])

    while(len(im0)==480):
        cv2.imshow(str(1), im0)
        key = cv2.waitKey(1)  # 1 millisecond
        if key == ord('s'):
            from pathlib import Path
            image_dir = Path("exp-data/images")
            nb_files = len(list(image_dir.glob("saved_img_*.jpg")))
            cv2.imwrite(str(image_dir / f"saved_img_{nb_files}.jpg"), im0)

