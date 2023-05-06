Tracked_based_on_last = True # 是否根據上一frame來追蹤? (True) 還是根據第一frame (False)
NORMALIZATION = True
focus_region = True
source = "samples/ntu_sample.avi"
# source = "samples/03555114_20221128_Hand_L_L.mp4FT.SVFI_Render.239fps.SR=realesr-animevideov3-x2_884983.mp4"
RizeRatio = 5 #5
import cv2
import numpy as np
import time 
import os
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# import cupy as cp
import imageio
symbo = "-"

def roi_choose(frame,winName = "ROI",ratioZoom=1):
        visu = frame.copy()
        roiH, roiW, _ = visu.shape
        x, y, w, h = cv2.selectROI(windowName=winName, img=cv2.resize(visu,(int(roiW/ratioZoom),int(roiH/ratioZoom))),showCrosshair=True, fromCenter=False)
        x, y, w, h = int(x*ratioZoom), int(y*ratioZoom), int(w*ratioZoom), int(h*ratioZoom)
        cv2.destroyWindow(winName)
        return x, y, w, h

def slideWindowCrop(img,winW,winH,normalization=NORMALIZATION):
        imgH,imgW,_ = img.shape 
        rois = np.zeros(((imgW-winW-1)*(imgH-winH-1),(winW*winH*3)))
        poses = []
        count=  0
        for x in range(imgW-winW-1):
                for y in range(imgH-winH-1):
                        x1,x2,y1,y2 = x,x+winW,y,y+winH
                        pose = [np.floor(winW/2).astype(int)+x,np.floor(winH/2).astype(int)+y]
                        rois[count] = img[y1:y2,x1:x2].flatten()
                        poses.append(pose)
                        count+=1
        return rois, poses

# Load the video
cap = cv2.VideoCapture(source)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# frameH,frameW = cap.get(cv2.CAP_PROP_FRAME_HEIGHT), cap.get(cv2.CAP_PROP_FRAME_WIDTH)
count = 0; reference_frame = None
xFOCUS,yFOCUS,wFOCUS,hFOCUS = None,None,None,None

if not os.path.exists("results"):
        os.mkdir("results")
        
while True:
        ret, frame = cap.read()
        if not ret:
                print("finished video........")
                break
        if focus_region:
                if count == 0:
                        xFOCUS,yFOCUS,wFOCUS,hFOCUS = roi_choose(frame,winName="Choose the focus region")
                frame = frame[yFOCUS:yFOCUS+hFOCUS,xFOCUS:xFOCUS+wFOCUS]
        frameH,frameW, _ = frame.shape
        frame = cv2.resize(frame,(int(frameW/RizeRatio),int(frameH/RizeRatio)))
        if RizeRatio!=1:
                frameH,frameW, _ = frame.shape
        if count == 0:

                x,y,w,h = roi_choose(frame,ratioZoom=0.5)
                x1,y1,x2,y2  = x,y,x+w,y+h
                x1_org,y1_org,x2_org,y2_org = x1,y1,x2,y2
                window_w, window_h = w,h
                reference_keypoint = np.array([(x2+x1)/2,(y2+y1)/2]).astype(int)
                visu = frame.copy()
                cv2.rectangle(visu,(x1,y1),(x2,y2),(0,0,255),2)
                fig, (ax1,ax2) = plt.subplots(1,2, figsize=(10,5)) 
                ax1.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                ax2.imshow(cv2.cvtColor(frame[y1:y2,x1:x2], cv2.COLOR_BGR2RGB))
                plt.show()
                rois,poses = slideWindowCrop(frame,window_w,window_h)

                # *********************************************************
                print("\n\nrois shape : ", rois.shape)
                print(rois.shape,rois.reshape(rois.shape[0],-1).shape)
                t1PCA = time.time()
                pca = PCA(n_components=128)
                print(f"{symbo*80}")
                print("\n\n[PCA fitting]")
                reference_frame_features = pca.fit_transform(rois)
                t2PCA = time.time()
                print(f"Finished fit ==> take time {t2PCA-t1PCA} (sec)")
                print('Shape of features sets : ', reference_frame_features.shape)
                kp_index = np.where((np.array(poses).T[0]== reference_keypoint[0]) & (np.array(poses).T[1]== reference_keypoint[1]))[0][0]
                referenceFeatures = reference_frame_features[kp_index]
                print('Index of keypoint in pose: ', kp_index)
                print('Shape of reference feature set : ', referenceFeatures.shape)
        
        else: # tracked frame
                print(f"\n\n{symbo*80}")
                print(f"Frame : {count+1}      Process : {int(100*((count+1)/frame_count))} %")
                t1 = time.time()
                crops,_ = slideWindowCrop(frame,window_w,window_h)
                tracked_frame_features = pca.transform(crops) # Feature extraction by PCA
                ssrs = np.sum((tracked_frame_features - referenceFeatures)**2,axis = 1)
                print('Minimum SSR position : ', np.argmin(ssrs),', with value : ',np.min(ssrs),', Shape of SSRs map : ', ssrs.shape)
                tracked_keypoint = poses[np.argmin(ssrs)]
                tracked_x1,tracked_x2 = tracked_keypoint[0] - window_w//2, tracked_keypoint[0] + window_w//2
                tracked_y1,tracked_y2 = tracked_keypoint[1] - window_h//2, tracked_keypoint[1] + window_h//2
                t2 = time.time()
                visuTracked = frame.copy()
                cv2.rectangle(visuTracked,(tracked_x1,tracked_y1),(tracked_x2,tracked_y2),(0,0,255),2)
                # cv2.imwrite(f"results/{count}_TrackedSSR_{np.min(ssrs)}.png",np.vstack([visu,visuTracked]))
                cv2.imwrite(f"results/{count}.png",np.vstack([visu,visuTracked]))

                if Tracked_based_on_last:
                        visu = visuTracked.copy()
                        referenceFeatures = tracked_frame_features[np.argmin(ssrs)]

        # Check for key presses
        if cv2.waitKey(1) & 0xFF == ord('q'):
                break           
        count+=1

cap.release()
cv2.destroyAllWindows()