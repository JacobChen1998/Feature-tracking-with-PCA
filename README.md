
# Algorithm Description
Traditional feature tracking techniques such as SIFT, SURF, and Lucas Kanade algorithms define key points in terms of finding poles and cannot specify specific tracking points. The general Deep Learning based tracking algorithms such as Siamese tracker require a lot of resources for neural network training. Here, we implement feature tracking using PCA. Our algorithm can specify tracking points and does not require extensive training.

The algorithm of SIFT is to find several poles and then extract the n x n interval centered on each poles into 128 feature values.
We refer to the concept of SIFT.
We cut the image into several mxn blocks in 1 pixel steps.
Then we input them into PCA for model training to achieve feature extraction at each point.
Therefore, each block will be extracted into 128 features.

The keypoint desired to track is selected in the reference frame. 
The width and height of target are also adjustable.
By "poses" we can match which the feature set is the reference feature set.

Similarly, block cutting is performed on the tracked frame.
Then use the trained PCA model to perform feature extraction on all blocks, so we can get a features set of each block.
  
The ideal feature set of the tracking point will theoretically have minimum error value with features set of reference keypoint. 
We use Sum of Squares of Residuals (SSR) as the metric.

Working pipeline:

![Demo_frames](https://github.com/JacobChen1998/Feature-tracking-with-PCA/blob/main/demo/flowchart.png)
  
 ## Prerequisites 
- Python > 3.6

## Quick start with anaconda 

#### 1. Environment create
```
    conda create --name pcatrack python=3.8
```

#### 2. Environment activate
```
    conda activate pcatrack
```

#### 3. Packages install
```
    pip install -r requirements.txt
```

#### 4. Adjust cfg.py file
```
Tracked_based_on_last = True # if False, tracked based on first frame
NORMALIZATION = True # Img Crop Normalization before PCA fit/transform
focus_region = True # Focus tracking on specific region
source = "samples/ntu_sample.avi" # video source
RizeRatio = 5 # lower resolution tracking avoid memory insufficient
featureNum = 128 # mapping feature num
```

#### 5. Run code
```
   python ObjTrack_PCA.py
```

## Some demo

### car tracking
 
![Ref_frame](https://github.com/JacobChen1998/Feature-tracking-with-PCA/blob/main/demo/reference_frame.png)
![Org_frames](https://github.com/JacobChen1998/Feature-tracking-with-PCA/blob/main/demo/origin.gif)
![Demo_frames](https://github.com/JacobChen1998/Feature-tracking-with-PCA/blob/main/demo/demo.gif)
  
Reference frame         /        Tracked frames      /           Tracking result

### person tracking

![Demo_frames](https://github.com/JacobChen1998/Feature-tracking-with-PCA/blob/main/demo/demo_person.gif)

