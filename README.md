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

![Demo_frames](https://github.com/JacobChen1998/Feature-tracking-with-PCA/blob/main/flowchart.png)
  
 ## Prerequisites 
- Python > 3.6

## Quick start with anaconda 

#### 1. Environment create
```
    conda create --name django-stream python=3.8
```

#### 2. Environment activate
```
    conda activate django-stream
```

#### 3. Packages install
```
    pip install -r requirements.txt
```

#### 4. Replace playing source in [cfg.py](https://github.com/JacobChen1998/Streaming-with-Django-and-OpenCV/blob/main/cfg.py)
```
   stream_link =  <your source>
```

#### 5. Run the server
```
   python manage.py runserver
```

![Ref_frame](https://github.com/JacobChen1998/Feature-tracking-with-PCA/blob/main/reference_frame.png)
![Org_frames](https://github.com/JacobChen1998/Feature-tracking-with-PCA/blob/main/origin.gif)
![Demo_frames](https://github.com/JacobChen1998/Feature-tracking-with-PCA/blob/main/demo.gif)
  
Reference frame         /        Tracked frames      /           Tracking result

Working pipeline:


