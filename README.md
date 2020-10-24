# Driver-Drowsiness-Detection-using-GRU-with-CNN-features

This algorithm detects driver drowsiness detection using Deep learning concepts like , CNN, RNN, GRU.
We use yawning as our basis

## Workflow

1)We extract frames from incoming video stream using openCV.
2)Extracted frames are sent to custom dlib shape predictor to predict mouth region and extract it.
2)We train our custom dlib shape predictor as guided in this well written article https://www.pyimagesearch.com/2019/12/16/training-a-custom-dlib-shape-predictor/

3)Cropped mouth region in sent in the features extractor, which extracts required spatial features.
4)Finally, features are concatenated in sets of 32 and are sent to yawn detector for yawning classification.

## 
## Dependencies


## Run it on your own 
1)Clone the repository
2)
