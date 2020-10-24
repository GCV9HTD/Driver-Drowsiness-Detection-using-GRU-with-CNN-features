# Driver-Drowsiness-Detection-using-GRU-with-CNN-features

This algorithm detects driver drowsiness detection using Deep learning concepts like , CNN, RNN, GRU.
We use yawning as our basis

## Workflow

1)We extract frames from incoming video stream using openCV.
2)Extracted frames are sent to custom dlib shape predictor to predict mouth region and extract it.
2)We train our custom dlib shape predictor as guided in this well written article https://www.pyimagesearch.com/2019/12/16/training-a-custom-dlib-shape-predictor/

3)Cropped mouth region in sent in the features extractor, which extracts required spatial features.
4)Finally, features are concatenated in sets of 32 and are sent to yawn detector for yawning classification.

## Custom dlib shape predictor 
Custom dlib shape predictor has been trained on [i-bug 300W](https://ibug.doc.ic.ac.uk/resources/300-W/) using steps provided [here](https://www.pyimagesearch.com/2019/12/16/training-a-custom-dlib-shape-predictor/)
Custom dlib shape predictor folder contains the following file.

1)parse_xml.py : Parses the train/test XML dataset files for eyes-only landmark coordinates.

2)train_shape_predictor.py : Accepts the parsed XML files to train our shape predictor with dlib.

3)evaluate_shape_predictor.py : Calculates the Mean Average Error (MAE) of our custom shape predictor to test and validate our model.

## Dependencies


## Run it on your own 
1)Clone the repository
2)
