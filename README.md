# Driver-Drowsiness-Detection-using-GRU-with-CNN-features

This algorithm detects driver drowsiness detection using Deep learning concepts like , CNN, RNN, GRU.
We use yawning as our basis

## Dependencies
* python 3.6.10
* Tensorflow (I did this using tf 1.14, for tf 2.0 or greater some changes will be needed)
* dlib
* openCV
* sklearn
* matplotlib
* os
* [keras-surgeon](https://github.com/BenWhetton/keras-surgeon)
* numpy
* argparse
* imutils
* time

<p align="center">
  <img width="500" height="500" src="https://github.com/srivastava-ayush/Driver-Drowsiness-Detection-using-GRU-with-CNN-features/blob/main/Real%20Time%20Inference/fig_1._3_50%25.jpg">

## Workflow

Incoming Video Feed->extract mouth region from individual frames->Send them to feature extractor->concatenate 32 frames->send to yawn detector->Yawning/Not Yawning

We use 3 model in it
!)zcustom Dlib 
2Feature Extractor
3)Yawn detector

1)We extract frames from incoming video stream using openCV.
2)Extracted frames are sent to custom dlib shape predictor to predict mouth region and extract it.
2)We train our custom dlib shape predictor as guided in this well written article https://www.pyimagesearch.com/2019/12/16/training-a-custom-dlib-shape-predictor/

3)Cropped mouth region in sent in the features extractor, which extracts required spatial features.
4)Finally, features are concatenated in sets of 32 and are sent to yawn detector for yawning classification.

## Custom dlib shape predictor 
Custom dlib shape predictor has been trained using steps provided [here](https://www.pyimagesearch.com/2019/12/16/training-a-custom-dlib-shape-predictor/)

* Dataset Used : [i-bug 300W](https://ibug.doc.ic.ac.uk/resources/300-W/)

* parse_xml.py : Parses the train/test XML dataset files for eyes-only landmark coordinates.

* train_shape_predictor.py : Accepts the parsed XML files to train our shape predictor with dlib, saves custom shape predictor with .dat file extension.

* evaluate_shape_predictor.py : Calculates the Mean Average Error (MAE) of our custom shape predictor to test and validate our model.

For running the files use

`python train_shape_predictor.py --training 'path/to/.xml/file/comtaining/mouth/only/landmarks' --model 'path/to/model.dat'`

`python evaluate_shape_predictor.py --predictor 'path/to/trained/shape_predictor' --xml 'path/to/.xml/file/comtaining/mouth/only/landmarks'`

`python parse_xml.py --input 'path/to/input/.xml/file'  --output 'path/to/output/.xml/file'`

## Feature Extractor

* Deep Convolution Neural Network model

* Dataset Used [AffectNet](http://mohammadmahoor.com/affectnet/)

* Feature Extractor/feature_extractor_train.py: trains the model and saves it in .h5 format.

* [keras-surgeon](https://github.com/BenWhetton/keras-surgeon) was used to remove the last dense layer used for classification to obtain features generated with in the model

Python command `python extract_features.py`
               `python feature_extractor_train.py`

## Yawn Detector

* Dataset Used: [YawDD](https://www.researchgate.net/publication/262255270_YawDD_A_yawning_detection_dataset)

* Yawning and Not Yawning frames were extracted from YawDD dataset and sent to our feature extractor. The extracted features were then used to train our yawn detector.

python command `python yawn_detector.py`



## Real time inference
Real time inference can be done using model_32_to_1.py file present in folder Real Time Inference.

Requirements:
1)Custom Dlib shape predictor
2)Trained Feature Extractor
3)Trained Yawn Detector

Commnad Line `python model_github.py --shape-predictor path/to/shape/predictor --input_method 1 or 0`

