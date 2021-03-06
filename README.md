# Driver-Drowsiness-Detection-using-GRU-with-CNN-features

Typical methods used to identify drowsy drivers are physiological based such as heartbeat, pulse rate, and Electrocardiogram. Vehicle based methods include accelerator pattern, acceleration, and steering movements. Behavioral methods include yawn, Eye Closure, Eye Blinking, etc. Yawning is one of the major indicators to detect drowsiness, tiredness, and fatigue. We use yawning to detect driver drowsiness.

## Dependencies
* Python 3.6.10
* Tensorflow (Used tf 1.14, for tf 2.0 or greater some changes will be needed)
* Dlib
* OpenCV
* Sklearn 
* Matplotib
* Os
* [Keras-surgeon](https://github.com/BenWhetton/keras-surgeon)
* Numpy
* Argparse
* Imutils
* Time



## Workflow

Incoming Video Feed->Extract Mouth region from Individual Frames->Send to Feature Extractor->Concatenate 32 Frames->Send to Yawn Detector->Yawning/Not Yawning

Three different model sequentially form the algorithm.

* Custom Dlib Shape Predictor
* Feature Extractor
* Yawn detector


### Custom dlib shape predictor 
* Trained using steps provided [here](https://www.pyimagesearch.com/2019/12/16/training-a-custom-dlib-shape-predictor/)

* **Dataset Used** : [i-bug 300W](https://ibug.doc.ic.ac.uk/resources/300-W/)

* parse_xml.py : Parses the train/test XML dataset files for eyes-only landmark coordinates.

* train_shape_predictor.py : Accepts the parsed XML files to train shape predictor with dlib, saves custom shape predictor with .dat file extension.

* evaluate_shape_predictor.py : Calculates the Mean Average Error (MAE) of our custom shape predictor to test and validate our model.

_Command lines to execute code:_

`python train_shape_predictor.py --training 'path/to/.xml/file/comtaining/mouth/only/landmarks' --model 'path/to/model.dat'`

`python evaluate_shape_predictor.py --predictor 'path/to/trained/shape_predictor' --xml 'path/to/.xml/file/comtaining/mouth/only/landmarks'`

`python parse_xml.py --input 'path/to/input/.xml/file'  --output 'path/to/output/.xml/file'`

### Feature Extractor

* **Dataset Used** : [AffectNet](http://mohammadmahoor.com/affectnet/)

* Feature Extractor/feature_extractor_train.py: Convolution Neural Network trains and is saved in .h5 format.

* [keras-surgeon](https://github.com/BenWhetton/keras-surgeon) was used to remove the last dense layer used for classification to obtain features generated with in the model


_Command lines to execute code:_  `python extract_features.py`  `python feature_extractor_train.py`

### Yawn Detector

* **Dataset Used** : [YawDD](https://www.researchgate.net/publication/262255270_YawDD_A_yawning_detection_dataset)

* Yawning and Not Yawning frames were extracted from YawDD dataset and sent to feature extractor. The extracted features were then used to train yawn detector.

_Command lines to execute code:_ `python yawn_detector.py`



## Real time inference
Real time inference can be done using model_32_to_1.py file present in folder Real Time Inference.

Requirements:

* Custom Dlib shape predictor

* Trained Feature Extractor

* Trained Yawn Detector

_Command lines to execute code:_
`python model_github.py --shape-predictor 'path/to/shape/predictor' --input_method 1 or 0`

## Result

It can be seen in top right corner when the model detects a yawn.

![alt_text](https://github.com/srivastava-ayush/Driver-Drowsiness-Detection-using-GRU-with-CNN-features/blob/main/Real%20Time%20Inference/yawdd_test.gif)
