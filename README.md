# Anime Facial Detector
This is a code base that aims to:
* Find anime faces in any given frame
* Categorize faces by character and return the "screen time" of any given character

We do this by utilizing transfer learning from these "backbones" (respectively)
* Mask_RCNN
* MobileNet V2
# Author
Apan

Special thanks to Najeeb for the inspiration and consultation!

Much of code/materials adapted from these sources:
* https://github.com/bchao1/Anime-Face-Dataset
* https://github.com/matterport/Mask_RCNN
* https://github.com/tensorflow/docs/blob/master/site/en/tutorials/images/transfer_learning.ipynb

# Getting Started
Here's the general workflow:
## 1) Create conda environment
Please download conda to install the necessary dependencies (https://docs.conda.io/en/latest/) 
```
### Please choose the environment/*yml file for your OS and hardware
Windows10 (CPU only): 
Windows10 (GPU): 
MacOS (CPU only):
MacOS (GPU):
Ubuntu (CPU only):
Ubuntu (GPU): 

# Example: Type this into your terminal / anaconda powershell
conda env create -f ./enviroment/linux-gpu.yml 
conda activate anime_face_tracker
```

## 2) Get frames via screen-recording
```
python main.py caputre --output /path/to/output/ --frames 0.2
```
## 3) Detect faces in each frame
```
python main.py detect --output /path/to/output/ --threads 1
```
## 4) Classify each face and output statistics
```
python main.py classify --output /path/to/output/ 
                        --model /path/to/output/transfer_model.h5
```
# Train models yourself!
Right now the classification model only works for the show "Quintessential Quintuplets"
This was chosen as I thought it would pose a challenge for the algorithm to train on since all the character look nearly identical (as the title would suggest).

You can continue to or re-train both the detector and classifier

## Retrain detector

## Train classifier
### 1) Annotate faces
```
# In my experience, I've found 50-200 annotated images per character
# to be sufficient
# Please open an additional terminal/powershell
# In terminal "1"

python serverTcp.py

# In terminal "2"

conda activate anime_face_tracker
python annotate.py annotate --output /path/to/output

# Please provide annotations in terminal "1"
# Feel free to ^C at any point
# Outputs are written to /path/to/output/data_out.csv

# Finally organize annotations for downstream processing
python annotate.py organize --output /path/to/output
```
### 2) Train
```
# Update model
python main.py train-classify --output /path/to/output --model /path/to/.h5

# Or completely retrain from baseline moblilenet v2
python main.py train-classify --output /path/to/output
```
