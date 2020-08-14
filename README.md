# Predicting decisions of Dutch Rent Tribunal cases
A repository for the work on predicting Dutch Rent Tribunal cases.
Made by Willem Datema

Clone this repository via the command line by running

```
git clone https://github.com/Willeem/predicting_rent_tribunal.git 
```

## Software needed
- Python3.6

After installing Python 3.6, create a new virtual environment by running
```
python3 -m venv predicting
```
Activate it by running
``` 
source predicting/bin/activate
``` 
Then, the neccessary packages can be installed by running

```
pip install -r requirements.txt
```

All cases can be downloaded from https://drive.google.com/file/d/1X9LPCuMRnUwqKOlKaUVLaiGNTRn2v0eG/view and https://drive.google.com/file/d/1TNI4v98hHSTKh8e6wJfH5zFUGoRCR3uT/view 
Make sure to unzip them and put the folders in the /data/ folder. 

Then, run

```
python3 categorize_and_clean_data.py
```
to be able to work with the files.

Then obtain the pretrained embeddings from https://drive.google.com/file/d/1MXrGCWV5ejUa2tI1kU2pV5uT4-BFjW3e/view?usp=sharing, https://drive.google.com/file/d/1aMrDVgwghRmCmNdcTB32WD2XjKMeCwK_/view?usp=sharing, https://drive.google.com/file/d/1t2bUsJUAoVKw6biQel3D-IF07vWwrz9D/view?usp=sharing and paste them in the /embeddings/ directory.

To recreate the classification experiment on the test set run:
```
python3 classifier.py --topic all --phase test --baseline True
```
Add the --dates True flag for the Future experiment.

To recreate the regression experiment on the test set run:
```
python3 regression.py --topic all --phase test --baseline True
```
Add the --dates True flag for the Future experiment.