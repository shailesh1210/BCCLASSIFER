# Breast Cancer Classifier with CNN

## Packages and version used 
- tensorflow 2.3.1
- python 3.7.9
- wxPython (GUI)
- tqdm 

## REPO 
https://github.com/shailesh1210/BCCLASSIFER.git

## Dataset description
Download Invasive Ductal Carcinoma (IDC) dataset from Kaggle:
https://www.kaggle.com/paultimothymooney/breast-histopathology-images

The dataset size is ~5GB. It needs to be downloaded from the link above.
`bcclassifer` package has `/resources/raw_data` folder. Please copy contents of downloaded dataset (of IDC_regular_ps50_idx5) into `raw_data` folder.

## Running training script:
Training script is `train.py`. It will create `training`, `validation`, `test` folders in `resources`.
To run the training script from root package:
```
python3 core/train.py
```

## Running prediction script:
Prediction script is `predict.py`. It will make predictions on the test set.
To run the prediction script from root:
```
python3 core/predict.py
```

## To Run the UI:
```
python3 ui_main.py
```
