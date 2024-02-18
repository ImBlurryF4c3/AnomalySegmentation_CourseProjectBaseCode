# Real-Time-Anomaly-Segmentation [[Course Project](https://docs.google.com/document/d/1ElljsAprT2qX8RpePSQ3E00y_3oXrtN_CKYC6wqxyFQ/edit?usp=sharing)]
This repository provides a code for the Real-Time Anomaly Segmentation project of the Machine Learning Course. This code is submitted for the 27/02/2024 exam by:
* Marco Colangelo, s309798
* Federica Aamato, s310275
* Roberto Pulvirenti, s317704

## Baselines - MSP, MaxLogit and MaxEntropy

The goal for this step is to evaluate a proposed anomaly segmentation method for urban scenes using a pre-trained ERF-Net model and a test dataset. The evaluation involves running the model on the test dataset and analyzing its performance in detecting anomalies.
Three different methods are used for the evaluation: MSP, maxLogit, and maxEntr.
The code for the inference analysis can be found in eval folder and in [evalAnomaly.py](eval/evalAnomaly.py) file, the code for the mIou analysis can be found in eval folder in the [eval_iou.py](eval/eval_iou.py) file.

## Baselines - Temperature Scaling

The goal for the second step of our project is to find the optimal temperature for a neural classification model that minimizes the calibration error and the negative log-likelihood of the predictions.
The method is to use a validation dataset and an optimization algorithm to tune the temperature parameter that scales the model outputs.
The result is a more calibrated model that can output more reliable probabilities and predictions.
The code for this part is avaible in [evalAnomaly.py](eval/evalAnomaly.py) and [eval_iou.py](eval/eval_iou.py) file, in order to choose the best temperature we use the code in [temperature_scaling.py](eval/temperature_scaling.py).


## Void Classifier

For this step we provide a method for anomaly detection using a semantic segmentation network with an extra class for anomalies.
We use the Cityscapes with the void class as a source of anomaly data and train two networks, ENet and BiSeNet, with this method.
The code for ENet and BiSeNet can be found respectivelly into [ENet-Github](https://github.com/federicamato00/PyTorch-ENet-Training.git) and [BiSeNet-Github](https://github.com/federicamato00/BiSeNet-Training.git) repositories.
For our evaluation we used the code into [eval_voidClassifier.py](eval/eval_voidClassifier.py) and [eval_iou.py](eval/eval_iou.py) file.


## Project Extention - Effect of Training Loss function

We explore different loss functions that are designed for anomaly detection, such as Jaccard Loss and Logit Normalization Loss, investigating how combining these loss functions with other common ones, such as Focal Loss and Cross-Entropy Loss, affects the model’s performance in segmenting and identifying anomalies in road scenes.

For this step, we modify the [main.py](train/main.py) file in order to implement this new loss functions.

## Command used

All the command used in the evaluation part can be found into [googleColab] (Project_Evaluation.ipynb) file, for the training part the files can be found into [training_file](training_file) folder.

## Project Presentation

Our presentation related to this project can be found at [Project Presentation] () link.



