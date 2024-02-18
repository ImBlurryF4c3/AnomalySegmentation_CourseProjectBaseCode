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
The code for this part is avaible in [evalAnomaly.py](eval/evalAnomaly.py) and [[eval_iou.py](eval/eval_iou.py) file, in order to choose the best temperature we use the code in [temperature_scaling.py](eval/temperature_scaling.py).

For this step we use 

## Void Classifier


## Project Extention - Effect of Training Loss function






