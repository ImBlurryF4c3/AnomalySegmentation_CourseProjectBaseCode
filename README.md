# Real-Time-Anomaly-Segmentation [[Course Project](https://docs.google.com/document/d/1ElljsAprT2qX8RpePSQ3E00y_3oXrtN_CKYC6wqxyFQ/edit?usp=sharing)]
This repository provides a code for the Real-Time Anomaly Segmentation project of the Machine Learning Course. This code is submitted for the 27/02/2024 exam by:
* Marco Colangelo, s309798
* Federica Aamato, s310275
* Roberto Pulvirenti, s317704

## Baselines

The goal for this step is to evaluate a proposed anomaly segmentation method for urban scenes using a pre-trained ERF-Net model and a test dataset.
The evaluation involves running the model on the test dataset and analyzing its performance in detecting anomalies.
Three different methods are used for the evaluation: MSP, maxLogit, and maxEntr.
The code for this analysis can be found in eval folder and in [evalAnomaly](eval/evalAnomaly.py) file.

## Requirements:

* [**The Cityscapes dataset**](https://www.cityscapes-dataset.com/): Download the "leftImg8bit" for the RGB images and the "gtFine" for the labels. **Please note that for training you should use the "_labelTrainIds" and not the "_labelIds", you can download the [cityscapes scripts](https://github.com/mcordts/cityscapesScripts) and use the [conversor](https://github.com/mcordts/cityscapesScripts/blob/master/cityscapesscripts/preparation/createTrainIdLabelImgs.py) to generate trainIds from labelIds**
* [**Python 3.6**](https://www.python.org/): If you don't have Python3.6 in your system, I recommend installing it with [Anaconda](https://www.anaconda.com/download/#linux)
* [**PyTorch**](http://pytorch.org/): Make sure to install the Pytorch version for Python 3.6 with CUDA support (code only tested for CUDA 8.0). 
* **Additional Python packages**: numpy, matplotlib, Pillow, torchvision and visdom (optional for --visualize flag)
* **For testing the anomaly segmentation model**: Road Anomaly, Road Obstacle, and Fishyscapes dataset. All testing images are provided here [Link](https://drive.google.com/file/d/1r2eFANvSlcUjxcerjC8l6dRa0slowMpx/view).

## Anomaly Inference:
* The repo provides a pre-trained ERFNet on the cityscapes dataset that can be used to perform anomaly segmentation on test anomaly datasets.
* Anomaly Inference Command:```python evalAnomaly.py --input '/home/shyam/ViT-Adapter/segmentation/unk-dataset/RoadAnomaly21/images/*.png```. Change the dataset path ```'/home/shyam/ViT-Adapter/segmentation/unk-dataset/RoadAnomaly21/images/*.png```accordingly.
