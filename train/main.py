# Main code for training ERFNet model in Cityscapes dataset
# Sept 2017
# Eduardo Romera
#######################

import os
import random
import time
import numpy as np
import torch
import math

from PIL import Image, ImageOps
from argparse import ArgumentParser

from torch.optim import SGD, Adam, lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, CenterCrop, Normalize, Resize, Pad
from torchvision.transforms import ToTensor, ToPILImage
import torch.nn.functional as F
from dataset import VOC12,cityscapes
from transform import Relabel, ToLabel, Colorize
from visualize import Dashboard

import importlib
from iouEval import iouEval, getColorEntry

import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from shutil import copyfile

NUM_CHANNELS = 3
NUM_CLASSES = 20 #pascal=22, cityscapes=20

color_transform = Colorize(NUM_CLASSES)
image_transform = ToPILImage()

class FocalLoss(torch.nn.Module):
    def __init__(self, gamma=2.0, alpha=1, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        # if isinstance(alpha,(float,int,torch.long)): self.alpha = torch.Tensor([alpha,1-alpha])
        # if isinstance(alpha,list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        # if self.alpha is not None:
        #     if self.alpha.type()!=input.data.type():
        #         self.alpha = self.alpha.type_as(input.data)
        #     at = self.alpha.gather(0,target.data.view(-1))
        #     logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class EnhancedIsotropyMaximizationLoss(torch.nn.Module):
    def __init__(self, model, weight=None):
        super(EnhancedIsotropyMaximizationLoss, self).__init__()
        self.weight = weight
        # Crea un input di esempio
        input = torch.randn(1, 3, 224, 224)
        # Passa l'input attraverso la rete
        output = model(input.cuda())
        # Stampa la dimensione dell'output
        print(output.size())
        self.feature_dim = output.size(1)
        # Inizializzare i prototipi con una distribuzione normale
        self.prototypes = torch.nn.Parameter(torch.randn(NUM_CLASSES, self.feature_dim).cuda())
        # Inizializzare la scala della distanza a uno
        self.distance_scale = torch.nn.Parameter(torch.tensor(1.0).cuda())


    def forward(self, output, target):
       # Compute the Enhanced Isotropy Maximization Loss
       
        # Normalizzare le caratteristiche in uscita dalla rete
        output = F.normalize(output, dim=1)
        # Calcolare le distanze tra le caratteristiche e i prototipi
        distances = torch.abs(self.distance_scale) * torch.cdist(output, self.prototypes, p=2.0)
        # Calcolare i logit come il negativo delle distanze
        logits = -distances
        # Calcolare le probabilità usando la funzione softmax sui logit
        probabilities = F.softmax(logits, dim=1)
        # Selezionare le probabilità corrispondenti alle etichette target
        probabilities_at_targets = probabilities[torch.arange(output.size(0)).cuda(), target.cuda()]
        # Calcolare la perdita come il negativo del logaritmo delle probabilità
        loss = -torch.log(probabilities_at_targets).mean()
        # Restituire la perdita
        return loss
        

class LogitNormalizationLoss(torch.nn.Module):
    def __init__(self, weight=None, tau=1.0):
        super(LogitNormalizationLoss, self).__init__()
        self.weight = weight
        self.tau = tau # il parametro di temperatura

    def forward(self, output, target):
        # normalizza il vettore di logit per avere una norma costante
        output_norm = output / (output.norm(dim=1, keepdim=True) + 1e-7)
        # applica la funzione softmax con la temperatura
        output_prob = torch.nn.functional.softmax(output_norm / self.tau, dim=1)
        # Sposta il tensore weight sulla GPU, se disponibile
        if self.weight is not None:
            self.weight = self.weight.cuda()
        # calcola la cross-entropy loss con i logit normalizzati
        loss = torch.nn.functional.cross_entropy(output_prob, target.cuda(), weight=self.weight)
        return loss

# class JaccardLoss2d(torch.nn.Module):

#     def __init__(self, weight=None):
#         super(JaccardLoss2d, self).__init__()
#         self.weight = weight

#     def forward(self, outputs, targets):
#         targets = torch.unsqueeze(targets, dim=1)
#         targets = targets.expand(-1, 20, -1, -1)  # Add channel dimension to targets

#         # weighting the data
#         if self.weight is not None:
#           self.weight = self.weight.cuda()
#           self.weight = self.weight.view(1, 20, 1, 1)
#           outputs = outputs * self.weight

#         # Flatten predictions and targets
#         outputs_flat = outputs.reshape(outputs.size()[0], -1)
#         targets_flat = targets.reshape(targets.size()[0], -1)

#         # Intersection and union
#         intersection = torch.sum(torch.min(outputs_flat, targets_flat), dim=1, keepdim=True)
#         union = torch.sum(torch.max(outputs_flat, targets_flat), dim=1, keepdim=True)

#         jaccard = (intersection + 1e-8) / (union + 1e-8)

#         # Average the Jaccard indices along the batches
#         loss = 1 - torch.mean(jaccard)

#         return loss

class JDTLoss(_Loss):
    def __init__(self,
                 mIoUD=1.0,
                 mIoUI=0.0,
                 mIoUC=0.0,
                 alpha=1.0,
                 beta=1.0,
                 gamma=1.0,
                 smooth=1.0,
                 threshold=0.01,
                 log_loss=False,
                 ignore_index=None,
                 class_weights=None,
                 active_classes_mode_hard="PRESENT",
                 active_classes_mode_soft="ALL"):
        """
        Arguments:
            mIoUD (float): The weight of the loss to optimize mIoUD.
            mIoUI (float): The weight of the loss to optimize mIoUI.
            mIoUC (float): The weight of the loss to optimize mIoUC.
            alpha (float): The coefficient of false positives in the Tversky loss.
            beta (float): The coefficient of false negatives in the Tversky loss.
            gamma (float): When `gamma` > 1, the loss focuses more on
                less accurate predictions that have been misclassified.
            smooth (float): A floating number to avoid `NaN` error.
            threshold (float): The threshold to select active classes.
            log_loss (bool): Compute the log loss or not.
            ignore_index (int | None): The class index to be ignored.
            class_weights (list[float] | None): The weight of each class.
                If it is `list[float]`, its size should be equal to the number of classes.
            active_classes_mode_hard (str): The mode to compute
                active classes when training with hard labels.
            active_classes_mode_soft (str): The mode to compute
                active classes when training with hard labels.

        Comments:
            Jaccard: `alpha`  = 1.0, `beta`  = 1.0
            Dice:    `alpha`  = 0.5, `beta`  = 0.5
            Tversky: `alpha` >= 0.0, `beta` >= 0.0
        """
        super().__init__()

        assert mIoUD >= 0 and mIoUI >= 0 and mIoUC >= 0 and \
               alpha >= 0 and beta >= 0 and gamma >= 1 and \
               smooth >= 0 and threshold >= 0
        assert ignore_index == None or isinstance(ignore_index, int)
        assert class_weights == None or all((isinstance(w, float)) for w in class_weights)
        assert active_classes_mode_hard in ["ALL", "PRESENT"]
        assert active_classes_mode_soft in ["ALL", "PRESENT"]

        self.mIoUD = mIoUD
        self.mIoUI = mIoUI
        self.mIoUC = mIoUC
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth
        self.threshold = threshold
        self.log_loss = log_loss
        self.ignore_index = ignore_index
        if class_weights == None:
            self.class_weights = class_weights
        else:
            self.class_weights = torch.tensor(class_weights)
        self.active_classes_mode_hard = active_classes_mode_hard
        self.active_classes_mode_soft = active_classes_mode_soft


    def forward(self, logits, label, keep_mask=None):
        """
        Arguments:
            logits (torch.Tensor): Its shape should be (B, C, D1, D2, ...).
            label (torch.Tensor):
                If it is hard label, its shape should be (B, D1, D2, ...).
                If it is soft label, its shape should be (B, C, D1, D2, ...).
            keep_mask (torch.Tensor | None):
                If it is `torch.Tensor`,
                    its shape should be (B, D1, D2, ...) and
                    its dtype should be `torch.bool`.
        """
        batch_size, num_classes = logits.shape[:2]
        hard_label = label.dtype == torch.long

        logits = logits.view(batch_size, num_classes, -1)
        prob = logits.log_softmax(dim=1).exp()

        if keep_mask != None:
            assert keep_mask.dtype == torch.bool
            keep_mask = keep_mask.view(batch_size, -1)
            keep_mask = keep_mask.unsqueeze(1).expand_as(prob)
        elif self.ignore_index != None and hard_label:
            keep_mask = label != self.ignore_index
            keep_mask = keep_mask.view(batch_size, -1)
            keep_mask = keep_mask.unsqueeze(1).expand_as(prob)

        if hard_label:
            label = torch.clamp(label, 0, num_classes - 1).view(batch_size, -1)
            label = F.one_hot(label, num_classes=num_classes).permute(0, 2, 1).float()
            active_classes_mode = self.active_classes_mode_hard
        else:
            label = label.view(batch_size, num_classes, -1)
            active_classes_mode = self.active_classes_mode_soft

        loss = self.forward_loss(prob, label, keep_mask, active_classes_mode)

        return loss


    def forward_loss(self, prob, label, keep_mask, active_classes_mode):
        if keep_mask != None:
            prob = prob * keep_mask
            label = label * keep_mask

        cardinality = torch.sum(prob + label, dim=2)
        difference = torch.sum(torch.abs(prob - label), dim=2)
        tp = (cardinality - difference) / 2
        fp = torch.sum(prob, dim=2) - tp
        fn = torch.sum(label, dim=2) - tp

        loss = 0
        batch_size, num_classes = prob.shape[:2]
        if self.mIoUD > 0:
            active_classes = self.compute_active_classes(label, active_classes_mode, num_classes, (0, 2))
            loss_mIoUD = self.forward_loss_mIoUD(tp, fp, fn, active_classes)
            loss += self.mIoUD * loss_mIoUD

        if self.mIoUI > 0 or self.mIoUC > 0:
            active_classes = self.compute_active_classes(label, active_classes_mode, (batch_size, num_classes), (2, ))
            loss_mIoUI, loss_mIoUC = self.forward_loss_mIoUIC(tp, fp, fn, active_classes)
            loss += self.mIoUI * loss_mIoUI + self.mIoUC * loss_mIoUC

        return loss


    def compute_active_classes(self, label, active_classes_mode, shape, dim):
        if active_classes_mode == "ALL":
            mask = torch.ones(shape, dtype=torch.bool)
        elif active_classes_mode == "PRESENT":
            mask = torch.amax(label, dim) > self.threshold

        active_classes = torch.zeros(shape, dtype=torch.bool, device=label.device)
        active_classes[mask] = 1

        return active_classes


    def forward_loss_mIoUD(self, tp, fp, fn, active_classes):
        if torch.sum(active_classes) < 1:
            return 0. * torch.sum(tp)

        tp = torch.sum(tp, dim=0)
        fp = torch.sum(fp, dim=0)
        fn = torch.sum(fn, dim=0)
        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if self.log_loss:
            loss_mIoUD = -torch.log(tversky)
        else:
            loss_mIoUD = 1.0 - tversky

        if self.gamma > 1:
            loss_mIoUD **= self.gamma

        if self.class_weights != None:
            loss_mIoUD *= self.class_weights

        loss_mIoUD = loss_mIoUD[active_classes]
        loss_mIoUD = torch.mean(loss_mIoUD)

        return loss_mIoUD


    def forward_loss_mIoUIC(self, tp, fp, fn, active_classes):
        if torch.sum(active_classes) < 1:
            return 0. * torch.sum(tp), 0. * torch.sum(tp)

        tversky = (tp + self.smooth) / (tp + self.alpha * fp + self.beta * fn + self.smooth)

        if self.log_loss:
            loss_matrix = -torch.log(tversky)
        else:
            loss_matrix = 1.0 - tversky

        if self.gamma > 1:
            loss_matrix **= self.gamma

        if self.class_weights != None:
            class_weights = self.class_weights.unsqueeze(0).expand_as(loss_matrix)
            loss_matrix *= class_weights

        loss_matrix *= active_classes
        loss_mIoUI = self.reduce(loss_matrix, active_classes, 1)
        loss_mIoUC = self.reduce(loss_matrix, active_classes, 0)

        return loss_mIoUI, loss_mIoUC


    def reduce(self, loss_matrix, active_classes, dim):
        active_sum = torch.sum(active_classes, dim)
        active_dim = active_sum > 0
        loss = torch.sum(loss_matrix, dim)
        loss = loss[active_dim] / active_sum[active_dim]
        loss = torch.mean(loss)

        return loss



#Augmentations - different function implemented to perform random augments on both image and target

class MyCoTransform(object):
    def __init__(self, enc, augment=True, height=512):
        self.enc=enc
        self.augment = augment
        self.height = height
        pass
    def __call__(self, input, target):
        # do something to both images
        input =  Resize(self.height, Image.BILINEAR)(input)
        target = Resize(self.height, Image.NEAREST)(target)

        if(self.augment):
            # Random hflip
            hflip = random.random()
            if (hflip < 0.5):
                input = input.transpose(Image.FLIP_LEFT_RIGHT)
                target = target.transpose(Image.FLIP_LEFT_RIGHT)
            
            #Random translation 0-2 pixels (fill rest with padding
            transX = random.randint(-2, 2) 
            transY = random.randint(-2, 2)

            input = ImageOps.expand(input, border=(transX,transY,0,0), fill=0)
            target = ImageOps.expand(target, border=(transX,transY,0,0), fill=255) #pad label filling with 255
            input = input.crop((0, 0, input.size[0]-transX, input.size[1]-transY))
            target = target.crop((0, 0, target.size[0]-transX, target.size[1]-transY))   

        input = ToTensor()(input)
        if (self.enc):
            target = Resize(int(self.height/8), Image.NEAREST)(target)
        target = ToLabel()(target)
        target = Relabel(255, 19)(target)

        return input, target


class CrossEntropyLoss2d(torch.nn.Module):
    def __init__(self, weight=None):
        super().__init__()

        self.loss = torch.nn.NLLLoss2d(weight)

    def forward(self, outputs, targets):
        # Sposta il peso della perdita sulla stessa GPU del tensore di output, se disponibile
        if outputs.is_cuda:
            self.loss.weight = self.loss.weight.to(outputs.device)
        return self.loss(torch.nn.functional.log_softmax(outputs, dim=1), targets)


def train(args, model, enc=False):
    best_acc = 0

    #TODO: calculate weights by processing dataset histogram (now its being set by hand from the torch values)
    #create a loder to run all images and calculate histogram of labels, then create weight array using class balancing
    
    def calculate_weights(dataset):
        label_counts = torch.zeros(NUM_CLASSES)
        for data in dataset:
            _, labels = data
            label_counts += torch.bincount(labels.flatten(), minlength=NUM_CLASSES)

        total_samples = sum(label_counts)
        weights = 1 / (label_counts / total_samples)

        return weights
    
    
        
    weight = torch.ones(NUM_CLASSES)
    if (enc):
        weight[0] = 2.3653597831726	
        weight[1] = 4.4237880706787	
        weight[2] = 2.9691488742828	
        weight[3] = 5.3442072868347	
        weight[4] = 5.2983593940735	
        weight[5] = 5.2275490760803	
        weight[6] = 5.4394111633301	
        weight[7] = 5.3659925460815	
        weight[8] = 3.4170460700989	
        weight[9] = 5.2414722442627	
        weight[10] = 4.7376127243042	
        weight[11] = 5.2286224365234	
        weight[12] = 5.455126285553	
        weight[13] = 4.3019247055054	
        weight[14] = 5.4264230728149	
        weight[15] = 5.4331531524658	
        weight[16] = 5.433765411377	
        weight[17] = 5.4631009101868	
        weight[18] = 5.3947434425354
    else:
        weight[0] = 2.8149201869965	
        weight[1] = 6.9850029945374	
        weight[2] = 3.7890393733978	
        weight[3] = 9.9428062438965	
        weight[4] = 9.7702074050903	
        weight[5] = 9.5110931396484	
        weight[6] = 10.311357498169	
        weight[7] = 10.026463508606	
        weight[8] = 4.6323022842407	
        weight[9] = 9.5608062744141	
        weight[10] = 7.8698215484619	
        weight[11] = 9.5168733596802	
        weight[12] = 10.373730659485	
        weight[13] = 6.6616044044495	
        weight[14] = 10.260489463806	
        weight[15] = 10.287888526917	
        weight[16] = 10.289801597595	
        weight[17] = 10.405355453491	
        weight[18] = 10.138095855713	

    weight[19] = 0
    
    assert os.path.exists(args.datadir), "Error: datadir (dataset directory) could not be loaded"
    print("transform")
    co_transform = MyCoTransform(enc, augment=True, height=args.height)#1024)
    co_transform_val = MyCoTransform(enc, augment=False, height=args.height)#1024)
    dataset_train = cityscapes(args.datadir, co_transform, 'train')
    dataset_val = cityscapes(args.datadir, co_transform_val, 'val')
    print("calcolo weights")
    weights = calculate_weights(dataset_train)
    weight = torch.tensor(weights)
    print("calcolo loader")
    loader = DataLoader(dataset_train, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=True)
    loader_val = DataLoader(dataset_val, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False)
    print("fine loader_val")
    if args.cuda:
        weight = weight.cuda()
    if args.lossfunction == "enhanced_isotropy":
        isotropy_loss = EnhancedIsotropyMaximizationLoss(model,weight)
    elif args.lossfunction == "logit_norm":
        normalization_loss = LogitNormalizationLoss(weight)
    elif args.lossfunction == "jaccard_loss":
        j_loss = JDTLoss(class_weights=weight)
    else:
      criterion = CrossEntropyLoss2d(weight)
    
    if args.focal_loss == True:
        focal_loss = FocalLoss()
    else:
      criterion = CrossEntropyLoss2d(weight)
    
    # criterion = CrossEntropyLoss2d(weight)
    # print(type(criterion))

    savedir = f'../save/{args.savedir}'

    if (enc):
        automated_log_path = savedir + "/automated_log_encoder.txt"
        modeltxtpath = savedir + "/model_encoder.txt"
    else:
        automated_log_path = savedir + "/automated_log.txt"
        modeltxtpath = savedir + "/model.txt"    

    if (not os.path.exists(automated_log_path)):    #dont add first line if it exists 
        with open(automated_log_path, "a") as myfile:
            myfile.write("Epoch\t\tTrain-loss\t\tTest-loss\t\tTrain-IoU\t\tTest-IoU\t\tlearningRate")

    with open(modeltxtpath, "w") as myfile:
        myfile.write(str(model))


    #TODO: reduce memory in first gpu: https://discuss.pytorch.org/t/multi-gpu-training-memory-usage-in-balance/4163/4        #https://github.com/pytorch/pytorch/issues/1893
    print("inizio optimizer")
    #optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=2e-4)     ## scheduler 1
    optimizer = Adam(model.parameters(), 5e-4, (0.9, 0.999),  eps=1e-08, weight_decay=1e-4)      ## scheduler 2
    print("fine optimizer")
    start_epoch = 1
    if args.resume:
        #Must load weights, optimizer, epoch and best value. 
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'

        assert os.path.exists(filenameCheckpoint), "Error: resume option was used but checkpoint was not found in folder"
        checkpoint = torch.load(filenameCheckpoint)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        best_acc = checkpoint['best_acc']
        print("=> Loaded checkpoint at epoch {})".format(checkpoint['epoch']))

    #scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5) # set up scheduler     ## scheduler 1
    lambda1 = lambda epoch: pow((1-((epoch-1)/args.num_epochs)),0.9)  ## scheduler 2
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda1)                             ## scheduler 2

    if args.visualize and args.steps_plot > 0:
        board = Dashboard(args.port)

    for epoch in range(start_epoch, args.num_epochs+1):
        print("----- TRAINING - EPOCH", epoch, "-----")

        scheduler.step(epoch)    ## scheduler 2

        epoch_loss = []
        time_train = []
     
        doIouTrain = args.iouTrain   
        doIouVal =  args.iouVal      

        if (doIouTrain):
            iouEvalTrain = iouEval(NUM_CLASSES)

        usedLr = 0
        for param_group in optimizer.param_groups:
            print("LEARNING RATE: ", param_group['lr'])
            usedLr = float(param_group['lr'])

        model.train()
        for step, (images, labels) in enumerate(loader):

            start_time = time.time()
            #print (labels.size())
            #print (np.unique(labels.numpy()))
            #print("labels: ", np.unique(labels[0].numpy()))
            #labels = torch.ones(4, 1, 512, 1024).long()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()
                #model = torch.nn.DataParallel(model).cuda()
                #model = model.to('cuda:0')


            inputs = Variable(images)
            targets = Variable(labels)
            outputs = model(inputs, only_encode=enc)
            
            

            #print("targets", np.unique(targets[:, 0].cpu().data.numpy()))

            optimizer.zero_grad()
            if args.lossfunction == "cross_entropy": ########codice di default
                loss = criterion(outputs, targets[:, 0])
                loss.backward()
                optimizer.step()
                epoch_loss.append(loss.item())

            elif args.lossfunction == "logit_norm":
                # Implement Focal Loss calculation using outputs and targets
                if args.onlyone == False:
                    if args.focal_loss == True:
                        loss = loss = focal_loss(outputs,targets[:, 0])
                    else:
                        loss = criterion(outputs, targets[:, 0])
                    logit_norm_loss = normalization_loss(outputs, targets[:, 0])
                    logit_norm_loss.backward(retain_graph=True)
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(loss.item())
                    epoch_loss.append(logit_norm_loss.item())
                else:
                    logit_norm_loss = normalization_loss(outputs, targets[:, 0])
                    logit_norm_loss.backward(retain_graph=True)
                    optimizer.step()
                    epoch_loss.append(logit_norm_loss.item())


            elif args.lossfunction == "enhanced_isotropy":
                eim_loss = isotropy_loss(outputs, targets[:, 0])
                if args.onlyone == False:
                    # Implement Focal Loss calculation using outputs and targets
                    if args.focal_loss == True:
                        loss = loss = focal_loss(outputs,targets[:, 0])
                    else:
                        loss = criterion(outputs, targets[:, 0])
                    eim_loss.backward()
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(eim_loss.item())
                    epoch_loss.append(loss.item())
                else:
                    eim_loss.backward()
                    optimizer.step()
                    epoch_loss.append(eim_loss.item())

            elif args.lossfunction == "jaccard_loss":
                jacc_loss = j_loss(outputs, targets[:, 0])
                if args.onlyone == False:
                    # Implement Focal Loss calculation using outputs and targets
                    if args.focal_loss == True:
                        loss = loss = focal_loss(outputs,targets[:, 0])
                    else:
                        loss = criterion(outputs, targets[:, 0])
                    jacc_loss.backward()
                    loss.backward()
                    optimizer.step()
                    epoch_loss.append(jacc_loss.item())
                    epoch_loss.append(loss.item())
                else:
                    jacc_loss.backward()
                    optimizer.step()
                    epoch_loss.append(jacc_loss.item())
                    



            time_train.append(time.time() - start_time)

            if (doIouTrain):
                #start_time_iou = time.time()
                iouEvalTrain.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)      

            #print(outputs.size())
            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                #image[0] = image[0] * .229 + .485
                #image[1] = image[1] * .224 + .456
                #image[2] = image[2] * .225 + .406
                #print("output", np.unique(outputs[0].cpu().max(0)[1].data.numpy()))
                board.image(image, f'input (epoch: {epoch}, step: {step})')
                if isinstance(outputs, list):   #merge gpu tensors
                    board.image(color_transform(outputs[0][0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'output (epoch: {epoch}, step: {step})')
                else:
                    board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'output (epoch: {epoch}, step: {step})')
                board.image(color_transform(targets[0].cpu().data),
                    f'target (epoch: {epoch}, step: {step})')
                print ("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss) / len(epoch_loss)
                #print(sum(epoch_loss))
                #print(len(epoch_loss))
                print(f'loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_train) / len(time_train) / args.batch_size))

            
        average_epoch_loss_train = sum(epoch_loss) / len(epoch_loss)
        
        iouTrain = 0
        if (doIouTrain):
            iouTrain, iou_classes = iouEvalTrain.getIoU()
            iouStr = getColorEntry(iouTrain)+'{:0.2f}'.format(iouTrain*100) + '\033[0m'
            print ("EPOCH IoU on TRAIN set: ", iouStr, "%")  

        #Validate on 500 val images after each epoch of training
            
            ################# anche qui si fa attenzione al loss_function ###############
        print("----- VALIDATING - EPOCH", epoch, "-----")
        model.eval()
        epoch_loss_val = []
        time_val = []

        if (doIouVal):
            iouEvalVal = iouEval(NUM_CLASSES)

        for step, (images, labels) in enumerate(loader_val):
            start_time = time.time()
            if args.cuda:
                images = images.cuda()
                labels = labels.cuda()

            inputs = Variable(images, volatile=True)    #volatile flag makes it free backward or outputs for eval
            targets = Variable(labels, volatile=True)
            outputs = model(inputs, only_encode=enc) 


            if args.lossfunction == "cross_entropy": ########codice di default
                loss = criterion(outputs, targets[:, 0])
                loss.backward()
                epoch_loss_val.append(loss.item())

            elif args.lossfunction == "logit_norm":
                # Implement Focal Loss calculation using outputs and targets
                if args.onlyone == False:
                    if args.focal_loss == True:
                        loss = focal_loss(outputs,targets[:, 0])
                    else:
                        loss = criterion(outputs, targets[:, 0])
                    logit_norm_loss = normalization_loss(outputs, targets[:, 0])
                    logit_norm_loss.backward(retain_graph=True)
                    loss.backward()
                    epoch_loss_val.append(loss.item())
                    epoch_loss_val.append(logit_norm_loss.item())
                else:
                    logit_norm_loss = normalization_loss(outputs, targets[:, 0])
                    logit_norm_loss.backward(retain_graph=True)
                    epoch_loss_val.append(logit_norm_loss.item())


            elif args.lossfunction == "enhanced_isotropy":
                eim_loss = isotropy_loss(outputs, targets[:, 0])
                if args.onlyone == False:
                    # Implement Focal Loss calculation using outputs and targets
                    if args.focal_loss == True:
                        loss = loss = focal_loss(outputs,targets[:, 0])
                    else:
                        loss = criterion(outputs, targets[:, 0])
                    eim_loss.backward()
                    loss.backward()
                    epoch_loss_val.append(eim_loss.item())
                    epoch_loss_val.append(loss.item())
                else:
                    eim_loss.backward()
                    epoch_loss_val.append(eim_loss.item())
            
            elif args.lossfunction == "jaccard_loss":
                jacc_loss = j_loss(outputs, targets[:, 0])
                if args.onlyone == False:
                    # Implement Focal Loss calculation using outputs and targets
                    if args.focal_loss == True:
                        loss = loss = focal_loss(outputs,targets[:, 0])
                    else:
                        loss = criterion(outputs, targets[:, 0])
                    jacc_loss.backward()
                    loss.backward()
                    epoch_loss_val.append(jacc_loss.item())
                    epoch_loss_val.append(loss.item())
                else:
                    jacc_loss.backward()
                    epoch_loss_val.append(jacc_loss.item())



            
            time_val.append(time.time() - start_time)


            #Add batch to calculate TP, FP and FN for iou estimation
            if (doIouVal):
                #start_time_iou = time.time()
                iouEvalVal.addBatch(outputs.max(1)[1].unsqueeze(1).data, targets.data)
                #print ("Time to add confusion matrix: ", time.time() - start_time_iou)

            if args.visualize and args.steps_plot > 0 and step % args.steps_plot == 0:
                start_time_plot = time.time()
                image = inputs[0].cpu().data
                board.image(image, f'VAL input (epoch: {epoch}, step: {step})')
                if isinstance(outputs, list):   #merge gpu tensors
                    board.image(color_transform(outputs[0][0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'VAL output (epoch: {epoch}, step: {step})')
                else:
                    board.image(color_transform(outputs[0].cpu().max(0)[1].data.unsqueeze(0)),
                    f'VAL output (epoch: {epoch}, step: {step})')
                board.image(color_transform(targets[0].cpu().data),
                    f'VAL target (epoch: {epoch}, step: {step})')
                print ("Time to paint images: ", time.time() - start_time_plot)
            if args.steps_loss > 0 and step % args.steps_loss == 0:
                average = sum(epoch_loss_val) / len(epoch_loss_val)
                print(sum(epoch_loss_val))
                print(len(epoch_loss_val))
                print(f'VAL loss: {average:0.4} (epoch: {epoch}, step: {step})', 
                        "// Avg time/img: %.4f s" % (sum(time_val) / len(time_val) / args.batch_size))
                       

        average_epoch_loss_val = sum(epoch_loss_val) / len(epoch_loss_val)
        #scheduler.step(average_epoch_loss_val, epoch)  ## scheduler 1   # update lr if needed

        iouVal = 0
        if (doIouVal):
            iouVal, iou_classes = iouEvalVal.getIoU()
            iouStr = getColorEntry(iouVal)+'{:0.2f}'.format(iouVal*100) + '\033[0m'
            print ("EPOCH IoU on VAL set: ", iouStr, "%") 
           

        # remember best valIoU and save checkpoint
        if iouVal == 0:
            current_acc = -average_epoch_loss_val
        else:
            current_acc = iouVal 
        is_best = current_acc > best_acc
        best_acc = max(current_acc, best_acc)
        if enc:
            filenameCheckpoint = savedir + '/checkpoint_enc.pth.tar'
            filenameBest = savedir + '/model_best_enc.pth.tar'    
        else:
            filenameCheckpoint = savedir + '/checkpoint.pth.tar'
            filenameBest = savedir + '/model_best.pth.tar'
        save_checkpoint({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'best_acc': best_acc,
            'optimizer' : optimizer.state_dict(),
        }, is_best, filenameCheckpoint, filenameBest)

        #SAVE MODEL AFTER EPOCH
        if (enc):
            filename = f'{savedir}/model_encoder-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_encoder_best.pth'
        else:
            filename = f'{savedir}/model-{epoch:03}.pth'
            filenamebest = f'{savedir}/model_best.pth'
        if args.epochs_save > 0 and step > 0 and step % args.epochs_save == 0:
            torch.save(model.state_dict(), filename)
            print(f'save: {filename} (epoch: {epoch})')
        if (is_best):
            torch.save(model.state_dict(), filenamebest)
            print(f'save: {filenamebest} (epoch: {epoch})')
            if (not enc):
                with open(savedir + "/best.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))   
            else:
                with open(savedir + "/best_encoder.txt", "w") as myfile:
                    myfile.write("Best epoch is %d, with Val-IoU= %.4f" % (epoch, iouVal))           

        #SAVE TO FILE A ROW WITH THE EPOCH RESULT (train loss, val loss, train IoU, val IoU)
        #Epoch		Train-loss		Test-loss	Train-IoU	Test-IoU		learningRate
        with open(automated_log_path, "a") as myfile:
            myfile.write("\n%d\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.4f\t\t%.8f" % (epoch, average_epoch_loss_train, average_epoch_loss_val, iouTrain, iouVal, usedLr ))
    
    return(model)   #return model (convenience for encoder-decoder training)

def save_checkpoint(state, is_best, filenameCheckpoint, filenameBest):
    torch.save(state, filenameCheckpoint)
    if is_best:
        print ("Saving model as best")
        torch.save(state, filenameBest)


def main(args):
    savedir = f'../save/{args.savedir}'

    if not os.path.exists(savedir):
        os.makedirs(savedir)

    with open(savedir + '/opts.txt', "w") as myfile:
        myfile.write(str(args))

    #Load Model
    assert os.path.exists(args.model + ".py"), "Error: model definition not found"
    model_file = importlib.import_module(args.model)
    model = model_file.Net(NUM_CLASSES)
    copyfile(args.model + ".py", savedir + '/' + args.model + ".py")
    
    if args.cuda:
        model = torch.nn.DataParallel(model).cuda()
    
    if args.state:
        #if args.state is provided then load this state for training
        #Note: this only loads initialized weights. If you want to resume a training use "--resume" option!!
        """
        try:
            model.load_state_dict(torch.load(args.state))
        except AssertionError:
            model.load_state_dict(torch.load(args.state,
                map_location=lambda storage, loc: storage))
        #When model is saved as DataParallel it adds a model. to each key. To remove:
        #state_dict = {k.partition('model.')[2]: v for k,v in state_dict}
        #https://discuss.pytorch.org/t/prefix-parameter-names-in-saved-model-if-trained-by-multi-gpu/494
        """
        def load_my_state_dict(model, state_dict):  #custom function to load model when not all dict keys are there
            own_state = model.state_dict()
            for name, param in state_dict.items():
                if name not in own_state:
                     continue
                own_state[name].copy_(param)
            return model

        #print(torch.load(args.state))
        model = load_my_state_dict(model, torch.load(args.state))

    """
    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            #m.weight.data.normal_(0.0, 0.02)
            n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            m.weight.data.normal_(0, math.sqrt(2. / n))
        elif classname.find('BatchNorm') != -1:
            #m.weight.data.normal_(1.0, 0.02)
            m.weight.data.fill_(1)
            m.bias.data.fill_(0)

    #TO ACCESS MODEL IN DataParallel: next(model.children())
    #next(model.children()).decoder.apply(weights_init)
    #Reinitialize weights for decoder
    
    next(model.children()).decoder.layers.apply(weights_init)
    next(model.children()).decoder.output_conv.apply(weights_init)

    #print(model.state_dict())
    f = open('weights5.txt', 'w')
    f.write(str(model.state_dict()))
    f.close()
    """

    #train(args, model)
    if (not args.decoder):
        print("========== ENCODER TRAINING ===========")
        model = train(args, model, True) #Train encoder
    #CAREFUL: for some reason, after training encoder alone, the decoder gets weights=0. 
    #We must reinit decoder weights or reload network passing only encoder in order to train decoder
    print("========== DECODER TRAINING ===========")
    if (not args.state):
        if args.pretrainedEncoder:
            print("Loading encoder pretrained in imagenet")
            from erfnet_imagenet import ERFNet as ERFNet_imagenet
            pretrainedEnc = torch.nn.DataParallel(ERFNet_imagenet(1000))
            pretrainedEnc.load_state_dict(torch.load(args.pretrainedEncoder)['state_dict'])
            pretrainedEnc = next(pretrainedEnc.children()).features.encoder
            if (not args.cuda):
                pretrainedEnc = pretrainedEnc.cpu()     #because loaded encoder is probably saved in cuda
        else:
            pretrainedEnc = next(model.children()).encoder
        model = model_file.Net(NUM_CLASSES, encoder=pretrainedEnc)  #Add decoder to encoder
        if args.cuda:
            model = torch.nn.DataParallel(model).cuda()
        #When loading encoder reinitialize weights for decoder because they are set to 0 when training dec
    model = train(args, model, False)   #Train decoder
    print("========== TRAINING FINISHED ===========")

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', action='store_true', default=True)  #NOTE: cpu-only has not been tested so you might have to change code if you deactivate this flag
    parser.add_argument('--model', default="erfnet")
    parser.add_argument('--state')

    parser.add_argument('--port', type=int, default=8097)
    parser.add_argument('--datadir', default=os.getenv("HOME") + "/datasets/cityscapes/")
    parser.add_argument('--height', type=int, default=512)
    parser.add_argument('--num-epochs', type=int, default=150)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=6)
    parser.add_argument('--steps-loss', type=int, default=50)
    parser.add_argument('--steps-plot', type=int, default=50)
    parser.add_argument('--onlyone', type= bool,default=True,required=True, help="do you want to analyze the effect of only one loss you define in --lossfunction? Default is true")
    parser.add_argument('--epochs-save', type=int, default=0)    #You can use this value to save model every X epochs
    parser.add_argument('--savedir', required=True)
    parser.add_argument('--decoder', action='store_true')
    parser.add_argument('--pretrainedEncoder') #, default="../trained_models/erfnet_encoder_pretrained.pth.tar")
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--lossfunction', type=str, default="cross_entropy", help="Choose for cross_entropy for default one, logit_norm for Logit Normalization Loss, and enhanced_isotropy for Enhanced Isotropy Maximization Loss") #using training loss function is default setted on false
    parser.add_argument('--focal_loss',type=bool, default=False, help="do you want to analyze the effect of losses when trained joinly with focal loss? default is false, you train with cross_entropy by default") #### Analyze the effect of these losses when trained jointly with focal loss and cross-entropy loss
    parser.add_argument('--iouTrain', action='store_true', default=False) #recommended: False (takes more time to train otherwise)
    parser.add_argument('--iouVal', action='store_true', default=True)  
    parser.add_argument('--resume', action='store_true')    #Use this flag to load last checkpoint for training  

    main(parser.parse_args())
