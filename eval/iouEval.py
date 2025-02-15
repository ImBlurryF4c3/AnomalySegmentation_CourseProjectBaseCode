# Code for evaluating IoU 
# Nov 2017
# Eduardo Romera
#######################

import torch

class iouEval:

    def __init__(self, nClasses, ignoreIndex=19):
        self.nClasses = nClasses
        self.ignoreIndex = ignoreIndex if nClasses>ignoreIndex else -1 #if ignoreIndex is larger than nClasses, consider no ignoreIndex
        self.reset()

    def reset (self):
        classes = self.nClasses if self.ignoreIndex==-1 else self.nClasses-1
        self.tp = torch.zeros(classes).double()
        self.fp = torch.zeros(classes).double()
        self.fn = torch.zeros(classes).double()        

    def addBatch(self, x, y):   #x=preds, y=targets
        #sizes should be "batch_size x nClasses x H x W"
        
        #print ("X is cuda: ", x.is_cuda)  -> cioè verifica se codice è allocato sulla GPU
        #print ("Y is cuda: ", y.is_cuda)

        if (x.is_cuda or y.is_cuda):
            x = x.cuda()
            y = y.cuda()

        #if size is "batch_size x 1 x H x W" scatter to onehot
            #La rappresentazione one-hot è un vettore che ha una 
            #lunghezza uguale al numero di classi e ha un 1 nella posizione della classe corrispondente e 0 altrove. 
            #Questo viene fatto utilizzando la funzione scatter_ di PyTorch.

        if (x.size(1) == 1):    #ovvero se il numero di classi è 1 -> come nel caso in cui volessimo applicare
            # allora x_onehot è un tensore di dimensione batch_size x nClasses x H x W
            # e tutti i valori sono 0 tranne quelli che corrispondono alla classe predetta, che sono 1.
            x_onehot = torch.zeros(x.size(0), self.nClasses, x.size(2), x.size(3))  
            if x.is_cuda:
                x_onehot = x_onehot.cuda()

            x_onehot.scatter_(1, x, 1).float() #scatter_(dim, index, src) -> dove src è il valore da assegnare e index è l'indice in cui assegnare il valore e dim è la dimensione in cui assegnare il valore
            #scatter_ è una funzione di PyTorch che assegna i valori di src a x_onehot in base agli indici forniti in index e serve a creare la rappresentazione one-hot.
        else:
            x_onehot = x.float()

        if (y.size(1) == 1):
            y_onehot = torch.zeros(y.size(0), self.nClasses, y.size(2), y.size(3))
            if y.is_cuda:
                y_onehot = y_onehot.cuda()
            y_onehot.scatter_(1, y, 1).float()
        else:
            y_onehot = y.float()

    # Se self.ignoreIndex non è -1, 
    #allora ci sono alcune classi che dovrebbero essere ignorate durante il calcolo dell'IoU.
    # In questo caso, il codice rimuove queste classi da x_onehot e y_onehot.
        if (self.ignoreIndex != -1): 
            ignores = y_onehot[:,self.ignoreIndex].unsqueeze(1)
            x_onehot = x_onehot[:, :self.ignoreIndex]
            y_onehot = y_onehot[:, :self.ignoreIndex]
        else:
            ignores=0

        #print(type(x_onehot))
        #print(type(y_onehot))
        #print(x_onehot.size())
        #print(y_onehot.size())

        # il codice calcola i veri positivi (tp), i falsi positivi (fp) e i falsi negativi (fn) per ogni classe. Questi sono calcolati utilizzando operazioni di moltiplicazione e somma su x_onehot e y_onehot. 
        # Questi valori sono poi sommati ai totali corrispondenti memorizzati nell'oggetto.
        tpmult = x_onehot * y_onehot    #times prediction and gt coincide is 1
        tp = torch.sum(torch.sum(torch.sum(tpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fpmult = x_onehot * (1-y_onehot-ignores) #times prediction says its that class and gt says its not (subtracting cases when its ignore label!)
        fp = torch.sum(torch.sum(torch.sum(fpmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze()
        fnmult = (1-x_onehot) * (y_onehot) #times prediction says its not that class and gt says it is
        fn = torch.sum(torch.sum(torch.sum(fnmult, dim=0, keepdim=True), dim=2, keepdim=True), dim=3, keepdim=True).squeeze() 

        self.tp += tp.double().cpu()
        self.fp += fp.double().cpu()
        self.fn += fn.double().cpu()

    def getIoU(self):
        num = self.tp
        den = self.tp + self.fp + self.fn + 1e-15
        iou = num / den
        return torch.mean(iou), iou     #returns "iou mean", "iou per class"

# Class for colors
class colors:
    RED       = '\033[31;1m'
    GREEN     = '\033[32;1m'
    YELLOW    = '\033[33;1m'
    BLUE      = '\033[34;1m'
    MAGENTA   = '\033[35;1m'
    CYAN      = '\033[36;1m'
    BOLD      = '\033[1m'
    UNDERLINE = '\033[4m'
    ENDC      = '\033[0m'

# Colored value output if colorized flag is activated.
def getColorEntry(val):
    if not isinstance(val, float):
        return colors.ENDC
    if (val < .20):
        return colors.RED
    elif (val < .40):
        return colors.YELLOW
    elif (val < .60):
        return colors.BLUE
    elif (val < .80):
        return colors.CYAN
    else:
        return colors.GREEN