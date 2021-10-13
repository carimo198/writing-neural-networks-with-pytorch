# %load student.py

"""
student.py

Neural Networks and Deep Learning

Author: Mohammad Hosseinzadeh 

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
a3main.py.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as
a basic tokenise function.  You are encouraged to modify these to improve
the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may use GloVe 6B word vectors as found in the torchtext package.
------------------------------------------------------------------------------
"""

import torch 
import torch.nn as tnn    
import torch.optim as toptim  
from torchtext.vocab import GloVe
import numpy as np
import sklearn 
import re
import torch.nn.init as init 

from config import device 


################################################################################
##### The following determines the processing of input data (review text) ######
################################################################################

# set seed for reproducibility
torch.manual_seed(1234)

def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed

# function for cleaning texts - to  be used in preprocessing() 
def clean_texts(text):
    """
    Clean text of reviews. 
    """
    # remove html mark tags
    text=re.sub("(<.*?>)", "", text)
    # remove newline
    text = re.sub('\n', '', text)    
    #remove non-ascii and digits
    text=re.sub("(\\W|\\d)", " ", text)  
    #remove other characters 
    text = re.sub('[,.";!?:\(\)-/$\'%`=><“·^\{\}_&#»«\[\]~|@、´，]+', "", text)
    #remove whitespace
    text=text.strip()

    return text    

def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # clean the review texts
    sample = [clean_texts(text) for text in sample]

    return sample

def postprocessing(batch, vocab):
    """
    Called after numericalising but before vectorising.
    """

    return batch

stopWords = {}

# number of features in the input
dimension = 300
wordVectors = GloVe(name='6B', dim = dimension) 

################################################################################
####### The following determines the processing of label data (ratings) ########
################################################################################

def convertNetOutput(ratingOutput, categoryOutput):
    """
    Your model will be assessed on the predictions it makes, which must be in
    the same format as the dataset ratings and business categories.  The
    predictions must be of type LongTensor, taking the values 0 or 1 for the
    rating, and 0, 1, 2, 3, or 4 for the business category.  If your network
    outputs a different representation convert the output here.
    """
    ## ensure prediction outputs are of LongTensor type
    ## convert the probabilities to discrete classes
    # rating prediction labels - binary, taking value of 0 or 1
    ratingOutput = torch.tensor([1 if x > 0.5 else 0 for x in ratingOutput]).to(device)
    
    # apply softmax to ensure category prediction labels are in [0, 1] range
    softmax = tnn.Softmax(dim=1)
    categoryOutput = softmax(categoryOutput)
    
    # predict label with the highest probability
    categoryOutput = torch.argmax(categoryOutput, dim=1) 
    
    return ratingOutput, categoryOutput

################################################################################
###################### The following determines the model ######################
################################################################################

class network(tnn.Module):
    """
    Class for creating the neural network.  The input to your network will be a
    batch of reviews (in word vector form).  As reviews will have different
    numbers of words in them, padding has been added to the end of the reviews
    so we can form a batch of reviews of equal length.  Your forward method
    should return an output for both the rating and the business category.
    """

    def __init__(self):
        super(network, self).__init__()
        # number of expected features in the input
        self.input_size = dimension
        # number of features in the hidden state h
        self.hidden_size = 200
        # number of recurrent layers
        self.layers = 2
        # ReLU activation function
        self.relu = tnn.ReLU()
        # sigmoid activation function 
        self.sigmoid = tnn.Sigmoid()
        # dropout layer - 30%
        self.dropout = tnn.Dropout(0.3)

        # define a multi-layer bidirectional LSTM RNN to an input sequence
        self.lstm = tnn.LSTM(input_size = self.input_size, 
                            hidden_size = self.hidden_size,
                            num_layers = self.layers, 
                            batch_first=True,
                            bidirectional=True,
                            dropout=0.2)

        # initial fully connected hidden linear layer - * 2 for bidirectional
        self.hidden_layer = tnn.Linear(self.hidden_size * 2, 100)

        # fully connected output linear layer for ratings - 0,1 class
        self.fc1 = tnn.Linear(100, 1)

        # fully connected output linear layer for category - 0,1,2,3,4 class
        self.fc2 = tnn.Linear(100, 5)

    def forward(self, input, length):
        # set initial states
        self.h = torch.zeros(self.layers*2, input.size(0), self.hidden_size).to(device)
        self.c = torch.zeros(self.layers*2, input.size(0), self.hidden_size).to(device)

        # kaiming weight initialization
        self.h = init.kaiming_normal_(self.h, mode='fan_out', nonlinearity='relu').to(device)
        self.c = init.kaiming_normal_(self.c, mode='fan_out', nonlinearity='relu').to(device)

        # pack a Tensor containing padded sequences of varying lengths,
        # improves computational efficiency
        if torch.cuda.is_available():
            embedded_packed = tnn.utils.rnn.pack_padded_sequence(input.cpu(), length.cpu(), batch_first=True).to(device) 
        else:
            embedded_packed = tnn.utils.rnn.pack_padded_sequence(input, length, batch_first=True)  
            
        # pass packed sequence through LSTM
        lstm_out, (self.h, self.c) = self.lstm(embedded_packed)  
        
        # hidden state output
        output = torch.cat((self.h[-2,:,:], self.h[-1,:,:]), dim=1)

        # propagate through initial hidden layer
        output = self.hidden_layer(output)
        # apply ReLU activation
        output = self.relu(output)
        # apply dropout
        output = self.dropout(output)

        # rating output - binary classification  
        rating_out = self.fc1(output) 
        rating_out = self.sigmoid(rating_out)

        # category output - multiclass classification
        category_out = self.fc2(output)
       
        return rating_out, category_out

class loss(tnn.Module):
    """
    Class for creating the loss function.  The labels and outputs from your
    network will be passed to the forward method during training.
    """

    def __init__(self):
        super(loss, self).__init__()
        # binary cross entropy loss function for ratingOutput
        self.binary_loss = tnn.BCELoss()

        # cross entropy loss function for categoryOutput
        self.cross_ent = tnn.CrossEntropyLoss()

    def forward(self, ratingOutput, categoryOutput, ratingTarget, categoryTarget):
        # ratingOutput is of float type; convert ratingTarget to float
        ratingTarget = ratingTarget.type(torch.FloatTensor).to(device)
        # remove all the dimensions of size 1
        ratingOutput = torch.squeeze(ratingOutput) 

        # apply rating loss function
        rating_loss = self.binary_loss(ratingOutput, ratingTarget)

        # apply category loss function
        category_loss = self.cross_ent(categoryOutput, categoryTarget)

        # compute total loss    
        total_loss = rating_loss + category_loss 

        return total_loss  

net = network()
lossFunc = loss()

################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.99 # change ratio to assist with design decisions aimed to avoid overfitting to the training data  
batchSize = 32
epochs = 11
optimiser = toptim.Adam(net.parameters(), lr=0.001, weight_decay=1e-6) 