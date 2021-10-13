#!/usr/bin/env python
# coding: utf-8

# # Bi-LSTM network with PyTorch
# ## Tokenization
# 
# When working with sequences ), each element of the input is referred to as a **token**. In Natural Language Processing (NLP) a token represents a word, or a component of a word. Tokenization is a fundamental step in NLP, where given a character sequence and a defined document unit, it will break up and separate the sequence into discrete elements (tokens). Therefore, tokens can take the form of words, characters, or sub-words. There are libraries such as *spaCy*, that can provide complex solutions to tokenization. 
# 
# However, for this project, the simple Python function *split()* was used to convert text into words. By default, the *split()* function splits words on white space.             

# In[ ]:


def tokenise(sample):
    """
    Called before any processing of the text has occurred.
    """

    processed = sample.split()

    return processed  


# ## Data preprocessing  
# 
# The following objects are required for data preparation in sentiment analysis task when using *torchtext.data*:
# 
# - **Field**: specifies how to preprocess each data column in our dataset.
# - **LabelField**: defines the label in the classification task.
# 
# The *main()* function in *a3main.py* script defines our *Field* and *LabelField* objects. We have defined the *Field* object to convert strings to lower case by passing *lower=True* argument. We have also set *include_length=True* to allow for dynamic padding by adding the lengths of the reviews to the dataset. In addition to the *Field* object, a *preprocessing()* function has been defined in *student.py* to perform the following additional preprocessing of the data using the regular expression package, *re*:
# 
# - remove html mark tags
# - remove non-ascii and digits
# - remove unwanted characters
# - remove extra white spaces.
# 
# The *preprocessing()* function will be called after tokenising but prior to numericalising to perform the text cleaning task. 

# In[ ]:


# function for cleaning texts - to  be used in preprocessing() function
def clean_texts(text):
    """
    Clean text of reviews. 
    """
    # remove html mark tags
    text = re.sub("(<.*?>)", "", text)
    # remove newline
    text = re.sub('\n', '', text)    
    #remove non-ascii and digits
    text = re.sub("(\\W|\\d)", " ", text)  
    #remove other characters 
    text = re.sub('[,.";!?:\(\)-/$\'%`=><“·^\{\}_&#»«\[\]~|@、´，]+', "", text)
    #remove whitespace
    text = text.strip()

    return text 


# In[ ]:


def preprocessing(sample):
    """
    Called after tokenising but before numericalising.
    """
    # clean the review texts
    sample = [clean_texts(text) for text in sample]

    return sample    


# ## Word embedding training algorithm
# 
# The model uses Global Vectors for Word Representation (GloVe 6B) unsupervised learning algorithm for obtaining vector representation for words (Pennington et al. 2014). GloVe is used for word embedding for text where it allows for words with similar meaning to have similar representation. The dimension of the vector was chosen to be 300 as the increase in dimension allows the vector to capture more information. For this task, dimension of 300 was found to perform better when we experimented with different dimension values (50, 150, 200, 250). However, it should be noted that increases in dimension size will result in greater computational complexity.

# In[ ]:


# number of features in the input
dimension = 300
# word embedding training algorithm 
wordVectors = GloVe(name='6B', dim = dimension)


# ## Preprocessing code
# 
# Putting all the code together:

# In[ ]:


"""
student.py

Neural Networks and Deep Learning

Author: Mohammad R. Hosseinzadeh 

You may modify this file however you wish, including creating additional
variables, functions, classes, etc., so long as your code runs with the
a3main.py file.

You have been given some default values for the variables stopWords,
wordVectors, trainValSplit, batchSize, epochs, and optimiser, as well as
a basic tokenise function.  You are encouraged to modify these to improve
the performance of your model.

The variable device may be used to refer to the CPU/GPU being used by PyTorch.
You may change this variable in the config.py file.

You may use GloVe 6B word vectors as found in the torchtext package.
------------------------------------------------------------------------------
"""

# Import packages
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

# Tokenization 
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


# ## Prediction output conversion 
# 
# We must ensure the output of our network is in the same Tensor data type (LongTensor) of the *rating* and *businessCategory* of *dataset* as defined by *main()* function in *a3main.py*. This has been achieved by defining *convertNetOutput(ratingOutput, categoryOutput)* function to process the prediction label data. 
# 
# Our model uses sigmoid activation function which outputs *FloatTensor* data type values between 0 and 1 to predict ratings. In this sentiment analysis task, our model is required to predict whether the review texts are negative or positive. Therefore, the ratings prediction is a binary classification task where it only requires to take on the value of 0 (negative) or 1 (positive). We can thus ensure the prediction label for *ratingOutput* is in *LongTensor* data type by applying the following process:
# 
# - *ratingOutput = torch.tensor([1 if x > 0.5 else 0 for x in ratingOutput]).to(device)*
# - prediction values over 0.5 given by our model will be assessed as positive and assigned 1 (integer64/long data type)
# - prediction values below 0.5 given by our model will be assessed as negative and assigned 0 (integer64/long data type).
# 
# The business categories target class labels in our dataset are [0, 1, 2, 3, 4], which represent restaurants, shopping, home services, health and medical, and automotive respectively. Our model uses the *CrossEntropyLoss()* loss function, which combines log softmax loss function and negative log-likelihood and outputs a probability distribution between 0 and 1. Therefore, the predicted label with the highest probability was assigned to the target class by applying the following conversion process:
# 
# - *categoryOutput = tnn.Softmax(categoryOutput, dim=1)* 
# - *categoryOutput = torch.argmax(categoryOutput, dim=1)*.
# 
# Prior to assigning a class label to the model's prediction *tnn.Softmax()* function is applied to to ensure the values lie in the range [0, 1] which can be interpreted as a probability distribution. Subsequently, the *torch.argmax()* function will return the indices of the maximum values of the category prediction output tensor in LongTensor data type representing the label with the highest probability (Paszke et al. 2019).    

# In[ ]:


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


# ## Network architecture 
# 
# Through experimentation, the following LSTM network architecture consistently produced satisfactory performance:
# 
# - batch size = 32
# - bidirectional
# - two recurrent layers, stacked LSTM
# - input size of 300 characters long
# - 148 features in each hidden state
# - 1 x fully connected hidden layer - 200 x 2 as input and 100 outputs
# - ReLU activation function
# - dropout - 30%
# - 1 x fully connected **rating** output layer - 100 inputs, 1 output for binary classification
# - sigmoid activation function to ensure predicted values are between 0 and 1 
# - 1 x fully connected **category** output layer - 100 inputs, 5 outputs. No activation function follows this layer as we are using *CrossEntropyLoss()*. 
# 
# Depth was added to our model by forming a two layer stacked LSTM. Increasing depth of a network can be viewed as a type of representational optimization. This can provide a model that requires fewer neurons to train and increasing computational efficiency by reducing training execution time.
# 
# Weight initialisation was also used to enhance the performance of our model. Teaching a neural network involves gradually improving network weights to minimize a loss function, resulting in a set of weights that can make optimal predictions (Glassner 2021). This process begins when initial values are assigned to the weights. In practice, weight initialisation has a major impact on the efficiency and performance of the model.    
# 
# For Rectified Linear Units (ReLU) it is recommended to use kaiming initialisation as shown by He et al. (2015), where the weights are chosen from Gaussian distribution with mean 0 and standard deviation $\frac{\sqrt{2}}{\sqrt{n_i}}$.  

# ## Overfitting 
# 
# To avoid our model from overfitting the data, the regularization methods dropout and weight decay were applied. When applying dropout, the dropout layer will temporarily disconnect some of the units inputs and outputs from the previous layer. Once the batch is completed, these units and all of their connections are restored. At the beginning of the next batch, a new random set of units are temporarily removed with this process repeating itself for each epoch. Dropout can delay overfitting as it prevents any unit from over-specialising on the training data. 
# 
# Weight decay, also known as L2 regularization, is another method used for regularising machine learning models. This is achieved by adding a penalty term to the loss function, which encourages the network weights to remain small. Weight decay indicates a Gaussian prior over the model parameters resulting in regularisation of the networks complexity (Graves 2011).
# 
# To evaluate the training of our network and obtain an estimate on the model's performance, we split the input data into a training set and validation set with 80% assigned to training and 20% assigned to validation. Subsequently, we obtain an estimate on the network's performance by making predictions for the validation set. This allows us to experiment with different hyperparameters and choose the best ones based on their performance on the validation set and help improve generalisation. Once the optimal parameters where found and we were confident that the model can generalise, the network was run on the complete training dataset. More training data helped increase the accuracy, specially when predicting business categories.     

# ## Cost function 
# 
# A cost function compares how far off a prediction is from its target in the training data and presents a real value score called the *loss*. The higher this score, the worse the network's prediction is. For tasks such as regression, MSE cost is preferred. However, it has a vital flaw in which neuron saturation can occur. This can negatively impact neural networks ability to learn. An alternative cost function is cross-entropy which is well suited to classification tasks. Cross-entropy loss function estimates the probability of a predicted label belonging to the target class label and leads to faster learning as well improved generalisation for classification problems. 
# 
# For the binary classification tasks such as rating prediction, the binary cross-entropy loss function can be applied. For implementation in PyTorch, the *nn.BCELoss()* function is used. To convert the subsequent output probabilities into two discrete classes, we can apply a decision boundary of 0.5. If the predicted probability is above 0.5, the predicted class is 1 (positive); otherwise the class is 0 (negative). The decision boundary value can also be tuned as a hyperparameter to achieve the desired accuracy.  
# 
# For a multi-class classification task such as predicting business categories, *nn.CrossEntropyLoss()* can be used. The outputs of this function can be interpreted as prediction probabilities of belonging to a target class label. Our aim is to have the probability of the correct class to be close to 1, with the other classes being close to 0. The target class with the highest predicted probability can be obtained by using *torch.argmax()*. 

# ## Optimiser 
# 
# The optimisers that were experimented with were the stochastic gradient descent (SGD) with momentum, and the adaptive moment estimation (Adam) optimizer. The performance of both were comparable. However, the speed of learning and the hyperparameters used differed. The network using SGD required the learning rate to be set to 0.07 with momentum of 0.75. On the other hand, the learning rate used for Adam was 0.001. For this particular task, the speed and efficiency of learning with Adam was slightly better compared to SGD with momentum. Therefore, Adam was chosen as the optimiser for the final network.
# 
# To find the optimal learning rate, many different values were applied and evaluated. At the end, the default value of 0.001 for Adam performed the best in conjunction with weight decay 1e-6, as well as all other parameters such as dropout, number of layers, number of hidden unit features etc.

# ## PyTorch model code
# 
# Below is code for creating our neural network model using PyTorch:

# In[ ]:


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


# In[ ]:


################################################################################
################## The following determines training options ###################
################################################################################

trainValSplit = 0.99    # change ratio to assist with design decisions aimed to avoid overfitting to the training data
batchSize = 32
epochs = 11
optimiser = toptim.Adam(net.parameters(), lr=0.001, weight_decay=1e-6)

