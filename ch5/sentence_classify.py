# -*- coding: utf-8 -*-
"""
Created on Tue Mar 17 21:58:03 2020

@author: Peter Hsu
"""

# In[1]:

# These are all the modules we'll be using later. Make sure you 
# can import them before proceeding further.

from __future__ import print_function
import collections
import numpy as np
import os
import tensorflow as tf
from six.moves import range
from six.moves.urllib.request import urlretrieve

# ## Downloading and Checking the Dataset
# This [dataset](Dataset: http://cogcomp.cs.illinois.edu/Data/QA/QC/)
# is composed of questions as inputs and their respective type as the
# output. For example, (e.g. Who was Abraham Lincon?) and the output
# or label would be Human.
# In[2]:
url = 'http://cogcomp.org/Data/QA/QC/'
dir_name = 'question-classif-data'

def maybe_download(dir_name, filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(dir_name):
        os.mkdir(dir_name) 
  if not os.path.exists(filename):  
    filename, _ = urlretrieve(url + filename,filename)
  print(os.path.join(dir_name,filename))
  statinfo = os.stat(filename)  
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % os.path.join(dir_name,filename))
  else:
    print(statinfo.st_size)
    raise Exception('Failed to verify ' + os.path.join(
            dir_name,filename) + '. Can you get to it with a browser?')
  return filename

filename = maybe_download(dir_name, 'train_1000.label', 60774)
test_filename = maybe_download(dir_name, 'TREC_10.label',23354)
# In[3]:
# Check the existence of files
filenames = ['train_1000.label','TREC_10.label']
num_files = len(filenames)
for i in range(len(filenames)):
    file_exists = os.path.isfile(filenames[i])
    #file_exists = os.path.isfile(os.path(dir_name,filenames[i]))
    assert file_exists
print('Files found and verified.')


# ## Loading and Preprocessing Data
# Below we load the text into the program and do some simple preprocessing 
# In[]:
''' 
分類(taxonomy)  疑問句
大類: 次類     疑問句本文  （Training Sets: train_1000.label）  
DESC:manner How did serfdom develop in and then leave Russia ?  
ENTY:cremat What films featured the character Popeye Doyle ?  
DESC:manner How can I find a list of celebrities ' real names ?  
ENTY:animal What fowl grabs the spotlight after the Chinese Year of the Monkey?  
ABBR:exp What is the full form of .com ?
......  

大類: 次類     疑問句本文  （Test Sets: TREC_10.label）    
NUM:dist How far is it from Denver to Aspen ?
LOC:city What county is Modesto , California in ?
HUM:desc Who was Galileo ?
DESC:def What is an atom ?
NUM:date When did Hawaii become a state ?
......'''
# In[4]:
# Records the maximum length of the sentences
# as we need to pad shorter sentences accordingly
max_sent_length = 0 

def read_data(filename):
  '''
  Read data from a file with given filename
  Returns a list of strings where each string is a lower case word
  '''
  global max_sent_length
  questions = []
  labels = []
  with open(filename,'r',encoding='latin-1') as f:        
    for row in f:
        row_str = row.split(":")
        lb,q = row_str[0],row_str[1]
        q = q.lower()        
        ''' labels是單一list: []  '''
        labels.append(lb)
        '''questions是list內又append q.split()形成list of list: [[],[],...]
        # 注意：q尚未形成list,是經q.split()才形成list,即 questions內[]'''
        questions.append(q.split())                
        if len(questions[-1])>max_sent_length:
            max_sent_length = len(questions[-1])
  return questions,labels

# Process train and Test data
for i in range(num_files):    
    print('\nProcessing file %s'%os.path.join(dir_name,filenames[i]))
    if i==0:
        # Processing training data
        train_questions,train_labels = read_data(os.path.join(
                                        dir_name,filenames[i]))
        # Making sure we got all the questions and corresponding labels
        assert len(train_questions)==len(train_labels)
    elif i==1:
        # Processing testing data
        test_questions,test_labels = read_data(os.path.join(
                                      dir_name,filenames[i]))
        # Making sure we got all the questions and corresponding labels.
        assert len(test_questions)==len(test_labels)
        
    # Print some data to see everything is okey
    if i==0:        
        for j in range(5) :  
            print('\tTraining Question %d: %s' %(j,train_questions[j]))
            print('\tTraining Label %d: %s\n'%(j,train_labels[j]))
    if i==1:        
        for j in range(5) :  
            print('\tTest Question %d: %s' %(j,test_questions[j]))
            print('\tTest Label %d: %s\n'%(j,test_labels[j]))
        
print('Max Sentence Length: %d'%max_sent_length)
print('\nNormalizing all sentences to same length')


# ## Padding Shorter Sentences
# We use padding to pad short sentences so that 
# all the sentences are of the same length.
# In[5]:

# Padding training data
'''enumerate對train_questions編排序號，回傳序號(qi)及內容(que,各 內[])'''
for qi,que in enumerate(train_questions):
    '''增加'PAD'長度為max_sent_length-len(que)'''   
    for _ in range(max_sent_length-len(que)):
        que.append('PAD')
    assert len(que)==max_sent_length
    train_questions[qi] = que  # 每句增加完'PAD'再載入train_questions，統一長度
print('Train questions padded')

# Padding testing data
for qi,que in enumerate(test_questions):
    for _ in range(max_sent_length-len(que)):
        que.append('PAD')
    assert len(que)==max_sent_length
    test_questions[qi] = que
print('\nTest questions padded')  

# Printing a test question to see if everything is correct
print('\nSample test question: %s',test_questions[0])


# ## Building the Dictionaries
# Builds the following. To understand each of these elements,
# let us also assume the text "I like to go to school"
# 
# * `dictionary`: maps a string word to an ID 
#                    (e.g. {I:0, like:1, to:2, go:3, school:4})
# * `reverse_dictionary`: maps an ID to a string word 
#                    (e.g. {0:I, 1:like, 2:to, 3:go, 4:school}
# * `count`: List of list of (word, frequency) elements 
#                    (e.g. [(I,1),(like,1),(to,2),(go,1),(school,1)]
# * `data` : Contain the string of text we read, where string 
#            words are replaced with word IDs (e.g. [0, 1, 2, 3, 2, 4])
# 
# We do not replace rare words with "UNK" because the
# vocabulary is already quite small.
# In[6]:
def build_dataset(questions):
    words = []
    text_digt = []
    count = []
    
    # First create a large list with all the words in all the questions
    for d in questions:
        words.extend(d)
    print('%d Words in total questions.'%len(words))    
    print('Set vocabulary and found %d words in the vocabulary.'\
          %len(collections.Counter(words).most_common()))
    
    # Sort words by there frequency
    '''count包含[(word, frequency),....]，由frequency多到少儲存'''
    count.extend(collections.Counter(words).most_common())       
    '''dictionary逐漸增加，所以len(dictionary)不同，
      此值藉dictionary[word]賦值給每一個word形成各自的數字化ID（由於
      count依出現先次數多到少順序排列，所以常見的字符ID值小,'PAD':0），
      並以此ID值代替此word做後續運算，並建立reverse_dictionary'''
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    
    # Traverse through all the text and 
    # replace the string words with the ID 
    # of the word found at that index
    '''藉數字化dictionary遍歷questions，取代word text成 digital text'''
    for d in questions:
        data = list()
        for word in d:
            index = dictionary[word]        
            data.append(index)
            
        text_digt.append(data)
    ''''''    
    reverse_dictionary = dict(zip(dictionary.values(),
                                  dictionary.keys())) 
    
    return text_digt, count, dictionary, reverse_dictionary

# Create a dataset with both train and test questions
#all_questions = train_questions
'''all_questions要用list(train_questions)宣告，以不影響train_questions大小1000
   若僅all_questions = train_questions宣告，則每執行一次
    train_questions大小會變大500->1500->2000->...，
    因為all_questions又extend(test_questions)加500，
    會反過來改變train_questions大小'''    
all_questions = list(train_questions)
all_questions.extend(test_questions)
# Use the above created dataset to build the vocabulary
all_question_digt, count, dictionary, reverse_dictionary =\
                  build_dataset(all_questions)

# Print some statistics about the processed data
print('\nAll words (count)', count[:5])
#print('dictionary:',dictionary)
print('\n0th entry in Reverse_dictionary: %s'%reverse_dictionary[0])
print('\nSample data', all_question_digt[0])
print('\nSample data', all_question_digt[1])
print('\nVocabulary: ',len(dictionary))
vocabulary_size = len(dictionary)

print('\nNumber of training questions: ',len(train_questions))
print('Number of testing questions: ',len(test_questions))


# ## Generating Batches of Data
# Below I show the code to generate a batch of
# data from a given set of questions and labels.
# In[7]:
batch_size = 16 # We process 16 questions at a time
sent_length = max_sent_length

num_classes = 6 # Number of classes
# All the types of question that are in the dataset
all_labels = ['NUM','LOC','HUM','DESC','ENTY','ABBR'] 

class BatchGenerator(object):
    '''
    Generates a batch of data
    傳入：batch_size, questions, labels
    '''
    def __init__(self,batch_size,questions,labels):
        self.questions = questions
        self.labels = labels
        self.text_size = len(questions)
        self.batch_size = batch_size
        self.data_index = 0
        assert len(self.questions)==len(self.labels)
        
    def generate_batch(self):
        '''
        產生inputs, labels_ohe 都是one hot encoded        
        回傳： inputs:[batch_size, sent_length, vocabulary_size]
                其中sent_length = max_sent_length = 33 
                   vocabulary_size = len(dictionary) = 3369
              labels_ohe:[batch_size, num_classes]
                其中num_classes是label 分類數 = 6
        i'''
        global sent_length,num_classes
        global dictionary, all_labels
        
        # Numpy arrays holding input and label data
        inputs = np.zeros((self.batch_size, sent_length,
                           vocabulary_size),dtype=np.float32)
        labels_ohe = np.zeros((self.batch_size,
                           num_classes),dtype=np.float32)
        
        # When we reach the end of the dataset
        # start from beginning
        if self.data_index + self.batch_size >= self.text_size:
            self.data_index = 0
            
        '''對input及labels_ohe進行one hot encoded'''
        ''' 對每一批次的questions編序qi,取值que(每一question)'''
        for qi,que in enumerate(self.questions[self.data_index:
            self.data_index+self.batch_size]): 
            '''對每一question編序wi, 取值word(question中的每個字)'''
            # For each word in the question
            for wi,word in enumerate(que):  
                # Set the element at the word ID index to 1
                # this gives the one-hot-encoded vector of that word
                '''在批次第qi個question(qi)中,第wi個字(sentence),
                   於字典(dictionary)所在ID處設為1
                '''
                inputs[qi,wi,dictionary[word]] = 1.0
            
            # Set the index corrsponding to that particular class to 1
            '''在批次第qi個question(qi)中, 藉查此qi的labels在all_labels
               的index所在處設為1
               其中，self.data_index + qi (簡寫a)指以全部labels取位置，
               所以，self.labels[a] (簡寫b)取此位置的值，例如'HUM'
               ＝> all_labels.index(b)在此值'HUM'處設為1 得[0,0,1,0,0,0]'''
            labels_ohe[qi, all_labels.index(
                    self.labels[self.data_index + qi])] = 1.0
        
        # Update the data index to get the next batch of data
        #self.data_index = (self.data_index+self.batch_size)%self.text_size
        self.data_index = (self.data_index + self.batch_size)                        
        return inputs, labels_ohe
    
    def return_index(self):
        # Get the current index of data
        return self.data_index

# Test our batch generator
sample_gen = BatchGenerator(batch_size,
                        train_questions,train_labels)
# Generate a single batch
sample_batch_inputs, sample_batch_labels =\
                        sample_gen.generate_batch()        
                       
# Generate another batch
sample_batch_inputs_2,sample_batch_labels_2 =\
                        sample_gen.generate_batch()

# Make sure that we in fact have the 
# question 0 as the 0th element of our batch
assert np.all(np.asarray([dictionary[w] for
              w in train_questions[0]],dtype=np.int32) 
              == np.argmax(sample_batch_inputs[0,:,:],axis=1))

# Print some data labels we obtained
print('sample_batch_inputs shape:\n',sample_batch_inputs.shape)
print('sample_batch_labels shape:\n',sample_batch_labels.shape)
print(np.argmax(sample_batch_labels,  axis=1))
print(np.argmax(sample_batch_labels_2,axis=1))


# ## Sentence Classifying Convolution Neural Network
# We are going to implement a very simple CNN to
# classify sentences. However you will see that even
# with this simple structure we achieve good accuracies.
# Our CNN will have one layer (with 3 different parallel layers). This will be followed by a pooling-over-time layer and finally a fully connected layer that produces the logits.

# ## Defining hyperparameters and inputs
# In[8]:
tf.reset_default_graph()

batch_size = 32
# Different filter sizes we use in a single convolution layer
filter_sizes = [3,5,7] 
# inputs and labels
sent_inputs = tf.placeholder(shape=[batch_size, sent_length,
        vocabulary_size],dtype=tf.float32,name='sentence_inputs')
sent_labels = tf.placeholder(shape=[batch_size, 
        num_classes],dtype=tf.float32,name='sentence_labels')

# In[]:
'''    
# ## Defining Model Parameters
 Our model has following parameters.
  * 3 sets of convolution layer
   weights and biases (one for each parallel layer)
  * 1 fully connected output layer
 3 filters with different context window sizes (3,5,7)
 Each of this filter spans the full one-hot-encoded 
 length of each word and the context window width
'''
# In[9]:
# Weights of the first parallel layer
# w1, w2, w3 nornal distbution shape(3, 3369, 1)
w1 = tf.Variable(tf.truncated_normal([filter_sizes[0],
                  vocabulary_size, 1], stddev=0.02,
                  dtype=tf.float32), name='weights_1')
# b1, b2, b3 uniform distbution shape(1) 
b1 = tf.Variable(tf.random_uniform([1], 0, 0.01,
                  dtype=tf.float32), name='bias_1')
# Weights of the second parallel layer
# w2 nornal distbution shape(5, 3369, 1)
w2 = tf.Variable(tf.truncated_normal([filter_sizes[1],
                  vocabulary_size, 1], stddev=0.02, 
                  dtype=tf.float32), name='weights_2')
b2 = tf.Variable(tf.random_uniform([1], 0, 0.01,
                  dtype=tf.float32), name='bias_2')

# Weights of the third parallel layer
# w3 nornal distbution shape(7, 3369, 1)
w3 = tf.Variable(tf.truncated_normal([filter_sizes[2],
                  vocabulary_size, 1], stddev=0.02, 
                  dtype=tf.float32), name='weights_3')
b3 = tf.Variable(tf.random_uniform([1], 0, 0.01, 
                  dtype=tf.float32), name='bias_3')

# Fully connected layer
# w_fc1 nornal distbution shape(3, 6)，
# len(filter_sizes)=3，其中 filter_sizes = [3,5,7]為 3 dimension
w_fc1 = tf.Variable(tf.truncated_normal(
           [len(filter_sizes), num_classes], stddev=0.5, 
            dtype=tf.float32), name='weights_fulcon_1')
b_fc1 = tf.Variable(tf.random_uniform([num_classes], 0, 0.01,
            dtype=tf.float32), name='bias_fulcon_1')

# ## Defining Inference of the CNN
# Here we define the CNN inference logic. First compute the convolution output for each parallel layer within the convolution layer. Then perform pooling-over-time over all the convolution outputs. Finally feed the output of the pooling layer to a fully connected layer to obtain the output logits.
# In[10]:
# Calculate the output for all the filters with a stride 1
# We use relu activation as the activation function
'''conv1d參數 sent_inputs shape:[batch_size, sent_length,
   vocabulary_size]的第3維要與w1 shape:[filter_sizes[0],
   vocabulary_size,1]第2維vocabulary_size的大小一樣 '''
h1_1 = tf.nn.relu(tf.nn.conv1d(sent_inputs, w1,
                  stride=1, padding='SAME') + b1)
h1_2 = tf.nn.relu(tf.nn.conv1d(sent_inputs, w2,
                  stride=1, padding='SAME') + b2)
h1_3 = tf.nn.relu(tf.nn.conv1d(sent_inputs, w3,
                  stride=1, padding='SAME') + b3)

# Pooling over time operation

# This is doing the max pooling.  
# Thereare two options to do the max pooling
# 1. Use tf.nn.max_pool operation on a tensor made by tensor
#    concatenating h1_1 ,h1_2, h1_3 and converting that
#    to 4D (Because max_pool takes a tensor of rank >= 4)
# 2. Do the max pooling separately for each filter output and
#   combine them using tf.concat(this is the one used in the code)
h2_1 = tf.reduce_max(h1_1, axis=1)
h2_2 = tf.reduce_max(h1_2, axis=1)
h2_3 = tf.reduce_max(h1_3, axis=1)

h2 = tf.concat([h2_1, h2_2, h2_3], axis=1)

# Calculate the fully connected layer output (no activation)
# Note: since h2 is 2d [batch_size,number of parallel filters] 
# reshaping the output is not required as it usually do in CNNs
logits = tf.matmul(h2, w_fc1) + b_fc1

# ## Model Loss and the Optimizer
# We compute the cross entropy loss and use the momentum optimizer (which works better than standard gradient descent) to optimize our model
# In[11]:
# Loss (Cross-Entropy)
loss = tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits_v2(
        labels = sent_labels, logits = logits))

# Momentum Optimizer
optimizer = tf.train.MomentumOptimizer(
        learning_rate = 0.01, momentum=0.9).minimize(loss)

# ## Model Predictions
# Note that we are not getting the raw predictions,
# but the index of the maximally activated element
# in the prediction vector.
# In[12]:
predictions = tf.argmax(tf.nn.softmax(logits), axis=1)

# ## Running Our Model to Classify Sentences
# 
# Below we run our algorithm for 50 epochs.
# With the provided hyperparameters you should achieve
# around 90% accuracy on the test set. However you 
# are welcome to play around with the hyperparameters.

# In[13]:
# With filter widths [3,5,7] and batch_size 32 the algorithm 
# achieves around ~90% accuracy on test dataset (50 epochs). 
# From batch sizes [16,32,64] I found 32 to give best performance

session = tf.InteractiveSession()

num_steps = 200 # Number of epochs the algorithm runs for

# Initialize all variables
tf.global_variables_initializer().run()
print('Initialized\n')

# Define data batch generators for train and test data
train_gen = BatchGenerator(batch_size,
                   train_questions, train_labels)
test_gen = BatchGenerator(batch_size, 
                   test_questions, test_labels)

# How often do we compute the test accuracy
test_interval = 1

# Compute accuracy for a given set of predictions and labels
def accuracy(labels, preds):
    return np.sum(np.argmax(
            labels, axis=1) == preds) / labels.shape[0]
# Running the algorithm
for step in range(num_steps):
    avg_loss = []
    
    # A single traverse through the whole training set
    for tr_i in range((len(
            train_questions)//batch_size)-1):
        # Get a batch of data
        tr_inputs, tr_labels = train_gen.generate_batch()
        # Optimize the network and compute the loss
        loss_val, _ = session.run([loss,optimizer],
               feed_dict={sent_inputs: tr_inputs,
                          sent_labels: tr_labels})
        avg_loss.append(loss_val)

    # Print average loss
    print('Train Loss at Epoch %d: %.2f'%(
            step,np.mean(avg_loss)))
    test_accuracy = []
    
    # Compute the test accuracy
    if (step + 1) % test_interval == 0:        
        for ts_i in range((
            len(test_questions)-1) // batch_size):
            # Get a batch of test data
            ts_inputs, ts_labels = test_gen.generate_batch()
            # Get predictions for that batch
            preds = session.run(predictions,feed_dict={
                    sent_inputs: ts_inputs,
                    sent_labels: ts_labels})
            # Compute test accuracy
            test_accuracy.append(accuracy(ts_labels, preds))
        
        # Display the mean test accuracy
        print('Test accuracy at Epoch %d: %.3f%%'%(
                step, np.mean(test_accuracy)*100.0))

