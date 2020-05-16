# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 20:17:25 2020

@author: Peter Hsu
"""

# In[1]:

# These are all the modules we'll be using later. 
# Make sure you can import them before proceeding further.
get_ipython().run_line_magic('matplotlib', 'inline')
from __future__ import print_function
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
import tensorflow as tf
from scipy.sparse import lil_matrix
#import nltk
#nltk.download() #tokenizers/punkt/PY3/english.pickle
# In[2_0]: ## Downloading Data
# 
# Downloading stories if not present in disk.
# There should be 100 files ('stories/001.txt',
#  'stories/002.txt', ...)
# In[2]:
url = 'https://www.cs.cmu.edu/~spok/grimmtmp/'

# Create a directory if needed
dir_name = 'stories'
if not os.path.exists(dir_name):
    os.mkdir(dir_name)
    
def maybe_download(filename):
  """Download a file if not present"""
  print('Downloading file: ', dir_name+ os.sep+filename)
    
  if not os.path.exists(dir_name+os.sep+filename):
    filename, _ = urlretrieve(url + filename, dir_name+os.sep+filename)
  else:
    print('File ',filename, ' already exists.')
  
  return filename

num_files = 100
filenames = [format(i, '03d')+'.txt' for i in range(1,101)]

for fn in filenames:
    maybe_download(fn)
# In[3_0]:
# ## Reading data
# Data will be stored in a list of lists where the each
#   list represents a document and document is a list
#  of words. We will then break the text into bigrams.
# In[3]:
'''把 filename 的句子拆開成各個小寫 character的list'''
def read_data(filename):
  with open(filename) as f:
    data = tf.compat.as_str(f.read())
    data = data.lower()
    data = list(data)
  return data

documents = []
global documents
for i in range(num_files):    
    print('\nProcessing file %s'%os.path.join(dir_name,filenames[i]))
    chars = read_data(os.path.join(dir_name,filenames[i]))
    #把character兩兩集合成str，例['th', 'er', 'e ',...]
    two_grams = [''.join(chars[ch_i:ch_i+2]) for ch_i
                 in range(0,len(chars)-2,2)]
    documents.append(two_grams)
    print('Data size (Characters) (Document %d) %d' %(i,len(two_grams)))
    print('Sample string (Document %d) %s'%(i,two_grams[:50]))

'''Final format of documents is list of lists (length of 100), 
    the outer list denote each document and the inner lists 
    denote words (Bigrams)in a given document.
    (e.g.[['in',' o','ld',...],['th','er','e ',...],...])'''
#print('length documents', len(documents))
#print('documents', documents[1:2][:50])

# In[4_0]:
'''
 Building the Dictionaries (Bigrams)
 
 Builds the following. To understand each of these
  elements, let us also assume the text 
  "I like to go to school"
 
 * `dictionary`: maps a string word to an ID 
    (e.g. {I:0, like:1, to:2, go:3, school:4})
    
 * `reverse_dictionary`: maps an ID to a string word 
    (e.g. {0:I, 1:like, 2:to, 3:go, 4:school}
    
 * `count`: List of list of (word, frequency) elements 
    (e.g. [(I,1),(like,1),(to,2),(go,1),(school,1)]
    
 * `data` : Contain the string of text we read, 
           where string words are replaced with word IDs
           (e.g. [0, 1, 2, 3, 2, 4])
 
 It also introduces an additional special token `UNK`
 to denote rare words to are too rare to make use of.
'''

# In[4]:
'''dictionary, reverse dictionary, IDs等都是按照documents建立，
   而此處是document傳入的是Bigrams，與之前章節傳入的words不同'''
def build_dataset(documents):
    chars = []
    # This is going to be a list of lists
    # Where the outer list denote each document
    # and the inner lists denote words in a given document
    text_digt = []
  
    for d in documents:
        chars.extend(d)
    print('%d Characters found.'%len(chars))
    count = []
    # Get the bigram sorted by their frequency (Highest comes first)
    count.extend(collections.Counter(chars).most_common())
    
    # Create an ID for each bigram by giving the current length
    #  of the dictionary, snd adding that item to the dictionary.    
    # Start with 'UNK' that is assigned to too rare words
    dictionary = dict({'UNK':0})
    for char, c in count:
        # Only add a bigram to dictionary if its frequency is more than 10
        if c > 10:
            dictionary[char] = len(dictionary)    
    
    unk_count = 0
    # Traverse through all the text we have
    # to replace each string word with the ID of the word
    for d in documents:
        data = list()
        for char in d:
            # If word is in the dictionary use the word ID,
            # else use the ID of the special token "UNK"
            if char in dictionary:
                index = dictionary[char]        
            else:
                index = dictionary['UNK']
                unk_count += 1
            data.append(index)
            
        text_digt.append(data)
        
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
    return text_digt, count, dictionary, reverse_dictionary

global text_digt, count, dictionary, reverse_dictionary,vocabulary_size

# Print some statistics about data
text_digt, count, dictionary, reverse_dictionary = build_dataset(documents)
print('Most common words (+UNK)', count[:5])
print('Least common words (+UNK)', count[-15:])
print('Sample data', text_digt[0][:10])
print('Sample data', text_digt[1][:10])
print('Vocabulary: ',len(dictionary))
vocabulary_size = len(dictionary)
del documents  # To reduce memory.

# In[5_0]:
'''# ## Generating Batches of Data
# The following object generates a batch of data which will
  be used to train the RNN. More specifically the generator
  breaks a given sequence of words into `batch_size` segments.
  We also maintain a cursor for each segment. So whenever we
  create a batch of data, we sample one item from each segment
  and update the cursor of each segment. 
'''
# In[5]:

class DataGeneratorOHE(object):
    
    def __init__(self,text,batch_size,num_unroll):
        # Text where a bigram is denoted by its ID
        self._text = text
        # Number of bigrams in the text
        self._text_size = len(self._text)
        # Number of datapoints in a batch of data
        self._batch_size = batch_size
        # Num unroll is the number of steps we unroll
        # the RNN in a single training step.
        # This relates to the truncated backpropagation
        # we discuss in Chapter 6 text
        self._num_unroll = num_unroll
        # We break the text in to several segments and
        # the batch of data is sampled by sampling 
        #  a single item from a single segment
        '''batch_size是每次處理的字(bigrams)數
           segment為間隔取樣點大小，若為6 (30/5)'''
        self._segments = self._text_size//self._batch_size
        '''cursor指標指向間隔取樣點，若segment=6，
           則cursor=[0, 6, 12, 18, 24]'''
        self._cursor = [offset * self._segments for offset
                        in range(self._batch_size)]        
    def next_batch(self):
        '''
        Generates a single batch of data
        '''
        # Train inputs (one-hot-encoded) and train outputs (one-hot-encoded)
        '''為了one-hot-encoded，column size取vocabulary_size'''
        batch_data = np.zeros((self._batch_size,vocabulary_size),
                              dtype=np.float32)
        batch_labels = np.zeros((self._batch_size,vocabulary_size),
                              dtype=np.float32)
        
        # Fill in the batch datapoint by datapoint        
        for b in range(self._batch_size): 
            # If the cursor of a given segment exceeds the text length
            # we reset the cursor back to the beginning of that segment
            if self._cursor[b]+1>=self._text_size:
                self._cursor[b] = b * self._segments
            
            # Add the text at the cursor as the input
            ''' batch input data (X)
            row (b)=>batch_data的第幾個字
            column =>cursor[b]指的text word ID處設為1 => one-hot-encoded'''
            batch_data[b,self._text[self._cursor[b]]] = 1.0
            # Add the preceding bigram as the label to be predicted
            ''' batch labels (y)
            column =>cursor[b]+1意味著以下一個text word (bigram)當label'''
            batch_labels[b,self._text[self._cursor[b]+1]]= 1.0                       
            # Update the cursor(e.g. [0,6,12,18,24]=>[1,7,13,19,25])
            self._cursor[b] = (self._cursor[b]+1)%self._text_size                    
        return batch_data,batch_labels
        
    def unroll_batches(self):
        '''
        This produces a list of num_unroll batches
        as required by a single step of training of the RNN
        '''
        unroll_data,unroll_labels = [],[]
        for ui in range(self._num_unroll):            
            data, labels = self.next_batch()            
            unroll_data.append(data)
            unroll_labels.append(labels)
        ''' unroll_data和unroll_labels格式 list of np arrays,
         即,[num_unroll個[batch_size, vocabulary_size] arrays] '''      
        return unroll_data, unroll_labels
    
    def reset_indices(self):
        '''
        Used to reset all the cursors if needed
        '''
        self._cursor = [offset * self._segments for offset
                        in range(self._batch_size)]
        
# Running a tiny set to see if things are correct
print('len(text_digt[0])', len(text_digt[0]))
print('len(text_digt[0][0:30])', len(text_digt[0][0:30]))
print('len(text_digt[0][0:30])', text_digt[0][0:30])
dg = DataGeneratorOHE(text_digt[0][0:30],5,6)
unroll_X, unroll_y = dg.unroll_batches()
print('segments =', len(text_digt[0][0:30])//5)

# Iterate through each data batch in the unrolled set of batches
for ui,(dat,lbl) in enumerate(zip(unroll_X,unroll_y)):   
    print('\nUnrolled index %d'%ui)
    dat_ind = np.argmax(dat,axis=1)
    lbl_ind = np.argmax(lbl,axis=1)
    print('Inputs:')
    for single_dat in dat_ind:
        print('%s (%d)'%(reverse_dictionary[single_dat],single_dat),end=", ")
    print('\nOutput:')
    for single_lbl in lbl_ind:        
        print('%s (%d)'%(reverse_dictionary[single_lbl],single_lbl),end=", ")
# In[6_0]:
# ## Recurrent Neural Network
# Here we implement and train our recurrent model
# that will take an output a new story

# ## Defining Hyperparameters
# 
# Here we define several hyperparameters required.
# * `num_unroll`: Number of steps we unroll over time during optimizing
# * `train_batchSize`: Number of datapoints in a single batch
# * `n_hidden`: Number of hidden neurons in the state

# In[6]:
        
tf.reset_default_graph()
 
train_batchSize = 60 # bigrams of each training time
valid_batchSize = 1
predPhase_batchSize = 1 # At predict phase time

train_n_unroll = 50 # Number of steps to unroll
valid_n_unroll= 1
predPhase_n_unroll = 1
# Number of hidden neurons in the state
n_hidden = 64
# Input size and output Size
n_inputs, n_outputs = vocabulary_size, vocabulary_size
# In[7_0]:
# ## Defining Inputs and Outputs
#  we define 
#  train inputs (`train_dataset`), train outputs(`train_labels`); 
#  validation inputs(`valid_dataset`), outputs (`valid_labels`);
#  predict phase inputs (`predPhase_dataset`).
# In[7]:
# Train dataset
# We use unrolling over time

global train_zeroState, valid_zeroState  
global predPhase_zeroState
  
#batchSize = tf.placeholder(tf.float32)
#n_unroll = tf.placeholder(tf.float32)
batchSize = train_batchSize
n_unroll = train_n_unroll
dataset = tf.placeholder(tf.float32,
                   [None, None, n_inputs],
                   name='dataset')  
labels = tf.placeholder(tf.float32,
                   [None, None, n_outputs],
                   name='labels')
''' 
(1).若cell只為簡單BasicRNNCell其train_outputs的shape
    為(train_batchSize, num_unroll, n_hidden)要經dense改造使
    trainScores的shape為(train_batchSize, num_unroll, n_outputs)，
    ，而與train_labels的shape相同才能計算loss。
(2)若cell再用OutputProjectionWrappe打包則其train_outputs的shape
    為(train_batchSize, num_unroll, n_outputs)，就可用來計算loss，
    但也可再經dense，其shape不變一樣可計算loss。此意味是多一層訓練
另，無論(1)或(2)states只回傳最後一個step的state，
    其shape為(train_batchSize, n_hidden)
    trainPredictions shape為(train_batchSize, num_unroll, n_outputs)   
'''
cell =  tf.nn.rnn_cell.BasicRNNCell(num_units= n_hidden,
       activation =tf.nn.relu)
#(2) cell = tf.contrib.rnn.OutputProjectionWrapper(
#        tf.nn.rnn_cell.BasicRNNCell(num_units= n_hidden,
#        activation =tf.nn.relu), output_size = n_outputs)
'''# In[]:
reset_train_state_op = tf.assign(prev_train_state,
                     tf.zeros([train_batchSize, n_hidden],
                     dtype=tf.float32))
reset_valid_state_op = tf.assign(prev_valid_state,
                     tf.zeros([1, n_hidden],
                     dtype=tf.float32))
'''
with tf.variable_scope('train_zeroState', reuse = True):
    train_zeroState = cell.zero_state(train_batchSize,
                dtype = tf.float32)
    train_zeroState = tf.zeros([train_batchSize, n_hidden],
                     dtype=tf.float32)

with tf.variable_scope('valid_zeroState', reuse = True):
    valid_zeroState = cell.zero_state(train_batchSize,
                dtype = tf.float32)               
    valid_zeroState = tf.zeros([train_batchSize, n_hidden],
                     dtype=tf.float32)        
with tf.variable_scope('predPhase_zeroState', reuse = True):
    predPhase_zeroState = cell.zero_state(train_batchSize,
                dtype = tf.float32)
    predPhase_zeroState = tf.truncated_normal(
              [train_batchSize, n_hidden],
              stddev=0.01, dtype=tf.float32)
batchSize = tf.placeholder(dtype =tf.int32, name='batchSize')  
n_unroll  = tf.placeholder(dtype =tf.int32, name='n_unroll')      

'''
print('batchSize 0:',batchSize)
print('n_unroll 0:', n_unroll)
print('zeroState 0:',zeroState)
batchSize = valid_batchSize
n_unroll = valid_n_unroll
 zeroState = valid_zeroState
print('batchSize 1:',batchSize)
print('n_unroll 1:', n_unroll)
print('zeroState 1:',zeroState)
initial_state = zeroState,'''

outputs, states = tf.nn.dynamic_rnn(cell,
                    dataset, dtype = tf.float32)
scores = tf.layers.dense(outputs, n_outputs)

predictions = tf.nn.softmax(scores)

loss = tf.reduce_mean(
             tf.nn.softmax_cross_entropy_with_logits_v2(
             logits=scores, labels=labels)) #純量

perplexity_without_exp = tf.reduce_sum(
        labels * - tf.log(predictions+1e-10)        
        )/(tf.cast(n_unroll, tf.float32
           )*tf.cast(batchSize, tf.float32))  #純量

'''!!!!! Test List !!!!! 
效果同上，但用list[tensor, tensor,...]計算，其中tensor shape
為[None(train_batchSize), n_outputs], list length為num_unroll

trainScores_list = tf.unstack(trainScores, axis=1) 
trainPredictions_list =  [tf.nn.softmax(trainScores_list[ui]) 
                 for ui in range(num_unroll)]

train_perplexity_without_exp_list = tf.reduce_sum(
        tf.concat(train_labels_list,0) * - tf.log(
        tf.concat(trainPredictions_list,0)+1e-10)
        )/(num_unroll*train_batchSize)  #純量
            
train_loss_list = tf.reduce_mean(
             tf.nn.softmax_cross_entropy_with_logits_v2(
             logits=tf.concat(trainScores_list, 0),
             labels=tf.concat(train_labels_list, 0))) #純量
!!!!! Test End !!!!!'''
rnn_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)
gradients, v = zip(*rnn_optimizer.compute_gradients(loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
rnn_optimizer = rnn_optimizer.apply_gradients(zip(gradients,v)) 
#training_op = rnn_optimizer.minimize(train_loss)
#print('zip(*rnn_optimizer.compute_gradients(train_loss)):',
#     type(zip(*rnn_optimizer.compute_gradients(train_loss))))
# In[10]:
def sample(distribution):
  '''
  Sample a word from the prediction distribution
  '''  
  best_idx = np.argmax(distribution)
  return best_idx
# In[9_0]:
# ## Running the RNN to Generate Text
# 
# Here we train the RNN on the available data and
#  generate text using the trained RNN for several steps.
# First we create a validation set by extracting text 
#  snippets (that are not present in training data) 
#  from longer documents. Then at each training step,
#  we train the RNN on several randomly picked documents.
# From each document we extract text for 
#  `train_times_per_doc` times. 
# We also report the train and validation perplexities 
#  at the end of each step.   
# Finally, predict (predict phase) the RNN by asking it to
# generate some new text starting from a randomly picked bigram.
# In[9]:
n_iterations = 1
#n_iterations = 26 # Number of iteration we run the algorithm for

# How many training times are performed for each 
# document in a single iteration
train_times_per_doc = 100 

# How often we run validation
valid_often = 1
# We run training documents with this set to both 20 and 100.
train_doc_count = 20
# Number of docs we use in a single iteration
# When train_doc_count = 20 => train_docs_to_use = 5
# # When train_doc_count = 100 => train_docs_to_use = 10
train_docs_to_use =5 

# Store the training and validation perplexity at each iteration
valid_perplexity_store = []
train_perplexity_store = []
# In[10]:

''' @@@  Generate Training and Validation Data & Label Phase @@@'''

'''找出前10個文件，這些文件的bigrams數都超過
(n_iterations+1) * train_times_per_doc個bigrams，以產生validation dataset，
並把這些文件編號放入long_doc_ids'''
long_doc_ids = []
#print('threshold:', (n_iterations+1)*train_times_per_doc)
for di in range(num_files):
  if len(text_digt[di])>(n_iterations+1)*train_times_per_doc:
    long_doc_ids.append(di)
    #print('text_digt[%d]=%d)'%(di,len(text_digt[di])))
  if len(long_doc_ids)==10:
    break
#print('long_doc_ids:',long_doc_ids)
# Generating validation data
train_gens = []
valid_gens = []
for fi in range(num_files):
  # Get all the bigrams if the document id is not in 
  # the validation document ids
  '''把所有不屬於long_doc_ids的文件bigrams放入train_gens'''
  if fi not in long_doc_ids:
    train_gens.append(DataGeneratorOHE(
            text_digt[fi], train_batchSize, train_n_unroll))
    
    '''把所有屬於long_doc_ids的文件的最後 train_times_per_doc
       個bigrams放入valid_gens當validation data，其他屬於
       long_doc_ids文件的bigrams再加入train_gens當training data
       bigrams放入valid_gens'''
  else:
    '''train_gens再加入非最後train_times_per_doc數的bigrams'''      
    train_gens.append(DataGeneratorOHE(
            text_digt[fi][:-train_times_per_doc],
            train_batchSize, train_n_unroll))
    # Defining the validation data generator
    '''屬於long_doc_ids的文件valid_gens的最後train_times_per_doc
    個bigrams放入valid_gen當validation data'''
    valid_gens.append(DataGeneratorOHE(
            text_digt[fi][-train_times_per_doc:],
            valid_batchSize, valid_n_unroll))
'''train_gens length=100; valid_gens length=10'''   
# In[11]:
session = tf.InteractiveSession()
# Initializing variables
tf.global_variables_initializer().run()
print('\nInitialized')
average_loss = 0
#average_loss_list = 0
''' @@@  Training Phase @@@'''
feed_train = {}
feed_valid = {}
feed_predPhase = {}
for iteration in range(n_iterations):
    print('\n')
    '''train_doc_count(20)隨機產生序列，再取其前
       train_docs_to_use(5)個編號文件使用'''    
    train_zeroState = tf.zeros([train_batchSize, n_hidden],
                     dtype=tf.float32)
    zeroState = train_zeroState    
    print('zeroState_train shape:', zeroState.shape)
    print('outputs shape:', outputs.shape)
    for di in np.random.permutation(
              train_doc_count)[ : train_docs_to_use]:                    
        doc_perplexity = 0        
        #doc_perplexity_list = 0
        '''每次取上述文件train_times_per_doc(100)個input和labels bigram訓練'''
        for doc_iteration_id in range(train_times_per_doc):            
            # Get a set of unrolled batches
            '''之前train_gens只是建立物件，此藉unroll_batches才產生data和label'''
            '''因為函式unroll_batches返回list，要經np.array => shape 
             (num_unroll, train_batchSize, n_inputs)的array,再經np.hstack()
              把shape轉為(train_batchSize, num_unroll, n_inputs)才能
              符合train_gens放到rnn basic cell'''
            unroll_X, unroll_y = train_gens[di].unroll_batches()
            unroll_X_arry = np.array(unroll_X)
            train_dat = np.reshape(np.hstack(unroll_X_arry),
                            (-1, train_n_unroll, n_inputs)) 
            unroll_y_arry = np.array(unroll_y)
            train_lbl = np.reshape(np.hstack(unroll_y_arry),
                            (-1, train_n_unroll, n_outputs))            
            
            '''train_dat.shape(train_batchSize, num_unroll, n_inputs)
               train_lbl.shape(train_batchSize, num_unroll, n_outputs)'''
            #feed_train[batchSize] = train_batchSize
            #feed_train[n_unroll] = train_n_unroll            
            feed_train[dataset]= train_dat
            feed_train[labels] = train_lbl            
            feed_train[batchSize]= train_batchSize
            feed_train[n_unroll] = train_n_unroll            
            # Running the TensorFlow operation
            trainDataSet, _, loss_val, iteration_predictions,\
              iteration_perplexity = session.run(             
               [dataset, rnn_optimizer, loss, predictions,
                perplexity_without_exp],feed_dict= feed_train)
            #print('doc_iteration_id:', doc_iteration_id)                               
            #print('train_loss @ value : ',loss_val)
            #print('train_perplexity_nonexp @ value : ',
            #      iteration_perplexity)
            
            '''!!!!! Test List !!!!!
            print('trainPredictions_list[0] @ shape : ',
                       iteration_predictions_list[0].shape)
            print('train_loss_list @ value : ',loss_val_list)
            print('train_perplexity_nonexp_list @ value : ',
                  iteration_perplexity_list) 
            !!!!!  Test End !!!!!'''
            # Update doc perplexity variable
            doc_perplexity += iteration_perplexity                       
            # Update average iteration perplexity 
            average_loss += iteration_perplexity
            
            '''!!!!!  Test !!!!!
            doc_perplexity_list += iteration_perplexity_list
            average_loss_list += iteration_perplexity_list
             !!!!!  Test End !!!!!'''
            
        '''顯示每個document訓練train_times_per_doc (100)次數後的Perplexity
        di指訓練的文件。新di，doc_perplexity=0，所以只記錄每文件的Perplexity'''        
        print('Document %d iteration %d processed\n Perplexity: %.2f:'
              %(di,iteration+1,np.exp(doc_perplexity/train_times_per_doc)))
        #print('Perplexity_list: %.2f:'
        #      %(np.exp(doc_perplexity_list/train_times_per_doc)))        
        # di為參與訓練的文件  doc_perplexity  
                
        #session.run(reset_train_state_op) 
    
    '''注意：此處average_loss是training phase的平均loss，
            底下的vld_perplexity才是Validation phase的平均loss'''
      
    '''average_loss為train_docs_to_use (5)個文件的平均loss，所以
      要除整個迭代train_docs_to_use*train_times_per_doc再取exp
    '''
    average_loss = average_loss / (train_docs_to_use 
                                 * train_times_per_doc)
      
    print('Average loss at iteration %d: %f' % (iteration+1,
                                         average_loss))
    print('\tPerplexity at iteration %d: %f' %(iteration+1,
                                np.exp(average_loss)))
    print('train dataset shape', trainDataSet.shape)
    train_perplexity_store.append(np.exp(average_loss))
    
    '''!!!!! Test List 
    average_loss_list = average_loss_list / (train_docs_to_use 
                                 * train_times_per_doc)
      
    print('Average loss list at iteration %d: %f' % (iteration+1,
                                         average_loss_list))
    print('\tPerplexity list at iteration %d: %f' %(iteration+1,
                                np.exp(average_loss_list)))  
    average_loss_list = 0
    !!!!!'''
    average_loss = 0 # reset loss
   
    ''' @@@ Validation Phase @@@ '''
   
    # valid_often: How often run validation
    if iteration % valid_often == 0:            
      validLoss = 0 # reset loss            
      # calculate valid perplexity      
      for vld_doc_id in range(len(valid_gens)):  
          '''Remember we process things as bigrams 
             So need to divide by 2'''
          '''在len(valid_gens)(10)範圍，每次取train_times_per_doc//2 
             (50)個 input和labels bigram驗證。
             其中valid_gens的valid_batchSize=1及num_unroll=1，因num_unroll=1
             (1個step)，故unroll_valid_data和unroll_valid_labels list len=1 
             內np array shape(1,544)(valid_batchSize, vocabulary_size) '''
          #print('valid document id:',vld_doc_id)               
          for vld_iteration in range(train_times_per_doc//2):              
            unroll_valid_X,unroll_valid_y = valid_gens[
                    vld_doc_id].unroll_batches()
            unroll_validXArray = np.array(unroll_valid_X)
            valid_dat = np.reshape(np.hstack(unroll_validXArray),
                            (-1, valid_n_unroll, n_inputs))
            unroll_validyArray = np.array(unroll_valid_y)
            valid_lbl = np.reshape(np.hstack(unroll_validyArray),
                            (-1, valid_n_unroll, n_outputs))
            #print('valid_dat:', valid_dat.shape)
            #print('valid_lbl:', valid_lbl.shape)
            feed_valid[dataset]= valid_dat
            feed_valid[labels] = valid_lbl            
            feed_valid[batchSize]= valid_batchSize
            feed_valid[n_unroll] = valid_n_unroll
            vld_loss_val,vld_perplx = session.run(
                    [loss, perplexity_without_exp],
                    feed_dict = feed_valid)
            '''vld_loss_val,vld_perplx = session.run([loss,
                                 perplexity_without_exp],
                                 feed_dict = {dataset: valid_dat,
                                 labels: valid_lbl})'''
            print('valid iteration:',vld_iteration)
            print('valid perplexity nonexp :',vld_perplx)             
            print('valid loss : ',vld_loss_val) 
            
            validLoss += vld_perplx            
          #session.run(reset_valid_state_op)
          valid_zeroState = tf.zeros([train_batchSize, n_hidden],
                     dtype=tf.float32)            
          zeroState = valid_zeroState
          # Reset validation data generator cursor
          valid_gens[vld_doc_id].reset_indices()    
    
      print()
      
      #vld_perplexity要除整個迭代len(valid_gens)*train_times_per_doc//2)
      vld_perplexity = np.exp(validLoss/(len(valid_gens)*
                              train_times_per_doc//2))
      print("Valid Perplexity: %.2f\n"%vld_perplexity)
      valid_perplexity_store.append(vld_perplexity)

      
      ''' @@@ Generating New Text Phase （precdict phase ）@@@'''
      
      # We will be generating one segment having 1000 bigrams
      # Feel free to generate several segments by changing
      # the value of segments_to_generate
      print('Generated Text after epoch %d ... '%iteration)  
      segments_to_generate = 1 # 產生新文章段落數
      chars_in_segment = 1000  # 每段落的字數
      batchSize = predPhase_batchSize
      n_unroll = predPhase_n_unroll
      predPhase_zeroState = tf.truncated_normal(
              [train_batchSize, n_hidden],
              stddev=0.01, dtype=tf.float32)
      zeroState = predPhase_zeroState
      
      print('zeroState_predPhase shape:', zeroState.shape)
      print('predPhase outputs shape:', outputs.shape)
      for _ in range(segments_to_generate):
        print('============== New text Segment ==============')
        # Start with a random word
        '''從text_digt（num_files隨機選擇文件，
        再於0到100位址處隨機選擇1個設為1）選擇1個bigram'''  
        predPhase_word = np.zeros((predPhase_batchSize,
               predPhase_n_unroll, n_inputs),dtype=np.float32) # shape(1, 544)
        predPhase_word[0, 0, text_digt[np.random.randint(
              0, num_files)][np.random.randint(0, 100)]] = 1.0
        print('predPhase_word shape:', predPhase_word.shape)
        '''用reverse_dictionary顯示這隨機的字（bigram）'''
        print("predPhase_word[0]:\n\t",reverse_dictionary[
                  np.argmax(predPhase_word[0,0])], end='')
        
        # Generating words within a segment by feeding in the 
        # previous prediction as the current input in a recursive manner
        '''用這隨機的字預測這段落
        注意：因為資料庫是bigram，所以reverse_dictionary預測
             創造的字也是bigram。在結合列印時用end=''就可組合
             成長字（而長字就用原自典裡bigram的空白來區隔不同字，
             例如：'th','e ','do','g ','is'=>'the dog is' '''
           
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        
        for _ in range(chars_in_segment):   # 預測chars_in_segment個字
          #feed_predPhase[batchSize] = predPhase_batchSize
          #feed_predPhase[n_unroll] = predPhase_n_unroll
          feed_predPhase[dataset]= predPhase_word  
          predPhaseDataSet, predPhase_pred = session.run([dataset,
                predictions], feed_dict = feed_predPhase)
          #print('\npredPhase_pred shape:',predPhase_pred.shape)
          '''上面定義sample()函數回傳，與預測字最高相近vector的index
            python的ravel()與flatten()一樣，降維成一維。'''
          next_ind = sample(predPhase_pred.ravel())          
          predPhase_word = np.zeros((predPhase_batchSize,
               predPhase_n_unroll, n_inputs),dtype=np.float32)
          '''設定預測字的one-hot-encoded'''
          predPhase_word[0, 0, next_ind] = 1.0
          '''因為預測字的vector index已有，故無需np.argmax(predPhase_word[0])]'''          
          print(reverse_dictionary[next_ind],end='')
        
        print('predPhase dataset shape', predPhaseDataSet.shape)
            
        #print("")
        # Reset predict phase state
        predPhase_zeroState = tf.truncated_normal(
              [train_batchSize, n_hidden],
              stddev=0.01, dtype=tf.float32)
        zeroState = predPhase_zeroState
        #session.run(reset_predPhase_state_op)
        print('============================================')
      print("")
    