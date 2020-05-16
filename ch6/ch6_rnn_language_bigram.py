
# coding: utf-8
# In[1_0]:
# ## Recurrent Neural Networks for Language Modeling 
# 
# Recurrent Neural Networks (RNNs) is a powerful family
#  of neural networks that are widely used for sequence
#  modeling tasks (e.g. stock price prediction,
#  language modeling). RNNs ability to exploit temporal
#  dependecies of entities in a sequence makes them powerful.
# In this exercise we will model a RNN and learn tips and
#  tricks to improve the performance.
# 
# In this exercise, we will do the following.
# 1. Create word vectors for a dataset created from stories.
#    available [here](https://www.cs.cmu.edu/~spok/grimmtmp/)
# 2. Train a RNN model on the dataset and use it to output
#    a new story.

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

# In[2_0]:# ## Downloading Data
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
# * `batch_size`: Number of datapoints in a single batch
# * `n_hidden`: Number of hidden neurons in the state

# In[6]:
        
tf.reset_default_graph()

# Number of steps to unroll
'''每個unroll step視為一個cell，其hidden層的output_h也是其state，
   換言之，有兩個同等輸出，output_h用來集合整體sequence的outputs，
   state(next_state）接到下一個cell當輸入(prev_state)用來與W_hh內積'''
num_unroll = 50 

batch_size = 60 # At train time
predPhase_batch_size = 1 # At predict phase time

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
'''train_dataset和train_labels 都是list of Tensor
   每個Tensor shape分別為 (batch_size, n_inputs)及
   (batch_size, n_outputs)，而list共有(num_unroll)組 '''
train_dataset, train_labels = [],[]
for ui in range(num_unroll):
    train_dataset.append(tf.placeholder(tf.float32,
     shape=[batch_size, n_inputs], name='train_dataset_%d'%ui))
    train_labels.append(tf.placeholder(tf.float32,
     shape=[batch_size, n_outputs],name='train_labels_%d'%ui))

# Validation dataset    
valid_dataset = tf.placeholder(tf.float32, 
                 shape=[1,n_inputs],name='valid_dataset')
valid_labels = tf.placeholder(tf.float32,
                 shape=[1,n_outputs],name='valid_labels')

# predict phase dataset
predPhase_dataset = tf.placeholder(tf.float32,
      shape=[predPhase_batch_size, n_inputs],name='predPhase_dataset')

# In[8_0]:
# ## Defining Model Parameters and Other Variables
# Here we define model parameters. First we define 
# three different sets of weights (`W_xh`,`W_hh` and `W_hy`).
# We also define a variable to maintain the hidden state.
# There needs to be three separate variables for the hidden
# state to be used during training(`prev_train_state`),
# validation (`prev_valid_state`) and predict phase (`prev_predPhase_state`).

# In[8]:


# Weights between inputs and h
W_xh = tf.Variable(tf.truncated_normal(
        [n_inputs, n_hidden],stddev=0.02,
       dtype=tf.float32),name='W_xh')

# Weights between h and h
W_hh = tf.Variable(tf.truncated_normal(
        [n_hidden,n_hidden],stddev=0.02,
        dtype=tf.float32),name='W_hh')

# Weights between h and y
W_hy = tf.Variable(tf.truncated_normal(
        [n_hidden,n_outputs],stddev=0.02,
        dtype=tf.float32),name='W_hy')

# Maintain the previous state of hidden nodes
# in an un-trainable variable (Training data)
prev_train_state = tf.Variable(tf.zeros([batch_size, n_hidden],
          dtype=tf.float32),name='train_h',trainable=False)

# Maintain the previous state of hidden nodes in
# an un-trainable variable (Validation data)
prev_valid_state = tf.Variable(tf.zeros([1, n_hidden],
          dtype=tf.float32),name='valid_h',trainable=False)

# Maintain the previous state of hidden nodes in
# predict phase
prev_predPhase_state = tf.Variable(tf.zeros([predPhase_batch_size, n_hidden]
         ,dtype=tf.float32),name='predPhase_h')

# In[9_0]:
# ## Defining Inference of the RNN
# This is the most crucial bit of RNN and what makes
# it different from feed forward networks.
# Here we define operations related to:
# * Calculating training/validation/predict phase hidden outputs
# * Calculating training/validation/predict phase predictions

# In[9]:

# ========================================================================
# Train score (unnormalized) values and predictions (normalized)
y_scores, y_predictions = [],[]

# Appending the calculated output of RNN for 
# each step in the num_unroll steps
outputs = list()
''''''
# This will be iteratively used within num_unroll 
# steps of calculation
#設next_train_state為格式等同prev_train_state的 Tensor
next_train_state = prev_train_state 
# Calculating the output of the RNN for num_unroll steps
# (as required by the truncated BPTT)
'''
1. tf.matmul(tf.concat([train_dataset[ui],next_train_state], 1),
             tf.concat([W_xh, W_hh],0))的效果等於
   tf.matmul(train_dataset[ui], W_xh) + tf.matmul(next_train_state, W_hh)
2. next_train_state = ...,next_train_state...，表這是一個迭代運算，
    前一個cell的(state dot W_hh)再與(train_dataset[ui] dot W_xh)相加
    取tanh得本cell的state
3. next_train_state也就是每cell的hidden層output，所以outputs是
   append(next_train_state)'''
for ui in range(num_unroll):   
        next_train_state = tf.nn.tanh(
          tf.matmul(tf.concat([train_dataset[ui], next_train_state], 1),
                    tf.concat([W_xh, W_hh],0)))             
        outputs.append(next_train_state)

# Get the scores and predictions for all the RNN outputs we 
# produced for num_unroll steps
'''y_scores為list of Tensors（藉由[tf.matmul()]把Tensors list起來），
  其中每個Tensor shape為(batch_size, n_outputs)(60, 544)來自
 (batch_size, n_inputs)●(n_inputs, n_hidden)●(n_hidden, n_outputs)
 ，共有num_unroll(50)個Tensor形成1個list'''
y_scores = [tf.matmul(outputs[ui],W_hy) for
            ui in range(num_unroll)]
#print('y_scores:\n', y_scores)
y_predictions = [tf.nn.softmax(y_scores[ui]) for
            ui in range(num_unroll)]

# We calculate train perplexity with the predictions
# made by the RNN
train_perplexity_without_exp = tf.reduce_sum(
        tf.concat(train_labels,0) * - tf.log(
        tf.concat(y_predictions,0)+1e-10)
        )/(num_unroll*batch_size)

# =========================================================================
# Validation data related inference logic 
# (very similar to the training inference logic)

# Compute the next valid state (only for 1 step)
next_valid_state = tf.nn.tanh(tf.matmul(valid_dataset, W_xh)+
                              tf.matmul(prev_valid_state, W_hh))

# Calculate the prediction using the state output of the RNN
# But before that, assign the latest state output of the RNN
# to the state variable of the validation phase
# So you need to make sure you execute valid_predictions 
# operation to update the validation state
with tf.control_dependencies([tf.assign(prev_valid_state,
                            next_valid_state)]):
    valid_scores = tf.matmul(next_valid_state,W_hy) 
    valid_predictions = tf.nn.softmax(valid_scores)

# Validation data related perplexity
valid_perplexity_without_exp = tf.reduce_sum(
        valid_labels*-tf.log(valid_predictions+1e-10))

# ========================================================================
# predict phase data realted inference logic

# Calculating hidden output for predict phase data
next_predPhase_state = tf.nn.tanh(tf.matmul(predPhase_dataset,W_xh) +
                             tf.matmul(prev_predPhase_state,W_hh))

# Making sure that the predict phase hidden state is updated 
# every time we make a prediction
with tf.control_dependencies([tf.assign(prev_predPhase_state,
                                        next_predPhase_state)]):
    predPhase_prediction = tf.nn.softmax(tf.matmul(
                                     next_predPhase_state, W_hy))

# In[10_0]:
# ## Calculating RNN Loss
# We calculate the training and validation loss of RNN here.
# It's a typical cross entropy loss calculated over all the
#  scores we obtained for training data (`rnn_train_loss`) and 
#  validation data (`rnn_valid_loss`).

# In[10]:


# Here we make sure that before calculating the loss, 
# the state variable is updated with the last RNN
# output state we obtained
with tf.control_dependencies([tf.assign(
                              prev_train_state, next_train_state)]):
    # We calculate the softmax cross entropy for all the 
    # predictions we obtained in all num_unroll steps at once. 
    '''y_scores為list of Tensor單1個list shape為(60, 544) (batch,
    n_outputs) 經tf.concat(y_scores, 0)後成為單一Tensor，它的shape為
    (3200, 544)，因為有50(num_unroll)組Tensor。
    另，tf.concat(train_labels, 0)也從list of Tensor成單一Tensor，
    shape也是(3200, 544)'''
    rnn_train_loss = tf.reduce_mean(
             tf.nn.softmax_cross_entropy_with_logits_v2(
             logits=tf.concat(y_scores, 0),
             labels=tf.concat(train_labels, 0))
             )
    #print('tf.concat(train_labels, 0):\n', tf.concat(train_labels, 0))
# Validation RNN loss    
rnn_valid_loss = tf.reduce_mean(
             tf.nn.softmax_cross_entropy_with_logits_v2(
             logits=valid_scores, labels=valid_labels))

# In[11_0]:
# Defining Learning Rate and the Optimizer with Gradient Clipping
# 
# Here we define the learning rate and the optimizer we're
# going to use. We will be using the Adam optimizer as it
# is one of the best optimizers out there. Furthermore we 
# use gradient clipping to prevent any gradient explosions.

# In[11]:


# Be very careful with the learning rate when using Adam
rnn_optimizer = tf.train.AdamOptimizer(learning_rate=0.001)

# Optimization with graident clipping
gradients, v = zip(*rnn_optimizer.compute_gradients(rnn_train_loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
rnn_optimizer = rnn_optimizer.apply_gradients(zip(gradients,v))

# In[12_0]:
# ## Resetting Operations for Resetting Hidden States
# Sometimes the state variable needs to be reset 
# (e.g.when starting predictions at a beginning of a
# new epoch)

# In[12]:
# Reset the hidden states
reset_train_state_op = tf.assign(prev_train_state,
                     tf.zeros([batch_size, n_hidden],
                     dtype=tf.float32))
reset_valid_state_op = tf.assign(prev_valid_state,
                     tf.zeros([1, n_hidden],
                     dtype=tf.float32))

# Note that we are using small imputations when 
# resetting the predict phase state. 
# As this helps to add more variation to the generated text
reset_predPhase_state_op = tf.assign(prev_predPhase_state,
        tf.truncated_normal([predPhase_batch_size, n_hidden],
        stddev=0.01,dtype=tf.float32))

# In[13_0]:
# ## predPhase Sampling
# We select the word corresponding to the highest 
# index of the prediction vector.  
# We will later see different sampling strategies.

# In[13]:
def sample(distribution):
  #Sample a word from the prediction distribution
    
  best_idx = np.argmax(distribution)
  return best_idx

# In[14_0]:
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

# In[14]:

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

session = tf.InteractiveSession()
# Initializing variables
tf.global_variables_initializer().run()

print('\nInitialized')
average_loss = 0

''' @@@  Generate Training and Validation Data & Label Phase @@@'''
# We use the first 10 documents that has more than 
# (n_iterations+1) * train_times_per_doc bigrams
# for creating the validation dataset

# Identify the first 10 documents following the above condition
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
            text_digt[fi], batch_size, num_unroll))
    
    '''把所有屬於long_doc_ids的文件的最後 train_times_per_doc
       個bigrams放入valid_gens當validation data，其他屬於
       long_doc_ids文件的bigrams再加入train_gens當training data
       bigrams放入valid_gens'''
  else:
    '''train_gens再加入非最後train_times_per_doc數的bigrams'''      
    train_gens.append(DataGeneratorOHE(
            text_digt[fi][:-train_times_per_doc],
            batch_size, num_unroll))
    # Defining the validation data generator
    '''屬於long_doc_ids的文件valid_gens的最後train_times_per_doc
    個bigrams放入valid_gen當validation data'''
    valid_gens.append(DataGeneratorOHE(
            text_digt[fi][-train_times_per_doc:],
            1, 1))
'''train_gens length=100; valid_gens length=10'''   

feed_dict = {}

''' @@@  Training Phase @@@'''
for iteration in range(n_iterations):
    print('\n')
    '''train_doc_count(20)隨機產生序列，再取其前
       train_docs_to_use(5)個編號文件使用'''
    for di in np.random.permutation(
              train_doc_count)[ : train_docs_to_use]:                    
        doc_perplexity = 0
        '''每次取上述文件train_times_per_doc(100)個input和labels bigram訓練'''
        for doc_iteration_id in range(train_times_per_doc):            
            # Get a set of unrolled batches
            '''之前train_gens只是建立物件，此藉unroll_batches才產生data和label'''
            unroll_X, unroll_y = train_gens[di].unroll_batches()
            '''train_gens[di]傳入的batch_size=60,num_unroll=50
              unroll_X和unroll_y的 list len=50 (num_unroll)
              list內np array shape(60, 544)(batch_size, vocabulary_size)'''
                        
            # Populate the feed dict by using each of the data batches
            # present in the unrolled data
            '''feed_dict有data和label又train_dataset和train_labels
            都是list of Tensors 所以又要分不同組Tensor所以要zip再enumerate編號'''
            for ui,(dat,lbl) in enumerate(zip(unroll_X,unroll_y)):  # ui: 0~49          
                feed_dict[train_dataset[ui]]= dat
                feed_dict[train_labels[ui]] = lbl            
              
            # Running the TensorFlow operation
            _, loss_val,iteration_predictions,y_scores_v,\
            _,iteration_labels,iteration_perplexity =\
              session.run([rnn_optimizer,rnn_train_loss, 
                          y_predictions,y_scores,
                          train_dataset,train_labels,
                          train_perplexity_without_exp],
                          feed_dict=feed_dict)
            
            # Update doc perplexity variable
            doc_perplexity += iteration_perplexity
            # Update average iteration perplexity 
            average_loss += iteration_perplexity
        '''顯示每個document訓練train_times_per_doc (100)次數後的Perplexity
        di指訓練的文件。新di，doc_perplexity=0，所以只記錄每文件的Perplexity'''        
        print('Document %d iteration %d processed (Perplexity: %.2f).'
              %(di,iteration+1,np.exp(doc_perplexity/train_times_per_doc)))
        # di為參與訓練的文件  doc_perplexity  
        
        # Reset hidden state after processing a single document
        # It's still questionable if this adds value in 
        #   terms of learning
        # One hand it's intuitive to reset the state when        
        #   learning a new document
        # On the other hand this approach creates a bias for
        #   the state to be zero
        # We encourage the reader to investigate further the
        #   effect of resetting the state
        session.run() 
    
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
    train_perplexity_store.append(np.exp(average_loss))
      
    average_loss = 0 # reset loss

    ''' @@@ Validation Phase @@@ '''
   
    # valid_often: How often run validation
    if iteration % valid_often == 0:            
      valid_loss = 0 # reset loss
      
      # calculate valid perplexity      
      for vld_doc_id in range(len(valid_gens)):  
          '''Remember we process things as bigrams 
             So need to divide by 2'''
          '''在len(valid_gens)(10)範圍，每次取train_times_per_doc//2 
             (50)個 input和labels bigram驗證。
             其中valid_gens的batch_size=1及num_unroll=1，因num_unroll=1
             (1個step)，故unroll_valid_data和unroll_valid_labels list len=1 
             內np array shape(1,544)(batch_size, vocabulary_size) '''             
          for vld_iteration in range(train_times_per_doc//2):              
            unroll_valid_data,unroll_valid_labels = valid_gens[
                    vld_doc_id].unroll_batches()
            
            # Run validation phase related TensorFlow operations
            '''因list len=1，所以無需像上述training data要再zip及enumerate'''
            vld_perplx = session.run(
                valid_perplexity_without_exp,
                feed_dict = {valid_dataset:unroll_valid_data[0],
                             valid_labels: unroll_valid_labels[0]})            

            valid_loss += vld_perplx
            
          session.run(reset_valid_state_op)
          # Reset validation data generator cursor
          valid_gens[vld_doc_id].reset_indices()    
    
      print()
      '''vld_perplexity要除整個迭代len(valid_gens)*train_times_per_doc//2)'''
      vld_perplexity = np.exp(valid_loss/(len(valid_gens)*
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
    
      for _ in range(segments_to_generate):
        print('============== New text Segment ==============')
        # Start with a random word
        '''從text_digt（num_files隨機選擇文件，
        再於0到100位址處隨機選擇1個設為1）選擇1個bigram'''
        predPhase_word = np.zeros((1, n_inputs),dtype=np.float32) # shape(1, 544)
        predPhase_word[0, text_digt[np.random.randint(
              0, num_files)][np.random.randint(0, 100)]] = 1.0
        '''用reverse_dictionary顯示這隨機的字（bigram）'''
        print("predPhase_word[0]:\n\t",reverse_dictionary[
                  np.argmax(predPhase_word[0])], end='')
        
        # Generating words within a segment by feeding in the 
        # previous prediction as the current input in a recursive manner
        '''用這隨機的字預測這段落
        注意：因為資料庫是bigram，所以reverse_dictionary預測
             創造的字也是bigram。在結合列印時用end=''就可組合
             成長字（而長字就用原自典裡bigram的空白來區隔不同字，
             例如：'th','e ','do','g ','is'=>'the dog is' '''
              
        print('\n~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
        for _ in range(chars_in_segment):   # 預測chars_in_segment個字
          predPhase_pred = session.run(predPhase_prediction,
                      feed_dict = {predPhase_dataset:predPhase_word})  
          '''上面定義sample()函數回傳，與預測字最高相近vector的index
            python的ravel()與flatten()一樣，降維成一維。'''
          next_ind = sample(predPhase_pred.ravel())
          predPhase_word = np.zeros((1, n_inputs),dtype=np.float32)
          '''設定預測字的one-hot-encoded'''
          predPhase_word[0, next_ind] = 1.0
          '''因為預測字的vector index已有，故無需np.argmax(predPhase_word[0])]'''          
          print(reverse_dictionary[next_ind],end='')
        
        print("")
        # Reset predict phase state
        session.run(reset_predPhase_state_op)
        print('============================================')
      print("")

# In[15_0]:
# ## Plotting Perplexity of RNN
# 
# After training the RNN, we plot the train and 
# valid perplexity side by side

# In[15]:


x_axis = np.arange(len(train_perplexity_store[1:25]))
f,(ax1,ax2)=pylab.subplots(1,2,figsize=(18,6))

ax1.plot(x_axis,train_perplexity_store[1:25],label='Train')
ax2.plot(x_axis, valid_perplexity_store[1:25], label='Valid')

pylab.title('Train and Valid Perplexity over Time',fontsize=24)
ax1.set_title('Train Perplexity',fontsize=20)
ax2.set_title('Valid Perplexity',fontsize=20)
ax1.set_xlabel('Epoch',fontsize=20)
ax2.set_xlabel('Epoch',fontsize=20)
pylab.savefig('RNN_perplexity.png')
pylab.show()

# In[16_0]:
# ## RNN-CF - RNN with Contextual Features
# 
# Here we implement an extension of RNN which is described
#  in this [paper](https://arxiv.org/pdf/1412.7753.pdf).
# RNN-CF differs from a standard RNN as the RNN-CF has
#  two different states and one state is specifically 
#  designed to retain long term memory.

# ## Defining Hyperparameters
# 
# Here we define several hyperparameters required.
# * `num_unroll`: Number of steps we unroll 
#                 over time during optimizing
# * `batch_size`: Number of datapoints in a single batch
# * `hidden`: Number of hidden neurons in the state
# * `hidden_context`: Number of hidden neurons in
#                     the context vector
# * `alpha` : RNN-CF specific hyperparameter

# In[16]:

tf.reset_default_graph()

# Number of steps to unroll
num_unroll = 50

batch_size = 60 # At train time
predPhase_batch_size = 1 # At predict phase time

# Number of hidden neurons in each state
n_hidden = 64
hidden_context = 64

alpha = 0.9

# Input size and output Size
n_inputs,n_outputs = vocabulary_size,vocabulary_size


# ## Defining Inputs and Outputs
# Here we define training inputs (`train_dataset`)
# and outputs (`train_labels`), validation inputs 
# (`valid_dataset`) and outputs (`valid_labels`)
#  and predict phase inputs (`predPhase_dataset`).

# In[17]:

# Train dataset
# We use unrolling over time
train_dataset, train_labels = [],[]
for ui in range(num_unroll):
    train_dataset.append(tf.placeholder(tf.float32,
      shape=[batch_size,n_inputs],name='train_dataset_%d'%ui))
    train_labels.append(tf.placeholder(tf.float32,
      shape=[batch_size,n_outputs],name='train_labels_%d'%ui))

# Validation dataset 
valid_dataset = tf.placeholder(tf.float32,
               shape=[1,n_inputs],name='valid_dataset')
valid_labels = tf.placeholder(tf.float32,
               shape=[1,n_outputs],name='valid_labels')

# predict phase dataset
predPhase_dataset = tf.placeholder(tf.float32,
   shape=[predPhase_batch_size,n_inputs],name='save_predPhase_dataset')

# In[18_0]:
# ## Defining Model Parameters and Other Variables
# Here we define model parameters. 
# First we define `A`,`B`,`R`,`P`,`U` and `V`.
#  We also define a variable to maintain the hidden state.
#  Each phase of training/validation/predict phase will have 
# two state variables. For example for training we have
# `prev_train_state` and `prev_train_s`.

# In[18]:

# Weights between inputs and h
A = tf.Variable(tf.truncated_normal([n_inputs,n_hidden],
                stddev=0.02,dtype=tf.float32),name='W_xh')
B = tf.Variable(tf.truncated_normal([n_inputs,hidden_context],
                stddev=0.02,dtype=tf.float32),name='W_xs')

# Weights between h and h
R = tf.Variable(tf.truncated_normal([n_hidden,n_hidden],
                stddev=0.02,dtype=tf.float32),name='W_hh')
P = tf.Variable(tf.truncated_normal([hidden_context,n_hidden],
                stddev=0.02,dtype=tf.float32),name='W_ss')

# Weights between h and y
U = tf.Variable(tf.truncated_normal([n_hidden,n_outputs],
                stddev=0.02,dtype=tf.float32),name='W_hy')
V = tf.Variable(tf.truncated_normal([hidden_context, n_outputs],
                stddev=0.02,dtype=tf.float32),name='W_sy')

# State variables for training data
prev_train_state = tf.Variable(tf.zeros([batch_size,n_hidden],
            dtype=tf.float32),name='train_h',trainable=False)
prev_train_s = tf.Variable(tf.zeros([batch_size,hidden_context],
            dtype=tf.float32),name='train_s',trainable=False)

# State variables for validation data
prev_valid_state = tf.Variable(tf.zeros([1,n_hidden],
            dtype=tf.float32),name='valid_h',trainable=False)
prev_valid_s = tf.Variable(tf.zeros([1,hidden_context],
            dtype=tf.float32),name='valid_s',trainable=False)

# State variables for predict phase data
prev_predPhase_state = tf.Variable(tf.zeros([predPhase_batch_size,n_hidden],
            dtype=tf.float32),name='predPhase_h')
prev_predPhase_s = tf.Variable(
                tf.zeros([predPhase_batch_size,hidden_context],
                dtype=tf.float32),name='predPhas_s')

# In[19_0]:
# ## Defining Inference of the RNN
# This is the most crucial bit of RNN and what makes
#  it different from feed forward networks. 
# Here we define operations related to:
# * Calculating training/validation/predict phase hidden outputs (h and s)
# * Calculating training/validation/predict phase predictions

# In[19]:

# ===============================================================
# Train score(unnormalized) values and predictions(normalized)
y_scores, y_predictions = [],[]

# These will be iteratively used within 
#  num_unroll steps of calculation
next_h_state = prev_train_state
next_s_state = prev_train_s

# Appending the calculated state outputs of RNN
#  for each step in the num_unroll steps
next_h_states_unrolled, next_s_states_unrolled = [],[]

# Calculating the output of the RNN for num_unroll steps
# (as required by the truncated BPTT)
for ui in range(num_unroll):      
    next_h_state = tf.nn.tanh(
        tf.matmul(tf.concat([train_dataset[ui],
                  prev_train_state,prev_train_s],1),
                  tf.concat([A,R,P],0))
    )    
    next_s_state = (1-alpha) * tf.matmul(train_dataset[ui], B 
                    ) + alpha * next_s_state
    next_h_states_unrolled.append(next_h_state)
    next_s_states_unrolled.append(next_s_state)

# Get the scores and predictions for all the RNN outputs
#  we produced for num_unroll steps
y_scores = [tf.matmul(next_h_states_unrolled[ui],U) + 
            tf.matmul(next_s_states_unrolled[ui],V) 
             for ui in range(num_unroll)]
y_predictions = [tf.nn.softmax(y_scores[ui]) for ui in
                 range(num_unroll)]

# We calculate train perplexity with the 
#  predictions made by the RNN
train_perplexity_without_exp = tf.reduce_sum(
        tf.concat(train_labels,0) * - tf.log(
        tf.concat(y_predictions,0)+1e-10))/(
        num_unroll*batch_size)

# ========================================================================
# Validation data related inference logic 
# (very similar to the training inference logic)

# Compute the next valid state (only for 1 step)
next_valid_s_state = (1-alpha) * tf.matmul(valid_dataset, B
                      ) + alpha * prev_valid_s
next_valid_h_state = tf.nn.tanh(tf.matmul(valid_dataset,A) + 
                                tf.matmul(prev_valid_s, P) +
                                tf.matmul(prev_valid_state,R))


# Calculate the prediction using the state output of the RNN
# But before that, assign the latest state output of the RNN
# to the state variable of the validation phase
# So you need to make sure you execute rnn_valid_loss operation
# To update the validation state
with tf.control_dependencies(
        [tf.assign(prev_valid_s,next_valid_s_state),
         tf.assign(prev_valid_state,next_valid_h_state)]):        
    valid_scores = tf.matmul(prev_valid_state, U) + tf.matmul(prev_valid_s, V) 
    valid_predictions = tf.nn.softmax(valid_scores)
        
# Validation data related perplexity
valid_perplexity_without_exp = tf.reduce_sum(
        valid_labels * -tf.log(valid_predictions + 1e-10))

# ========================================================================
# predict phase data realted inference logic

# Calculating hidden output for predict phase data
next_predPhase_s = (1-alpha) * tf.matmul(
              predPhase_dataset, B) + alpha * prev_predPhase_s
                         
next_predPhase_h = tf.nn.tanh(
    tf.matmul(predPhase_dataset,A) + tf.matmul(prev_predPhase_s,P) + 
    tf.matmul(prev_predPhase_state, R))
                         

# Making sure that the predict phase hidden state is updated 
# every time we make a prediction
with tf.control_dependencies(
        [tf.assign(prev_predPhase_s,next_predPhase_s),
         tf.assign(prev_predPhase_state,next_predPhase_h)]):
    predPhase_prediction = tf.nn.softmax(
        tf.matmul(prev_predPhase_state,U) + tf.matmul(prev_predPhase_s,V)
    ) 

# In[20_0]:
# ## Calculating RNN Loss
# We calculate the training and validation loss of RNN here.
#  It's a typical cross entropy loss calculated over all
#  the scores we obtained for training data (`rnn_train_loss`)
#  and validation data (`rnn_valid_loss`).

# In[20]:


# Here we make sure that before calculating the loss, the state 
# variables are updated with the last RNN output state we obtained.
with tf.control_dependencies([
                tf.assign(prev_train_s, next_s_state),
                tf.assign(prev_train_state,next_h_state)]):
    rnn_train_loss = tf.reduce_mean(
               tf.nn.softmax_cross_entropy_with_logits_v2(
               logits = tf.concat(y_scores,0),
               labels = tf.concat(train_labels,0)))
    
        
rnn_valid_loss = tf.reduce_mean(
                 tf.nn.softmax_cross_entropy_with_logits_v2(
                 logits=valid_scores, labels=valid_labels))

# In[21_0]:
# Defining Learning Rate and the Optimizer with Gradient Clipping 
# 
# Here we define the learning rate and the optimizer
#  we're going to use. We will be using the Adam 
#  optimizer as it is one of the best optimizers out there.
# Furthermore we use gradient clipping to prevent any 
#  gradient explosions.

# In[21]:


rnn_optimizer = tf.train.AdamOptimizer(learning_rate=.001)

gradients, v = zip(*rnn_optimizer.compute_gradients(rnn_train_loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
rnn_optimizer = rnn_optimizer.apply_gradients(
        zip(gradients, v))


# ## Resetting Operations for Resetting Hidden States
# Sometimes the state variable needs to be reset
# (e.g. when starting predictions at a beginning of a new epoch)

# In[22]:


reset_prev_train_state_op = tf.assign(
        prev_train_state, tf.zeros([batch_size,n_hidden],
                               dtype=tf.float32))
reset_prev_train_s_op = tf.assign(
        prev_train_s,tf.zeros([batch_size,hidden_context],
                              dtype=tf.float32))

reset_valid_state_op = tf.assign(
        prev_valid_state,tf.zeros([1, n_hidden],
                              dtype=tf.float32))
reset_valid_s_op = tf.assign(
        prev_valid_s,tf.zeros([1, hidden_context],
                              dtype=tf.float32))

# Input the predict phase states with noise
reset_predPhase_state_op = tf.assign(
        prev_predPhase_state, tf.truncated_normal(
            [predPhase_batch_size,n_hidden], stddev=0.01,
            dtype=tf.float32))
reset_predPhase_s_op = tf.assign(
        prev_predPhase_s,tf.truncated_normal(
            [predPhase_batch_size, hidden_context],stddev=0.01,
            dtype=tf.float32))

# In[23_0]:
# ## Running the RNN-CF to Generate Text
# 
# Here we train the RNN on the available data and generate
#  text using the trained RNN for several iterations.
# First we create a validation set by extracting text
#  snippets (that are not present in training data) from
#  longer documents. Then at each training iteration,
#  we train the RNN on several randomly picked documents.
# From each document we extract text for `train_times_per_doc`
#  iterations. We also report the train and validation 
#  perplexities at the end of each iteration. 
# Finally, predicting (predict phase) the RNN by asking it to generate some new
#  text starting from a randomly picked bigram.

# In[23]:


n_iterations = 26 # Number of iterations we run the algorithm for

# How many training times are performed for each document
#  in a single iteration
train_times_per_doc = 100 

# How often we run validation
valid_often = 1

# We run training documents with this set to both 20 and 100.
train_doc_count = 100
train_docs_to_use = 10 # Number of docs we use in a single iteration

# Store the training and validation perplexity at each iteration
cf_valid_perplexity_store = []
cf_train_perplexity_store = []

session = tf.InteractiveSession()
# Initializing variables
tf.global_variables_initializer().run()

print('Initialized')
average_loss = 0

# We use the first 10 documents that has 
# more than (n_iterations+1)*train_times_per_doc bigrams 
# for creating the validation dataset

# Identify the first 10 documents following the 
#  above condition
long_doc_ids = []
for di in range(num_files):
  if len(text_digt[di])>(n_iterations+1)*train_times_per_doc:
    long_doc_ids.append(di)
  if len(long_doc_ids)==10:
    break

# Generating validation data
train_gens = []
valid_gens = []
for fi in range(num_files):  
  # Get all the bigrams if the document id is not in the 
  # validation document ids
  if fi not in long_doc_ids:
    train_gens.append(DataGeneratorOHE(
                text_digt[fi], batch_size, num_unroll))
  # if the document is in the validation doc ids, only 
  #  get up to the 
  # last train_times_per_doc bigrams and use the last 
  #  train_times_per_doc bigrams as validation data
  else:
    train_gens.append(DataGeneratorOHE(
            text_digt[fi][: -train_times_per_doc],
                          batch_size, num_unroll))
    # Defining the validation data generator
    valid_gens.append(DataGeneratorOHE(
            text_digt[fi][-train_times_per_doc:],1,1))

feed_dict={}
for iteration in range(n_iterations):
    print('\n')
    for di in np.random.permutation(
            train_doc_count)[:train_docs_to_use]:                    
        doc_perplexity = 0
        for doc_iteration_id in range(train_times_per_doc):
            
            # Get a set of unrolled batches
            unroll_X, unroll_y = train_gens[di].unroll_batches()
            
            # Populate the feed dict by using each of the data batches
            # present in the unrolled data
            for ui,(dat,lbl) in enumerate(zip(unroll_X,unroll_y)):            
                feed_dict[train_dataset[ui]]= dat
                feed_dict[train_labels[ui]] = lbl
            
            # Running the TensorFlow operations
            _, l, _, _, _, perp = session.run(
                [rnn_optimizer, rnn_train_loss, y_predictions,
                 train_dataset, train_labels,
                 train_perplexity_without_exp], 
                 feed_dict=feed_dict)
            
            # Update doc_perpelxity variable
            doc_perplexity += perp
            
            # Update the average_loss variable
            average_loss += perp
            
        print('Document %d iteration %d processed (Perplexity: %.2f).'
              %(di, iteration+1, np.exp(doc_perplexity / (
                      train_times_per_doc)))
             )

        # resetting hidden state after processing a single document
        # It's still questionable if this adds value
        #  in terms of learning
        # One one hand it's intuitive to reset the state 
        #  when learning a new document
        # On the other hand this approach creates a bias
        #  for the state to be zero
        # We encourage the reader to investigate further
        #  the effect of resetting the state
        session.run(
            [reset_prev_train_state_op, reset_prev_train_s_op]) 
        # resetting hidden state for each document
    
    # Validation phase
    if iteration % valid_often == 0:
      
      # Compute the average validation perplexity
      average_loss = average_loss / (
              train_docs_to_use * train_times_per_doc * 
              valid_often)
      
      # Print losses
      print('Average loss at iteration %d: %f' % (
              iteration+1, average_loss))
      print('\tPerplexity at iteration %d: %f' %(
              iteration+1, np.exp(average_loss)))
    
      cf_train_perplexity_store.append(np.exp(average_loss))
      average_loss = 0 # reset loss
      valid_loss = 0 # reset loss
      
      # calculate valid perplexity
      for vld_doc_id in range(10):
          # Remember we process things as bigrams
          # So need to divide by 2
          for vld_iteration in range(train_times_per_doc//2):
            unroll_valid_data,unroll_valid_labels = valid_gens[
                    vld_doc_id].unroll_batches()        

            # Run validation phase related TensorFlow operations       
            vld_perplx = session.run(
                valid_perplexity_without_exp,
                feed_dict = {valid_dataset: unroll_valid_data[0],
                             valid_labels: unroll_valid_labels[0]}
            )

            valid_loss += vld_perplx
            
          session.run([reset_valid_state_op, reset_valid_s_op])
          # Reset validation data generator cursor
          valid_gens[vld_doc_id].reset_indices()    
    
      print()
      vld_perplexity = np.exp(valid_loss /(
              train_times_per_doc * 10.0 //2 ))
      print("Valid Perplexity: %.2f\n"%vld_perplexity)
      cf_valid_perplexity_store.append(vld_perplexity)
      
        
      # Generating new text ...
      # We will be generating one segment having 1000 bigrams
      # Feel free to generate several segments by changing
      # the value of segments_to_generate
      print('Generated Text after epoch %d ... '%iteration)  
      segments_to_generate = 1
      chars_in_segment = 1000
    
      for _ in range(segments_to_generate):
        print('============= New text Segment ==============')
        # Start with a random word
        predPhase_word = np.zeros((1,n_inputs),dtype=np.float32)
        predPhase_word[0, text_digt[np.random.randint(0, num_files
                )][np.random.randint(0,100)]] = 1.0
        print("\t",reverse_dictionary[
                np.argmax(predPhase_word[0])], end='')
        
        # Generating words within a segment by feeding in the previous
        #  prediction as the current input in a recursive manner.
        for _ in range(chars_in_segment):    
          predPhase_pred = session.run(predPhase_prediction,
                      feed_dict = {predPhase_dataset:predPhase_word})  
          next_ind = sample(predPhase_pred.ravel())
          predPhase_word = np.zeros((1,n_inputs),dtype=np.float32)
          predPhase_word[0,next_ind] = 1.0
          print(reverse_dictionary[next_ind],end='')
        
        print("")
        # Reset predict phase state
        session.run([reset_predPhase_state_op, reset_predPhase_s_op])
      print("")


# In[24]:



x_axis = np.arange(len(train_perplexity_store[1:25]))
f,(ax1,ax2)=pylab.subplots(1,2,figsize=(18,6))

ax1.plot(x_axis,train_perplexity_store[1:25],
         label='RNN',linewidth=2,linestyle='--')
ax1.plot(x_axis,cf_train_perplexity_store[1:25],
         label='RNN-CF',linewidth=2)
ax2.plot(x_axis, valid_perplexity_store[1:25],
         label='RNN',linewidth=2,linestyle='--')
ax2.plot(x_axis, cf_valid_perplexity_store[1:25],
         label='RNN-CF',linewidth=2)
ax1.legend(loc=1, fontsize=20)
ax2.legend(loc=1, fontsize=20)
pylab.title('Train and Valid Perplexity over Time (RNN vs RNN-CF)',
            fontsize=24)
ax1.set_title('Train Perplexity',fontsize=20)
ax2.set_title('Valid Perplexity',fontsize=20)
ax1.set_xlabel('Epoch',fontsize=20)
ax2.set_xlabel('Epoch',fontsize=20)
pylab.savefig('RNN_perplexity_cf.png')
pylab.show()

