
# coding: utf-8

# ## Recurrent Neural Networks for Language Modeling 
# 
# Recurrent Neural Networks (RNNs) is a powerful family of neural networks that are widely used for sequence modeling tasks (e.g. stock price prediction, language modeling). RNNs ability to exploit temporal dependecies of entities in a sequence makes them powerful. In this exercise we will model a RNN and learn tips and tricks to improve the performance.
# 
# In this exercise, we will do the following.
# 1. Create word vectors for a dataset created from stories available at http://clarkesworldmagazine.com/
# 2. Train a RNN model on the dataset and use it to output a new story

# In[1]:


# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
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


# ## Downloading Data
# 
# Downloading stories if not present in disk. There should be 100 files ('stories/001.txt','stories/002.txt', ...)

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


# ## Reading data
# Data will be stored in a list of lists where the each list represents a document and document is a list of words. We will then break the text into bigrams

# In[3]:


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
    two_grams = [''.join(chars[ch_i:ch_i+2]) for ch_i in range(0,len(chars)-2,2)]
    documents.append(two_grams)
    print('Data size (Characters) (Document %d) %d' %(i,len(two_grams)))
    print('Sample string (Document %d) %s'%(i,two_grams[:50]))


# ## Building the Dictionaries (Bigrams)
# Builds the following. To understand each of these elements, let us also assume the text "I like to go to school"
# 
# * `dictionary`: maps a string word to an ID (e.g. {I:0, like:1, to:2, go:3, school:4})
# * `reverse_dictionary`: maps an ID to a string word (e.g. {0:I, 1:like, 2:to, 3:go, 4:school}
# * `count`: List of list of (word, frequency) elements (e.g. [(I,1),(like,1),(to,2),(go,1),(school,1)]
# * `data` : Contain the string of text we read, where string words are replaced with word IDs (e.g. [0, 1, 2, 3, 2, 4])
# 
# It also introduces an additional special token `UNK` to denote rare words to are too rare to make use of.

# In[4]:




def build_dataset(documents):
    chars = []
    # This is going to be a list of lists
    # Where the outer list denote each document
    # and the inner lists denote words in a given document
    data_list = []
  
    for d in documents:
        chars.extend(d)
    print('%d Characters found.'%len(chars))
    count = []
    # Get the bigram sorted by their frequency (Highest comes first)
    count.extend(collections.Counter(chars).most_common())
    
    # Create an ID for each bigram by giving the current length of the dictionary
    # And adding that item to the dictionary
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
            
        data_list.append(data)
        
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
    return data_list, count, dictionary, reverse_dictionary

global data_list, count, dictionary, reverse_dictionary,vocabulary_size

# Print some statistics about data
data_list, count, dictionary, reverse_dictionary = build_dataset(documents)
print('Most common words (+UNK)', count[:5])
print('Least common words (+UNK)', count[-15:])
print('Sample data', data_list[0][:10])
print('Sample data', data_list[1][:10])
print('Vocabulary: ',len(dictionary))
vocabulary_size = len(dictionary)
del documents  # To reduce memory.


# ## Generating Batches of Data
# The following object generates a batch of data which will be used to train the RNN. More specifically the generator breaks a given sequence of words into `batch_size` segments. We also maintain a cursor for each segment. So whenever we create a batch of data, we sample one item from each segment and update the cursor of each segment. 

# In[5]:


class DataGeneratorOHE(object):
    
    def __init__(self,text,batch_size,num_unroll):
        # Text where a bigram is denoted by its ID
        self._text = text
        # Number of bigrams in the text
        self._text_size = len(self._text)
        # Number of datapoints in a batch of data
        self._batch_size = batch_size
        # Num unroll is the number of steps we unroll the RNN in a single training step
        # This relates to the truncated backpropagation we discuss in Chapter 6 text
        self._num_unroll = num_unroll
        # We break the text in to several segments and the batch of data is sampled by
        # sampling a single item from a single segment
        self._segments = self._text_size//self._batch_size
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]
        
    def next_batch(self):
        '''
        Generates a single batch of data
        '''
        # Train inputs (one-hot-encoded) and train outputs (one-hot-encoded)
        batch_data = np.zeros((self._batch_size,vocabulary_size),dtype=np.float32)
        batch_labels = np.zeros((self._batch_size,vocabulary_size),dtype=np.float32)
        
        # Fill in the batch datapoint by datapoint
        for b in range(self._batch_size):
            # If the cursor of a given segment exceeds the segment length
            # we reset the cursor back to the beginning of that segment
            if self._cursor[b]+1>=self._text_size:
                self._cursor[b] = b * self._segments
            
            # Add the text at the cursor as the input
            batch_data[b,self._text[self._cursor[b]]] = 1.0
            # Add the preceding bigram as the label to be predicted
            batch_labels[b,self._text[self._cursor[b]+1]]= 1.0                       
            # Update the cursor
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
        
        return unroll_data, unroll_labels
    
    def reset_indices(self):
        '''
        Used to reset all the cursors if needed
        '''
        self._cursor = [offset * self._segments for offset in range(self._batch_size)]
        
# Running a tiny set to see if things are correct
dg = DataGeneratorOHE(data_list[0][25:50],5,5)
u_data, u_labels = dg.unroll_batches()

# Iterate through each data batch in the unrolled set of batches
for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):   
    print('\n\nUnrolled index %d'%ui)
    dat_ind = np.argmax(dat,axis=1)
    lbl_ind = np.argmax(lbl,axis=1)
    print('\tInputs:')
    for sing_dat in dat_ind:
        print('\t%s (%d)'%(reverse_dictionary[sing_dat],sing_dat),end=", ")
    print('\n\tOutput:')
    for sing_lbl in lbl_ind:        
        print('\t%s (%d)'%(reverse_dictionary[sing_lbl],sing_lbl),end=", ")


# ## Recurrent Neural Network
# Here we implement and train our recurrent model that will take an output a new story

# ## Defining Hyperparameters
# 
# Here we define several hyperparameters required.
# * `num_unroll`: Number of steps we unroll over time during optimizing
# * `batch_size`: Number of datapoints in a single batch
# * `hidden_1`: Number of hidden neurons in the state

# In[6]:


tf.reset_default_graph()

num_unroll = 50
batch_size = 64
test_batch_size = 1

hidden_sizes = [128,64,32]
scopes = ['first','second','third']
input_sizes = [vocabulary_size, 128, 64]
out_size = vocabulary_size


# ## Defining Inputs and Outputs
# Here we define training inputs (`train_dataset`) and outputs (`train_labels`), validation inputs (`valid_dataset`) and outputs (`valid_labels`) and test inputs (`test_dataset`).

# In[7]:


# Train dataset
# We use unrolling over time
train_dataset, train_labels = [],[]
for ui in range(num_unroll):
    train_dataset.append(tf.placeholder(tf.float32, shape=[batch_size,input_sizes[0]],name='train_dataset_%d'%ui))
    train_labels.append(tf.placeholder(tf.float32, shape=[batch_size,out_size],name='train_labels_%d'%ui))

# Validation dataset    
valid_dataset = tf.placeholder(tf.float32, shape=[1,input_sizes[0]],name='valid_dataset')
valid_labels = tf.placeholder(tf.float32, shape=[1,out_size],name='valid_labels')

# Test dataset
test_dataset = tf.placeholder(tf.float32, shape=[test_batch_size,input_sizes[0]],name='save_test_dataset')


# ## Defining Model Parameters and Other Variables
# Here we define model parameters. First we define two sets of weights (`W_xh` and `W_hh`) for each layer and a final output layer (`W_hy`). We also define a variable to maintain the hidden state. There needs to be three separate variables for the hidden state to be used during training(`train_h`), validation (`valid_h`) and testing (`test_h`) for each layer.

# In[8]:


# We will use variable scoping to define variables in multi layer RNN

for scope, h, i in zip(scopes,hidden_sizes, input_sizes):
    with tf.variable_scope(scope):
        print('Weights shape: ',scope,'/W_xh',[i,h])
        print('Weights shape: ',scope,'/W_hh',[h,h])
        # Weights between inputs and h1
        tf.get_variable('W_xh', shape=[i, h], initializer=tf.truncated_normal_initializer(stddev=0.02))

        # Weights between h1 and h1
        tf.get_variable('W_hh',shape=[h,h], initializer = tf.truncated_normal_initializer(stddev=0.02))

        # Maintain the previous state of hidden nodes in an un-trainable variable (Training data)
        tf.get_variable('train_h',shape=[batch_size,h], initializer=tf.zeros_initializer(), trainable=False)
        # Maintain the previous state of hidden nodes in an un-trainable variable (Validation data)
        tf.get_variable('valid_h',shape=[1,h], initializer=tf.zeros_initializer(), trainable=False)
        # Test state
        tf.get_variable('test_h',shape=[1,h], initializer=tf.zeros_initializer(), trainable=False)
        
# Weights between last state and y
with tf.variable_scope('out'):
    tf.get_variable('W_hy',shape=[hidden_sizes[-1], out_size], initializer=tf.truncated_normal_initializer(stddev=0.02))


# ## Defining Inference of the RNN
# This is the most crucial bit of RNN and what makes it different from feed forward networks. Here we define operations related to:
# * Define RNN computations as a function `rnn_cell`
# * Calculating training/validation/test hidden outputs
# * Calculating training/validation/test predictions

# In[9]:


def rnn_cell(scope, x, h_minus_1):
    '''
    Define computations of the RNN cell
    '''
    with tf.variable_scope(scope, reuse=True):
        W_xh, W_hh = tf.get_variable('W_xh'), tf.get_variable('W_hh') 
        h = tf.nn.tanh(tf.matmul(tf.concat([x, h_minus_1],1), tf.concat([W_xh, W_hh],0)))
        
        return h


# ===============================================================================
# Train score (unnormalized) values and predictions (normalized)
y_scores, y_predictions = [],[]

# Setting the initial state to get the current state of the RNN
# training ,validation and testing phases
next_state_h, next_valid_state_h, next_test_state_h = [],[],[]
for scope in scopes:
    with tf.variable_scope(scope, reuse=True):
        next_state_h.append(tf.get_variable('train_h'))
        next_valid_state_h.append(tf.get_variable('valid_h'))
        next_test_state_h.append(tf.get_variable('test_h'))
print('Initial update to all the states')

# Maintains the last state output for all the layers
last_state_unrolled = [] 

# Calculating the output of the RNN for num_unroll steps
# (as required by the truncated BPTT)
for ui in range(num_unroll):
    x = train_dataset[ui]
    for lyr_i, scope in enumerate(scopes):
        # Recursively compute the RNN output
        next_state_h[lyr_i] = rnn_cell(scope, x, next_state_h[lyr_i])
        # Set the previous layer's output as the next layers input
        x = next_state_h[lyr_i]
            
    last_state_unrolled.append(x)
print('\n Defined training stage RNN computations')

# Updating the state variables with the latest state output at Training phase
tf_train_state_update_ops = []
for lyr_i, scope in enumerate(scopes):
    with tf.variable_scope(scope, reuse=True):
        tf_train_state_update_ops.append(
            tf.assign(tf.get_variable('train_h'),next_state_h[lyr_i])
        )
print('\n Defined training state update ops')

with tf.variable_scope('out',reuse=True):
    W_hy = tf.get_variable('W_hy')
    # Get the scores and predictions for all the RNN outputs we produced for num_unroll steps
    y_scores = [tf.matmul(last_state_unrolled[ui],W_hy) for ui in range(num_unroll)]
    y_predictions = [tf.nn.softmax(y_scores[ui]) for ui in range(num_unroll)]
    
    # We calculate train perplexity with the predictions made by the RNN
    train_perplexity_without_exp = tf.reduce_sum(
        tf.concat(train_labels,0)*-tf.log(tf.concat(y_predictions,0)+1e-10))/(num_unroll*batch_size)
print('\n Definined training predictions')

# ===============================================================================
# Validation data related inference logic 
# (very similar to the training inference logic)

# Compute the next valid state (only for 1 step)
x = valid_dataset
last_valid_state = None
for lyr_i, scope in enumerate(scopes):
    # Recursively compute the RNN output (validation)
    next_valid_state_h[lyr_i] = rnn_cell(scope, x, next_valid_state_h[lyr_i])
    # Set the previous layer's output as the next layers input (validation)
    x = next_valid_state_h[lyr_i]
    
last_valid_state = next_valid_state_h[-1]
print('\n Defined validation stage RNN computations')

# Updating the state variables with the latest state output at validation phase
tf_valid_state_update_ops = []
for lyr_i, scope in enumerate(scopes):
    with tf.variable_scope(scope, reuse=True):
        tf_valid_state_update_ops.append(
            tf.assign(tf.get_variable('valid_h'),next_valid_state_h[lyr_i])
        )
print('\n Defined validation state update ops')

with tf.control_dependencies(tf_valid_state_update_ops):
    with tf.variable_scope('out',reuse=True):
        W_hy = tf.get_variable('W_hy')
        valid_scores = tf.matmul(last_valid_state,W_hy) 
        valid_predictions = tf.nn.softmax(valid_scores)

valid_perplexity_without_exp = tf.reduce_sum(tf.concat(valid_labels,0)*-tf.log(tf.concat(valid_predictions,0)+1e-10))
print('\n Definined validation predictions')
# ===============================================================================
# Test data realted inference logic

# Calculating hidden output for test data
x = test_dataset
last_test_state = None
for lyr_i, scope in enumerate(scopes):
    # Recursively compute the RNN output (test)
    next_test_state_h[lyr_i] = rnn_cell(scope, x, next_test_state_h[lyr_i])
    # Set the previous layer's output as the next layers input (test)
    x = next_test_state_h[lyr_i]
    
last_test_state = next_test_state_h[-1]
print('\n Defined testing stage RNN computations')

# Updating the state variables with the latest state output at test phase
tf_test_state_update_ops = []
for lyr_i, scope in enumerate(scopes):
    with tf.variable_scope(scope, reuse=True):
        tf_test_state_update_ops.append(
            tf.assign(tf.get_variable('test_h'),next_test_state_h[lyr_i])
        )
print('\n Defined testing state update ops')

with tf.control_dependencies(tf_test_state_update_ops):
    with tf.variable_scope('out',reuse=True):
        W_hy = tf.get_variable('W_hy')
        test_scores = tf.matmul(last_test_state,W_hy) 
        test_predictions = tf.nn.softmax(test_scores)
        
print('\n Definined testing predictions')


# ## Calculating RNN Loss
# We calculate the training and validation loss of RNN here. It's a typical cross entropy loss calculated over all the scores we obtained for training data (`rnn_loss`) and validation data (`rnn_valid_loss`).

# In[10]:


with tf.control_dependencies(tf_train_state_update_ops):
    rnn_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        logits=tf.concat(y_scores,0), labels=tf.concat(train_labels,0)
    ))

rnn_valid_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
       logits=valid_scores, labels=valid_labels))


# ## Defining Learning Rate and the Optimizer with Gradient Clipping
# Here we define the learning rate and the optimizer we're going to use. We will be using the Adam optimizer as it is one of the best optimizers out there. Furthermore we use gradient clipping to prevent any gradient explosions.

# In[11]:


rnn_optimizer = tf.train.AdamOptimizer(learning_rate=0.0005)

gradients, v = zip(*rnn_optimizer.compute_gradients(rnn_loss))
gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
rnn_optimizer = rnn_optimizer.apply_gradients(zip(gradients, v))


# ## Resetting Operations for Resetting Hidden States
# Sometimes the state variable needs to be reset (e.g. when starting predictions at a beginning of a new epoch)

# In[12]:


training_reset_ops, valid_reset_ops, test_reset_ops = [],[],[]
for lyr_i, (scope,h) in enumerate(zip(scopes,hidden_sizes)):
    with tf.variable_scope(scope, reuse=True):
        training_reset_ops.append(tf.assign(tf.get_variable('train_h'),tf.zeros([batch_size,h],dtype=tf.float32)))
        valid_reset_ops.append(tf.assign(tf.get_variable('valid_h'),tf.zeros([1,h],dtype=tf.float32)))
        test_reset_ops.append(tf.assign(tf.get_variable('test_h'),tf.zeros([1,h],dtype=tf.float32)))


# ## Prediction Sampling
# We select the word corresponding to the highest index of the prediction vector. We will later see different sampling strategies.

# In[13]:


def sample(distribution):
  '''
  Sample a word from the prediction distribution
  '''  
  best_idx = np.argmax(distribution)
  return best_idx


# ## Running the RNN to Generate Text
# 
# Here we train the RNN on the available data and generate text using the trained RNN for several steps. First we create a validation set by extracting text snippets (that are not present in training data) from longer documents. Then at each training step, we train the RNN on several randomly picked documents. From each document we extract text for `steps_per_document` steps. We also report the train and validation perplexities at the end of each step. Finally we test the RNN by asking it to generate some new text starting from a randomly picked bigram.

# In[14]:


num_steps = 50 # Number of steps we run the algorithm for
# How many training steps are performed for each document in a single step
steps_per_document = 100 

# How often we run validation
valid_summary = 5

# In the book we run tests with this set to both 20 and 100
train_doc_count = 100
train_docs_to_use =10 # Number of docs we use in a single step

# Store the training and validation perplexity at each step
valid_perpelxity_ot = []
train_perplexity_ot = []

session = tf.InteractiveSession()
# Initializing variables
tf.global_variables_initializer().run()

print('Initialized')
average_loss = 0

# We use the first 10 documents that has 
# more than (num_steps+1)*steps_per_document bigrams for creating the validation dataset

# Identify the first 10 documents following the above condition
long_doc_ids = []
for di in range(num_files):
  if len(data_list[di])>(num_steps+1)*steps_per_document:
    long_doc_ids.append(di)
  if len(long_doc_ids)==10:
    break

# Generating validation data
data_gens = []
valid_data = []
for fi in range(num_files):
  # Get all the bigrams if the document id is not in the validation document ids
  if fi not in long_doc_ids:
    data_gens.append(DataGeneratorOHE(data_list[fi],batch_size,num_unroll))
  # if the document is in the validation doc ids, only get up to the 
  # last steps_per_document bigrams and use the last steps_per_document bigrams as validation data
  else:
    data_gens.append(DataGeneratorOHE(data_list[fi][:len(data_list[fi])-steps_per_document],batch_size,num_unroll))
    valid_data.extend(data_list[fi][-steps_per_document:])

# Defining the validation data generator
valid_gen = DataGeneratorOHE(valid_data,1,1)

feed_dict = {}
for step in range(num_steps):
    print('\n')
    for di in np.random.permutation(train_doc_count)[:train_docs_to_use]:                    
        doc_perplexity = 0
        for doc_step_id in range(steps_per_document):
            
            # Get a set of unrolled batches
            u_data, u_labels = data_gens[di].unroll_batches()
            
            # Populate the feed dict by using each of the data batches
            # present in the unrolled data
            for ui,(dat,lbl) in enumerate(zip(u_data,u_labels)):            
                feed_dict[train_dataset[ui]]=dat
                feed_dict[train_labels[ui]] = lbl            
            
            # Running the TensorFlow operation
            _, l, step_predictions, _, step_labels, step_perplexity =             session.run([rnn_optimizer, rnn_loss, y_predictions,
                         train_dataset,train_labels,train_perplexity_without_exp], 
                        feed_dict=feed_dict)
            
            # Update doc perplexity variable
            doc_perplexity += step_perplexity
            # Update average step perplexity 
            average_loss += step_perplexity
                
        print('Document %d Step %d processed (Perplexity: %.2f).'
              %(di,step+1,np.exp(doc_perplexity/steps_per_document))
             )
        
    # resetting hidden state after processing a single document
    session.run(training_reset_ops) 
    
    # Validation phase
    if step % valid_summary == 0:
      
      # Compute average loss
      average_loss = average_loss / (train_docs_to_use*steps_per_document*valid_summary)
      
      print('Average loss at step %d: %f' % (step+1, average_loss))
      print('\tPerplexity at step %d: %f' %(step+1, np.exp(average_loss)))
      train_perplexity_ot.append(np.exp(average_loss))
      
      average_loss = 0 # reset loss
      
      valid_loss = 0 # reset loss
      
      # calculate valid perplexity
      for v_step in range(steps_per_document*10):
        uvalid_data,uvalid_labels = valid_gen.unroll_batches()        
        
        # Run validation phase related TensorFlow operations
        v_loss,v_preds,v_labels,v_preplexity = session.run(
            [rnn_valid_loss,valid_predictions,valid_labels, valid_perplexity_without_exp],
            feed_dict = {valid_dataset:uvalid_data[0],valid_labels: uvalid_labels[0]}
        )
        
        # Update validation perplexity
        valid_loss += v_preplexity        
      
      # Reset validation data generator cursor
      valid_gen.reset_indices()  
    
      print()
      print("Valid Perplexity: %.2f\n"%np.exp(valid_loss/(steps_per_document*10)))
      valid_perpelxity_ot.append(np.exp(valid_loss/(steps_per_document*10)))
      session.run(valid_reset_ops)
        
      # Generating new text ...
      # We will be generating one segment having 1000 bigrams
      # Feel free to generate several segments by changing
      # the value of segments_to_generate
      print('Generated Text after epoch %d ... '%step)  
      segments_to_generate = 1
      chars_in_segment = 1000
    
      for _ in range(segments_to_generate):
        print('======================== New text Segment ==========================')
        # Start with a random word
        test_word = np.zeros((1,input_sizes[0]),dtype=np.float32)
        test_word[0,data_list[np.random.randint(0,num_files)][np.random.randint(0,500)]] = 1.0
        print("\t",reverse_dictionary[np.argmax(test_word[0])],end='')
        
        # Generating words within a segment by feeding in the previous prediction
        # as the current input in a recursive manner
        for _ in range(chars_in_segment):    
          test_pred = session.run(test_predictions, feed_dict = {test_dataset:test_word})  
          next_ind = sample(test_pred.ravel())
          test_word = np.zeros((1,input_sizes[0]),dtype=np.float32)
          test_word[0,next_ind] = 1.0
          print(reverse_dictionary[next_ind],end='')
        
        print("")
        # Reset test state
        session.run(test_reset_ops)
        print('====================================================================')
      print("")

