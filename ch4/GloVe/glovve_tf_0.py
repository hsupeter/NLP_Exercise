# -*- coding: utf-8 -*-
# # GloVe: Global Vectors for Word2Vec

# In[1]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
#%matplotlib inline
from __future__ import division, print_function, unicode_literals
import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import bz2
import pandas as pd
from matplotlib import pylab
import matplotlib.pyplot as plt
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from scipy.sparse import lil_matrix
import nltk # standard preprocessing
import operator # sorting items in dictionary by value
#nltk.download() #tokenizers/punkt/PY3/english.pickle
from math import ceil
import csv
from tensorflow_graph_in_jupyter import show_graph

# In[2]:

# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
CHAPTER_ID = "nlp_tens"

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

# In[3]:
# # Dataset
# This code downloads a dataset consisting of several Wikipedia articles 
# totaling up to roughly 61 megabytes. Additionally the code 
# makes sure the file has the correct size after downloading it.

url = 'http://www.evanjones.ca/software/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    print('Downloading file...')
    filename, _ = urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified %s' % filename)
  else:
    print(statinfo.st_size)
    raise Exception(
      'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

filename = maybe_download('../wikipedia2text-extracted.txt.bz2', 18377035)

# In[4]:
''' Read Data with Preprocessing with NLTK'''
''' Reads data as it is to a string, convert to lower-case and  
 tokenize it using the nltk library.'''
def read_data(filename):
  """
  Extract the first file enclosed in a zip file as a list of 
  words and pre-processes it using the nltk python library.
  """
  with bz2.BZ2File(filename) as f:
      data = []
      file_size = os.stat(filename).st_size
      print('file_size',file_size)
      chunk_size = 1024 * 1024 # reading 1 MB at a time as the
                               # dataset is moderately large
      print('Reading data...')
      for i in range(ceil(file_size//chunk_size)+1):
          bytes_to_read = min(chunk_size, file_size-(i*chunk_size))
          file_string = f.read(bytes_to_read).decode('utf-8')
          #print('string_token:', file_string)
          file_string = file_string.lower()
          # tokenizes a string to word residing in a list
          file_string = nltk.word_tokenize(file_string)
          #print('\nstring_token:', file_string)
          data.extend(file_string)
  return data

words = read_data(filename)
print('Data size %d' % len(words))
print('Example words (start):', words[:10])
print('Example words (end):', words[-10:])
# In[5]:
'''
## Building the Dictionaries

由要處理的文章來建立Dictionary (不是使用已有的資料庫當字典)
Builds the following. To understand each of these elements,
let us also assume the text "I like to go to school"

. dictionary: maps a string word to an ID 
             (e.g. {I:0, like:1, to:2, go:3, school:4})
. reverse_dictionary: maps an ID to a string word 
             (e.g. {0:I, 1:like, 2:to, 3:go, 4:school}
. count: List of list of (word, frequency) elements 
             (e.g. [(I,1),(like,1),(to,2),(go,1),(school,1)]
. text_digt : Contain the string of text we read, where string 
      words are replaced with word IDs (e.g. [0, 1, 2, 3, 2, 4])

It also introduces an additional special token UNK to denote
rare words to are too rare to make use of.'''
# In[6]:
# we restrict our vocabulary size to 50000
vocabulary_size = 50000

def build_dataset(words):
    count = [['UNK', -1]]
    # Gets only the vocabulary_size most common words as the vocabulary
    # All the other words will be replaced with UNK token
    count.extend(collections.Counter(words).most_common(vocabulary_size-1))
    dictionary = dict()
    
    # Create an ID for each word by giving the current length of the dictionary
    # And adding that item to the dictionary
    '''以count出現的次序處理，即出現次數高低次序，例，最高次數'UNK'，此時
       len(dictionary)=0，dictionary['UNK']=0 => 
       len(dictionary)=1，dictionary['the']=1 => 循環... =>   
       dictionary：{UNK:0, the:1, ,:2, .:3, of:4, ...}
       此dictionary value就是index，再經text_digt處理成原文字的index'''
       
    for word, _ in count:
        dictionary[word] = len(dictionary)  
        
    text_digt = list()
    unk_count = 0
    # Traverse through all the text we have and produce a list
    # where each element corresponds to the ID of the word found at that index
    for word in words:
        # If word of the dictionary uses the word ID,
        # else use the ID of the special token "UNK"
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0   # dictionary['UNK']
            unk_count += 1
        text_digt.append(index)
            
    # updata the count variable with the number of UNK occurences
    count[0][1] = unk_count
    
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    # Make sure the dictionary is of size of the vocabulary
    assert len(dictionary) == vocabulary_size
    
    return text_digt, count, dictionary, reverse_dictionary

text_digt, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK): ', count[:10])
print('words:', words[:10])
print('Sample text_digitize:', text_digt[:10])
del words
# In[7]:
'''
## Generating Batches of Data for GloVe

 Generates a batch or target words (batch) and a batch of 
 corresponding context words (labels). It reads 2*window_size+1
 words at a time (called a span) and create 2*window_size 
 datapoints in a single span. The function continue in this 
 manner until batch_size datapoints are created. Everytime we 
 reach the end of the word sequence, we start from beginning.'''
# In[8]:
data_index = 0

def generate_batch_GloVe(text_digt, batch_size, window_size):
  # data_index is updated by 1 everytime we read a data point
  global data_index 
    
  # two numpy arras to hold target words (batch)
  # and context words (labels)
  batch = np.ndarray(shape=(batch_size), dtype=np.int32)
  labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
  weights = np.ndarray(shape=(batch_size), dtype=np.float32)

  # span defines the total window size, where
  # data we consider at an instance looks as follows. 
  # [ skip_window target skip_window ]
  span = 2 * window_size + 1 
    
  # The buffer holds the data contained within the span
  buffer = collections.deque(maxlen=span)
  
  # Fill the buffer and update the data_index
  for _ in range(span):
    buffer.append(text_digt[data_index])
    data_index = (data_index + 1) % len(text_digt)
  
  # This is the number of context words we sample for 
  # a single target word
  num_samples = 2*window_size 

  # We break the batch reading into two for loops
  # The inner for loop fills in the batch and labels with 
  # num_samples data points using data contained withing the span
  # The outper for loop repeat this for 
  # batch_size//num_samples times to produce a full batch
  for i in range(batch_size // num_samples):
    k=0
    # avoid the target word itself as a prediction
    # fill in batch and label numpy arrays
    for j in list(range(window_size))+list(range(window_size+1,\
                                            2*window_size+1)):
      batch[i * num_samples + k] = buffer[window_size]
      labels[i * num_samples + k, 0] = buffer[j]
      '''近目標字其weight值高，最大的隔壁字的weight值=1，依序比例降低'''
      weights[i * num_samples + k] = abs(1.0/(j - window_size))
      k += 1 
    
    # Everytime we read num_samples data points,
    # we have created the maximum number of datapoints possible
    # withing a single span, so we need to move the span by 1
    # to create a fresh new span
    buffer.append(text_digt[data_index])
    data_index = (data_index + 1) % len(text_digt)
  return batch, labels, weights

print('text_digt:', [reverse_dictionary[di] for di in text_digt[:8]])

for window_size in [2, 4]:
    data_index = 0
    batch, labels, weights = generate_batch_GloVe(text_digt,\
                        batch_size=8, window_size=window_size)
    print('\nwith window_size = %d:' %window_size)
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li\
                          in labels.reshape(8)])
    print('    weights:', [w for w in weights])
# In[9]:
'''    
## Creating the Word Co-Occurance Matrix
 Why GloVe shine above context window based method is that
 it employs global statistics of the corpus in to the model
 (according to authors). This is done by using information from
 the word co-occurance matrix to optimize the word vectors.
 Basically, the X(i,j) entry of the co-occurance matrix says
 how frequent word i to appear near j. We also use a weighting
 mechanishm to give more weight to words close together than 
 to ones further-apart (from experiments section of the paper).
'''
# In[10]:
# We are creating the co-occurance matrix as a compressed sparse colum matrix from scipy. 
cooc_data_index = 0
dataset_size = len(text_digt) # We iterate through the full text
cooc_batch_size=8 # Each tackling coocurences batch size
skip_window = 4 # How many words to consider left and right.

# The sparse matrix that stores the word co-occurences
''' lil_matrix用來處理sparse matrix可節省記憶體
    lil_matrix矩陣存放格式每位置：指標(inp, lbl)及值(w)，
                                如  (0, 1)     260.41 '''
cooc_mat = lil_matrix((vocabulary_size, vocabulary_size), dtype=np.float32)

print(cooc_mat.shape)
'''建立co-occurrence的weight矩陣
   返回cooc_mat，為(inp, lbl)及值(w)的組合，意味著
   input(文本中心字)與lbl(文本背景字)同存的累加weight值
   如(0, 1) 260.41表示"UNK(0)和"the(1)的weight為260.41"'''
def generate_cooc(cooc_batch_size, skip_window):
    '''
    Generate co-occurence matrix by processing batches of data
    '''
    data_index = 0
    print('Running %d iterations to compute the co-occurance matrix'\
          %(dataset_size//cooc_batch_size))
    for i in range(dataset_size//cooc_batch_size):
        # Printing progress
        if i>0 and i%100000==0:
            print('\tFinished %d iterations'%i)
            #print('batch_co:\n',batch_co)
            #print('labels_co:\n',labels_co)
            #print('weights_co:\n',weights_co)            
        # Generating a single batch of data
        batch_co, labels_co, weights_co = generate_batch_GloVe(\
                         text_digt, cooc_batch_size, skip_window)
        labels_co = labels_co.reshape(-1)
        
        # Incrementing the sparse matrix entries accordingly
        for inp,lbl,w in zip(batch_co,labels_co,weights_co):            
            cooc_mat[inp,lbl] += (1.0*w) # 累加 w
    # cooc_mat矩陣存放格式：指標(inp, lbl)及值(w)，如(0, 1)  260.41638         
    #print('cooc_mat[inp,lbl]:',cooc_mat[:20, :10])

# Generate the matrix
generate_cooc(cooc_batch_size, skip_window)    

# Just printing some parts of co-occurance matrix
print('Sample chunks of co-occurance matrix')


# Basically calculates the highest cooccurance of several chosen word
for i in range(10):
    idx_target = i
    
    # get the ith row of the sparse matrix and make it dense
    '''ith_row取cooc_mat的row；ith_row_dense再去指標而成矩陣
       ith_row:(0, 0)  144.50
               (0, 1)  257.41 ...
     =>ith_row_dense: [144.50 257.41 ...]          
    '''
    ith_row = cooc_mat.getrow(idx_target) # shape(1,50000)    
    #print('ith_row[,:10]:\n', ith_row[:,:10])     
    ith_row_dense = ith_row.toarray('C').reshape(-1) # shape(50000, )       
    #print('\nith_row_dense[:10]:\n', ith_row_dense[:10])    
    
    # select target words only with a reasonable words around it.
    while np.sum(ith_row_dense)<10 or np.sum(ith_row_dense)>50000:
        # Choose a random word
        idx_target = np.random.randint(0,vocabulary_size)
        
        # get the ith row of the sparse matrix and make it dense
        ith_row = cooc_mat.getrow(idx_target) 
        ith_row_dense = ith_row.toarray('C').reshape(-1)    
        
    '''ith_row_dense存放累加的weight為float (未排序)
       np.argsort(ith_row_dense)由小到大排序ith_row_dense的index為int
       np.flip(sort_indices,axis=0)再翻轉排序成由大到小的index
       而此index正好是dictionary（依出現次數高低順序形成value，
       eg.{'UNK':0, 'the':1, ',':2, '.':3, 'of':4, ...}）的排序'''       
    print('\nTarget Word: "%s"'%reverse_dictionary[idx_target])
    print('ith_row_dense[:10]:\n', ith_row_dense[:10])  #shape (50000,)    
    sort_indices = np.argsort(ith_row_dense).reshape(-1) #shape (50000,)
    print('sort_indices [-10:]:\n', sort_indices[-10:])
    # reverse the array (to get max values to the start)
    sort_indices = np.flip(sort_indices,axis=0) #沿軸 0翻轉 shape (50000,)
    print('sort_indices_flip [:10]:\n', sort_indices[:10])
    
    # printing several context words to make sure cooc_mat is correct
    print('Context word:',end='')
    for j in range(10):        
        idx_context = sort_indices[j]       
        print('"%s"(id:%d,count:%.2f), '%(reverse_dictionary[idx_context],idx_context,ith_row_dense[idx_context]),end='')
    print()
# In[11]:
'''
## GloVe Algorithm
### Defining Hyperparameters
 Here we define several hyperparameters including batch_size 
 (amount of samples in a single batch) embedding_size (size of 
 embedding vectors) window_size (context window size).'''

# In[12]:
import random
batch_size = 128 # Data Points in a single batch.
embedding_size = 128 # Dimension of the embedding vector.
window_size = 4  # How many words to consider left and right.

# We pick a random validation set to sample nearest neighbors.
valid_size = 16 # Random set of words to evaluate similarity.
# We sample valid datapoints randomly from a large  
# window without always being deterministic.
valid_window = 50

# 第一行 valid example, 選擇某些最常出現的字 
# 第二行 valid exampleas 再加入同量較少出現的字
# random.sample(population樣本, k取樣數) 
# 回傳長度為 k 且元素唯一的 list。此為非重置抽樣 (sampling without replacement)
valid_examples = np.array(random.sample(range(valid_window), valid_size))
valid_examples = np.append(valid_examples,random.sample(\
            range(1000, 1000+valid_window), valid_size),axis=0)

num_sampled = 32 # Number of negative examples to sample.

epsilon = 1 # used for the stability of log in the loss function
# In[13]:
'''
### Defining Inputs and Outputs
 
 Here we define placeholders for feeding in training 
 inputs and outputs (each of size batch_size) and a 
 constant tensor to contain validation examples.'''
# In[14]:
reset_graph()
''''此處train_dataset和train_labels是文本數字化的ID，
也就是上面generate_batch_skip_gram()傳回的batch, labels'''
# Training input data (target word IDs).
train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
# Training input label dasta (context word IDs)
train_labels = tf.placeholder(tf.int32, shape=[batch_size])
# Validation input data, we don't need a placeholder as we have
# already defined the IDs of the words selected as validation data.
'''valid_dataset 隨機選擇用來驗證的 ID dataset被tf.constant固定
   即，每次驗證的valid_examples是同一組，除非重執行valid_examples'''
valid_dataset = tf.constant(valid_examples, dtype = tf.int32)
# In[15]:
'''
### Defining Model Parameters and Other Variables
 
 We now define four TensorFlow variables which is 
 composed of an embedding layer, a bias for each
 input and output words.
'''
# In[16]:
# Variables.
in_embeddings = tf.Variable(tf.random_uniform(\
   [vocabulary_size, embedding_size],-1.0,1.0),name='in_embeddings')
                      
in_bias_embeddings = tf.Variable(tf.random_uniform(\
   [vocabulary_size],0.0,0.01,dtype=tf.float32),name='in_bias_embeddings')

out_embeddings = tf.Variable(tf.random_uniform(\
   [vocabulary_size, embedding_size],-1.0,1.0),name='out_embeddings')
out_bias_embeddings = tf.Variable(tf.random_uniform(\
   [vocabulary_size],0.0,0.01,dtype=tf.float32),name='out_bias_embeddings')
# In[17]:
'''
### Defining the Model Computations

 We first defing a lookup function to fetch the corresponding
 embedding vectors for a set of given inputs. Then we define
 a placeholder that takes in the weights for a given batch of
 data points (weights_x) and co-occurence matrix weights (x_ij).
 Weights_x measures the importance of a data point with 
 respect to how much those two words co-occur and x_ij denotes
 the co-occurence matrix value for the row and column denoted
 by the words in a datapoint. With these defined, we can define
 the loss as shown below. For exact details referChapter 4 text. 
'''
# In[18]:
os.system('glove_loss.png') # show loss function
# Look up embeddings for inputs and outputs
# Have two seperate embedding vector spaces for inputs and outputs
embed_in = tf.nn.embedding_lookup(in_embeddings, train_dataset)
embed_out = tf.nn.embedding_lookup(out_embeddings, train_labels)
embed_bias_in = tf.nn.embedding_lookup(in_bias_embeddings,\
                                       train_dataset)
embed_bias_out=tf.nn.embedding_lookup(out_bias_embeddings,\
                                        train_labels)

# weights_x measures the importance of a data point with 
# respect to how much those two words co-occur
weights_x = tf.placeholder(tf.float32,shape=[batch_size],\
                           name='weights_x') 
# Cooccurence value for that position
# co-occurence matrix weights
x_ij = tf.placeholder(tf.float32,shape=[batch_size],\
                      name='x_ij')

# Compute the loss defined in the paper. Note that 
# I'm not following the exact equation given (which is computing a pair of words at a time)
# I'm calculating the loss for a batch at one time, but the calculations are identical.
# I also made an assumption about the bias, that it is a smaller type of embedding
loss = tf.reduce_mean(\
  weights_x * (tf.reduce_sum(embed_in*embed_out,axis=1)\
  +embed_bias_in +embed_bias_out -tf.log(epsilon+x_ij))**2)

# In[19]:
'''
### Calculating Word Similarities
 
 We calculate the similarity between two given words
 in terms of the cosine distance. To do this efficiently 
 we use matrix operations to do so, as shown below.
'''
# In[20]:

#Compute the similaruty between minibatch examples and all embeddings.
#We use the cosine distance:
# Embedding Vector都是學習過的值，所以相似性都是 valid_dataset
# 對所有字學習完的 Embedding Vector的cosine distance比較
# Embedding Vector 正規化（歸一化，normalization）
#reduce_sum(tensors, axis, keepdims)的reduce有降維作用
#因此keepdims=True則保持維度不降維
'''此處embeddings取in_embeddings 和 out_embeddings平均'''
embeddings = (in_embeddings + out_embeddings)/2.0
norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
normalized_embeddings = embeddings/norm
#vembedding_lookup(tensor, id)在tensor中查找id對應的列或行元素
#v其中valid_dataset 隨機選擇用來驗證的 ID dataset
# 找到的正規化的embedding vector(已學習改變的vector)用來計算相似性similarity
# similarity = cosine distance= A‧B/|A||B| 因經正規化,所以|A|=|B|=1，
# => similarity = A‧B (矩陣內積)（0度最相似cos(0)=1 => A=B）
#  此處A為vaild example的embedding vector，B為所有embedding vector
# 其shape [valid_examples size, vocabulary_size] (32, 50000)
#v故，similarity意味valid_examples的字對vocabulary所有字計算cosine distance    

valid_embeddings = tf.nn.embedding_lookup(\
                   normalized_embeddings, valid_dataset)                   
similarity = tf.matmul(valid_embeddings,\
                    tf.transpose(normalized_embeddings))
# A類:去除單位用於差異單位及誇度的資料比較分析，及利於梯度下降收斂
#   Normalization(正規化): 不改變原有資料分布
#      1. Xnom = (X-Xmin)/(Xmax-Xmin) ∊ N(0,1)
#      2. Xnom = (X-μ)/(Xmax-Xmin)  ∊ N(-1,1)
#   Standardization(標準化)：變成常態分布 
#         Z=(X-μ)/σ  ∊ N(0,1)
# B類：用於降低overfitting    
#   Regulization(正則化): l1 norm(Lasso), l2 norm(Ridge)
#      1. l1 = λ*∑|weights|    
#      2. l2 = λ*∑(weights)**2
 #     註 cost = predict err + regulation

# In[21]:
'''
### Model Parameter Optimizer

 We then define a constant learning rate and an optimizer
 which uses the Adagrad method. 
'''
# In[22]:
# Optimizer.
optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
# In[23]:
'''
### Running the GloVe Algorithm

 Here we run the GloVe algorithm we defined above. 
 Specifically, we first initialize variables, and then train
 the algorithm for many steps (num_steps). And every few
 steps we evaluate the algorithm on a fixed validation
 set and print out the words that appear to be closest for
 a given set of words.
'''
# In[24]:
num_steps = 100001
glove_loss = []

average_loss = 0
with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as session:
    
    tf.global_variables_initializer().run()
    print('Initialized')
    
    for step in range(num_steps):
        
        # generate a single batch (data,labels,co-occurance weights)
        batch_data, batch_labels, batch_weights =\
            generate_batch_GloVe(text_digt,\
            batch_size, skip_window) 
        
        # Computing the weights required by the loss function
        batch_weights = [] # weighting used in the loss function
        batch_xij = [] # weighted frequency of finding i near j
        
        # Compute the weights for each datapoint in the batch
        ''' 統計同時存在weight的cooc_mat已在 generate_cooc建立
            此處batch_data及batch_labels只是用來在此batch段text中
            每個inp(中心詞), lbl(背景詞)對裡查找其對應的weights_co'''
        for inp,lbl in zip(batch_data,batch_labels.reshape(-1)):     
            point_weight = (cooc_mat[inp,lbl]/100.0)**0.75\
                if cooc_mat[inp,lbl]<100.0 else 1.0 
            batch_weights.append(point_weight) # 調整後的batch weight
            batch_xij.append(cooc_mat[inp,lbl]) # 未調整的batch weight
        batch_weights = np.clip(batch_weights,-100,1) #每值取最小-100,最大1
        batch_xij = np.asarray(batch_xij)
        
        # Populate the feed_dict and run the optimizer (minimize loss)
        # and compute the loss. Specifically we provide
        # train_dataset/train_labels: training inputs and training labels
        # weights_x: measures the importance of a data point with respect to how much those two words co-occur
        # x_ij: co-occurence matrix value for the row and column denoted by the words in a datapoint
        feed_dict = {train_dataset : batch_data.reshape(-1),\
                train_labels : batch_labels.reshape(-1),\
                weights_x:batch_weights, x_ij:batch_xij}
        _, loss_val = session.run([optimizer, loss],\
                                  feed_dict=feed_dict)
        
        # Update the average loss variable
        average_loss += loss_val
        if step % 2000 == 0:
          if step > 0:
            average_loss = average_loss / 2000
          # The average loss is an estimate of the loss over the last 2000 batches.
          print('Average loss at step %d: %f' % (step, average_loss))
          glove_loss.append(average_loss)
          average_loss = 0
        
        # Here we compute the top_k closest words for a given validation word
        # in terms of the cosine distance
        # We do this for all the words in the validation set
        # Note: This is an expensive step
        if step % 10000 == 0:
          sim = similarity.eval()
          for i in range(valid_size):
            valid_word = reverse_dictionary[valid_examples[i]]
            top_k = 8 # number of nearest neighbors
            nearest = (-sim[i, :]).argsort()[1:top_k+1]
            log = 'Nearest to "%s":' % valid_word
            for k in range(top_k):
              close_word = reverse_dictionary[nearest[k]]
              log = '%s %s,' % (log, close_word)
            print(log)
            
    glove_final_embeddings = normalized_embeddings.eval()
    
# We will save the word vectors learned and the loss over time
# when this information is required later for comparisons.
np.save('glove_embeddings', glove_final_embeddings)

with open('glove_loss.csv', 'wt') as f:
  writer = csv.writer(f, delimiter=',')
  writer.writerow(glove_loss)
# In[25]:
glove_embed_load=np.load('glove_embeddings.npy')
# In[26]:
def find_clustered_embeddings(embeddings,distance_threshold,sample_threshold):
    ''' 
    返回embeddings_distance>distance_threshold的索引
    Find only the closely clustered embeddings. 
    This gets rid of more sparsly distributed word embeddings
    and make the visualization clearer.
    This is useful for t-SNE visualization
    
    distance_threshold: maximum distance between two points 
                        to qualify as neighbors
    sample_threshold: number of neighbors required to be 
                      considered a cluster.
             參看圖skip_gram_samp0及skip_gram_samp10差異
    '''
    
    ''' calculate cosine similarity '''
    '''計算所有embeddings vector 的相似性，及正規化
       cosine_sim 及 norm是shape(1000,1000)的對稱矩陣'''
    cosine_sim = np.dot(embeddings, np.transpose(embeddings))
    norm = np.dot(np.sum(embeddings**2,axis=1).reshape(-1,1),\
         np.sum(np.transpose(embeddings)**2,axis=0).reshape(1,-1))
    # assert <test>, <message> 其中test是狀態測試，
    # message是斷言失敗時呈現的訊息。若無message則返回AssertionError
    assert cosine_sim.shape == norm.shape # 僅 assert <test>
    cosine_sim /= norm  # cosine_sim = cosine_sim/norm
     # make all the diagonal entries -1.0 otherwise this will be picked as highest
    np.fill_diagonal(cosine_sim, -1.0) # 矩陣對角線填-1.0    
    argmax_cos_sim = np.argmax(cosine_sim, axis=1) # 返回axis軸最大值的索引
    mod_cos_sim = cosine_sim 
       
    ''' foor loop 用來把每一row最大值index設為-1，意味排除相似性  
        此用意應該是排除中心詞附近的相似性，盡量找text內的其他字的相似性
        
        註：雖然指令是設定mod_cos_sim(row, argmax index)=-1，
           但因為mod_cos_sim和cosine_sim是相同指摽，
           所以mod_cos_sim和cosine_sim矩陣值同時改'''
    for _ in range(sample_threshold-1):
        # cosine_sim把每一row最大值的index        
        np.set_printoptions(precision=2)        
        argmax_cos_sim = np.argmax(cosine_sim, axis=1)
        # np.arange(mod_cos_sim.shape[0]=[0, 1,...,999]
        # mod_cos_sim每一row的最大值index(argmax_cos_sim指定)處值為-1
        mod_cos_sim[np.arange(mod_cos_sim.shape[0]),argmax_cos_sim] = -1
    #找出每一列最大值（已排除 sample_threshold 的字）    
    max_cosine_sim = np.max(mod_cos_sim,axis=1) # shape(1000,)
    #print('max_cosine_sim', max_cosine_sim)
    '''np.where(condition) 返回符合條件(僅返回大於dist_threshold字)的索引
      np.where()返回tuple所以要用np.where()[0]取陣列'''
    return np.where(max_cosine_sim>distance_threshold)[0]

# In[27]:
num_points = 1000 # we will use a large sample space to build the T-SNE 
                  # manifold and then prune it using cosine similarity

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

print('Fitting embeddings to T-SNE. This can take some time ...')
# get the T-SNE manifold
selected_embeddings = glove_embed_load[:num_points, :]
two_d_embeddings = tsne.fit_transform(selected_embeddings)

print('Pruning the T-SNE embeddings')
# prune the embeddings by getting ones only more than 
# n-many sample above the similarity threshold
# this unclutters the visualization
selected_ids = find_clustered_embeddings(selected_embeddings,.25,10)
two_d_embeddings = two_d_embeddings[selected_ids,:]

print('Out of ',num_points,' samples, ', selected_ids.shape[0],' samples were selected by pruning')

# In[28]:
'''
### Plotting the t-SNE Results with Matplotlib
'''

# In[29]:
import matplotlib.cm as cm
def plot(two_d_embed, labels):
  
  n_clusters = 20 # number of clusters
  # automatically build a discrete set of colors, each for cluster
  cmap = cm.get_cmap("Spectral") 
  #label_colors = [pylab.cm.spectral(float(i) /n_clusters) for i in range(n_clusters)]
  label_colors = [cmap(float(i) /n_clusters) for i in range(n_clusters)]
  
  assert two_d_embed.shape[0] >= len(labels), 'More labels than embeddings'
  
  # Define K-Means
  kmeans = KMeans(n_clusters=n_clusters, init='k-means++', random_state=0).fit(embeddings)
  kmeans_labels = kmeans.labels_
  
  pylab.figure(figsize=(15,15))  # in inches
    
  # plot all the embeddings and their corresponding words
  for i, (label,klabel) in enumerate(zip(labels,kmeans_labels)):
    x, y = two_d_embed[i,:]
    pylab.scatter(x, y, c=label_colors[klabel])    
        
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom',fontsize=10)

  # use for saving the figure if needed
  #pylab.savefig('word_embeddings.png')
  pylab.show()

words = [reverse_dictionary[i] for i in selected_ids]
plot(two_d_embeddings, words)
# In[30]:
'''改find_clustered_embeddings的sample_threshold=1)'''
num_points = 1000 # we will use a large sample space to build the T-SNE 
                  # manifold and then prune it using cosine similarity

tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)

print('Fitting embeddings to T-SNE. This can take some time ...')
# get the T-SNE manifold
selected_embeddings = glove_embed_load[:num_points, :]
two_d_embeddings = tsne.fit_transform(selected_embeddings)

print('Pruning the T-SNE embeddings')
# prune the embeddings by getting ones only more than 
# n-many sample above the similarity threshold
# this unclutters the visualization
selected_ids = find_clustered_embeddings(selected_embeddings,.25,1)
two_d_embeddings = two_d_embeddings[selected_ids,:]

print('Out of ',num_points,' samples, ',selected_ids.shape[0],\
      ' samples were selected by pruning')

# In[31]:

os.system('glove_loss.png')