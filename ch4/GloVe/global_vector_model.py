# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 23:17:30 2020

@author: Peter Hsu
"""
# In[1]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import division, print_function, unicode_literals
import collections
import numpy as np
import random
import tensorflow as tf
from scipy.sparse import lil_matrix
import csv

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

class gloVeModel(object):
   
# # Dataset
# This code downloads a dataset consisting of several Wikipedia articles 
# totaling up to roughly 61 megabytes. Additionally the code 
# makes sure the file has the correct size after downloading it.
        
    def __init__(self,
                 batch_size=128,
                 window_size = 4,
                 num_steps = 100001,                 
                 embedding_size = 128,                 
                 valid_size = 16,
                 valid_window = 50,
                 num_sampled = 32,
                 vocab_size=50000,
                 data_index = 0
                 ):
        """Constructor.
        Args:
            vocab_size: maximum  vocabularysize 
        """
        
        cooc_mat = lil_matrix((vocab_size,
                    vocab_size), dtype=np.float32)
        self.batch_size = batch_size
        self.window_size = window_size
        self.num_steps = num_steps        
        self.vocab_size = vocab_size
        self.cooc_mat = cooc_mat
        self.embedding_size = embedding_size # Dimension of the embedding vector.
        self.valid_size = valid_size # Random set of words to evaluate similarity.
        self.valid_window = valid_window
        self.num_sampled = num_sampled # Number of negative examples to sample.        
        self.data_index = data_index
        self._session = None      
                 
        #self.dataset_size = len(text_digt) # We iterate through the full text
        
    ''' Generating Batches of Data for GloVe    
     Generates a batch or target words (batch) and a batch of 
     corresponding context words (labels). It reads 2*window_size+1
     words at a time (called a span) and create 2*window_size 
     datapoints in a single span. The function continue in this 
     manner until batch_size datapoints are created. Everytime we 
     reach the end of the word sequence, we start from beginning.'''

        
    def _generate_batch_GloVe(self, text_digt, batch_size,
                               window_size):        
        # data_index is updated by 1 everytime we read a data point
        #global data_index                        
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
            buffer.append(text_digt[self.data_index])
            self.data_index = (self.data_index + 1) % len(text_digt)
      
        # This is the number of context words we sample for 
        # a single target word
        num_samples = 2 * window_size 
    
        # We break the batch reading into two for loops
        # The inner for loop fills in the batch and labels with 
        # num_samples data points using data contained withing the span
        # The outper for loop repeat this for 
        # batch_size//num_samples times to produce a full batch
        for i in range(batch_size // num_samples):
            k=0
            # avoid the target word itself as a prediction
            # fill in batch and label numpy arrays
            for j in list(range(window_size))+list(\
                    range(window_size + 1, 2 * window_size + 1)):
                batch[i * num_samples + k] = buffer[window_size]
                labels[i * num_samples + k, 0] = buffer[j]
                '''近目標字其weight值高，最大的隔壁字的weight值=1，依序比例降低'''
                weights[i * num_samples + k] = abs(1.0/(j - window_size))
                k += 1        
            # Everytime we read num_samples data points,
            # we have created the maximum number of datapoints possible
            # withing a single span, so we need to move the span by 1
            # to create a fresh new span
            buffer.append(text_digt[self.data_index])
            self.data_index = (self.data_index + 1) % len(text_digt)
        return batch, labels, weights, self.data_index    
          
    '''## Creating the Word Co-Occurance Matrix
     Why GloVe shine above context window based method is that
     it employs global statistics of the corpus in to the model
     (according to authors). This is done by using information from
     the word co-occurance matrix to optimize the word vectors.
     Basically, the X(i,j) entry of the co-occurance matrix says
     how frequent word i to appear near j. We also use a weighting
     mechanishm to give more weight to words close together than 
     to ones further-apart (from experiments section of the paper).
    '''
   
     # We iterate through the full text
     # Each tackling coocurences batch size
     # The sparse matrix that stores the word co-occurences       
       
    def _generate_cooc(self, text_digt, cooc_batch_size=8,
                       skip_window=4):        
        '''
        Generate co-occurence matrix by processing batches of data
        '''
        
        ''' lil_matrix用來處理sparse matrix可節省記憶體
        lil_matrix矩陣存放格式每位置：指標(inp, lbl)及值(w)，
                                    如  (0, 1)     260.41 '''
        '''建立co-occurrence的weight矩陣
        返回cooc_mat，為(inp, lbl)及值(w)的組合，意味著
        input(文本中心字)與lbl(文本背景字)同存的累加weight值
        如(0, 1) 260.41表示"UNK(0)和"the(1)的weight為260.41"'''                                               
        print(self.cooc_mat.shape)         
        print('Running %d iterations to compute the co-occurance matrix'
              %(self.dataset_size//cooc_batch_size))
        for i in range(self.dataset_size//cooc_batch_size):
            # Printing progress
            if i>0 and i%100000==0:
                print('\tFinished %d iterations'%i)                           
            # Generating a single batch of data
            dt_id=0
            batch_co, labels_co, weights_co, dt_id = self._generate_batch_GloVe(\
                     text_digt, cooc_batch_size, skip_window)
            labels_co = labels_co.reshape(-1)            
            # Incrementing the sparse matrix entries accordingly
            for inp,lbl,w in zip(batch_co,labels_co,weights_co):            
                self.cooc_mat[inp,lbl] += (1.0*w) # 累加 w                
                
        # cooc_mat矩陣存放格式：指標(inp, lbl)及值(w)，如(0, 1)  260.41638         
        #print('cooc_mat[inp,lbl]:',cooc_mat[:20, :10])
        return self.cooc_mat, dt_id
    
    '''
    ## GloVe Algorithm
    ### Defining Hyperparameters
     Here we define several hyperparameters including batch_size 
     (amount of samples in a single batch) embedding_size (size of 
     embedding vectors) window_size (context window size).'''    
        
    
    def _build_graph(self, name='build_graph'):
        epsilon = 1        
        with tf.name_scope(name='create_embeddings'):
            '''此處train_dataset和train_labels是文本數字化的ID，
                也就是上面generate_batch_skip_gram()傳回的batch, labels'''             
            train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size])               
            train_labels = tf.placeholder(tf.int32, shape=[self.batch_size])
             
            # weights_Xij measures the importance of a data point with 
            # respect to how much those two words co-occur
            weights_Xij = tf.placeholder(tf.float32,shape=[self.batch_size],\
                               name='weights_Xij') 
            # Cooccurence value for the position; cooccurence matrix weights
            Xij = tf.placeholder(tf.float32,shape=[self.batch_size],name='Xij')                         
             
            ''' Defining Model Parameters which is composed of an
             embedding layer, a bias for each input and output words.'''
            in_embeddings = tf.Variable(tf.random_uniform(\
              [self.vocab_size, self.embedding_size],-1.0,1.0),\
                 name='in_embeddings')
            in_bias_embeddings = tf.Variable(tf.random_uniform(\
                [self.vocab_size],0.0,0.01,dtype=tf.float32),\
                 name='in_bias_embeddings')    
            out_embeddings = tf.Variable(tf.random_uniform(\
                [self.vocab_size,self.embedding_size],-1.0,1.0),\
                 name='out_embeddings')
            out_bias_embeddings = tf.Variable(tf.random_uniform(\
                [self.vocab_size],0.0,0.01,dtype=tf.float32),\
                 name='out_bias_embeddings')
             
            # Look up embeddings for inputs and outputs
            # Have two seperate embedding vector spaces for inputs and outputs
            embed_in = tf.nn.embedding_lookup(in_embeddings, train_dataset)
            embed_out = tf.nn.embedding_lookup(out_embeddings, train_labels)
            embed_bias_in = tf.nn.embedding_lookup(in_bias_embeddings,\
                                           train_dataset)
            embed_bias_out=tf.nn.embedding_lookup(out_bias_embeddings,\
                                            train_labels)
             
        with tf.name_scope(name='loss'):
            loss = tf.reduce_mean(weights_Xij * (tf.reduce_sum(
                     embed_in * embed_out, axis=1) + embed_bias_in +\
                     embed_bias_out - tf.log(epsilon+Xij))**2)
                 
         
        with tf.name_scope(name = 'optimizer'):
            optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
            
        with tf.name_scope(name='valid_dataset'):
            # Validation input data, we don't need a placeholder as we have
            # already defined the IDs of the words selected as validation data.
            valid_examples = np.array(random.sample\
                              (range(self.valid_window), self.valid_size))
            valid_examples = np.append(valid_examples,\
                random.sample(range(1000, 1000 + self.valid_window),\
                        self.valid_size),axis=0)
            '''valid_dataset 隨機選擇用來驗證的 ID dataset被tf.constant固定
             即，每次驗證的valid_examples是同一組，除非重執行valid_examples'''
            valid_dataset = tf.constant(valid_examples, dtype = tf.int32)
            self.valid_examples = valid_examples
            
        with tf.name_scope(name = 'similarity'):             
            # Compute the similaruty between minibatch examples and all embeddings.
            # We use the cosine distance:
            # Embedding Vector都是學習過的值，所以相似性都是 valid_dataset
            # 對所有字學習完的 Embedding Vector的cosine distance比較
            # Embedding Vector 正規化（歸一化，normalization）
            # reduce_sum(tensors, axis, keepdims)的reduce有降維作用
            # 因此keepdims=True則保持維度不降維
            '''此處embeddings取in_embeddings 和 out_embeddings平均'''
            embeddings = (in_embeddings + out_embeddings) / 2.0
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1,\
                            keepdims=True))
            normalized_embeddings = embeddings/norm
             # embedding_lookup(tensor, id)在tensor中查找id對應的列或行元素
             # 其中valid_dataset 隨機選擇用來驗證的 ID dataset
             # 找到的正規化的embedding vector(已學習改變的vector)用來計算相似性similarity
             # similarity = cosine distance= A‧B/|A||B| 因經正規化,所以|A|=|B|=1，
             # => similarity = A‧B (矩陣內積)（0度最相似cos(0)=1 => A=B）
             #  此處A為vaild example的embedding vector，B為所有embedding vector
             # 其shape [valid_examples size, vocab_size] (32, 50000)
             # 故，similarity意味valid_examples的字對vocabulary所有字計算
             # cosine distance    
            valid_embeddings = tf.nn.embedding_lookup(\
                               normalized_embeddings, valid_dataset)                   
            similarity = tf.matmul(valid_embeddings,tf.transpose(\
                                normalized_embeddings))
        init = tf.global_variables_initializer()
        saver= tf.train.Saver()
        self._train_dataset = train_dataset
        self._train_labels = train_labels
        self._weights_Xij = weights_Xij
        self._Xij = Xij
        self._optimizer, self._loss = optimizer, loss
        self._similarity = similarity
        self._normalized_embeddings = normalized_embeddings
        self._init, self._saver = init, saver
        
    def close_session(self):
        if self._session:
            self._session.close()    
    
    '''
    ### Running the GloVe Algorithm
    
     Here we run the GloVe algorithm we defined above. 
     Specifically, we first initialize variables, and then train
     the algorithm for many steps (num_steps). And every few
     steps we evaluate the algorithm on a fixed validation
     set and print out the words that appear to be closest for
     a given set of words.
    '''
    
    def fit(self, text_digt, reverse_dictionary,
            batch_size, window_size, name='fit'):        
        self.close_session()
        self.dataset_size = len(text_digt)         
        glove_loss = []        
        average_loss = 0
        cooc_mat, dt_id = self._generate_cooc(text_digt)
        print('co-occurance type:', type(cooc_mat))
        print('co-occurance last data index', dt_id )
        
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph()
        config=tf.ConfigProto(allow_soft_placement=True)
        self._session = tf.Session(graph=self._graph, config=config)
        with self._session.as_default() as sess:                               
            self._init.run()
            print('Initialized')
            data_index = 0            
            for step in range(self.num_steps):                
                # generate a single batch (data,labels,co-occurance weights)
                batch_input, batch_labels, batch_weights, data_index =\
                 self._generate_batch_GloVe(text_digt,
                                            batch_size,
                                            window_size) 
                
                # Computing the weights required by the loss function
                batch_weights = [] # weighting used in the loss function
                batch_xij = [] # weighted frequency of finding i near j
                
                # Compute the weights for each datapoint in the batch
                ''' 統計同時存在weight的cooc_mat已在 generate_cooc建立
                    此處batch_input及batch_labels只是用來在此batch段text中
                    每個inp(中心詞), lbl(背景詞)對裡查找其對應的weights_co'''
                for inp,lbl in zip(batch_input,batch_labels.reshape(-1)):     
                    point_weight = (cooc_mat[inp,lbl]/100.0)**0.75\
                        if cooc_mat[inp,lbl]<100.0 else 1.0 
                    batch_weights.append(point_weight) # 調整後的batch weight
                    batch_xij.append(cooc_mat[inp,lbl]) # 未調整的batch weight
                batch_weights = np.clip(batch_weights,-100,1) #每值取最小-100,最大1
                batch_xij = np.asarray(batch_xij)
                
                # Populate the feed_dict and run the optimizer (minimize loss)
                # and compute the loss. Specifically we provide
                # train_dataset/train_labels: training inputs and training labels
                # weights_Xij: measures the importance of a data point with respect to how much those two words co-occur
                # Xij: co-occurence matrix value for the row and column denoted by the words in a datapoint
                feed_dict = {self._train_dataset : batch_input.reshape(-1),
                        self._train_labels : batch_labels.reshape(-1),
                        self._weights_Xij:batch_weights, self._Xij:batch_xij}
                _, loss_val = sess.run([self._optimizer, self._loss],
                                          feed_dict=feed_dict)
                
                # Update the average loss variable
                average_loss += loss_val
                if step % 2000 == 0:
                  if step > 0:
                    average_loss = average_loss / 2000
                  # The average loss is an estimate of the loss over the last 2000 batches.
                  print('\nAverage loss at step %d: %f' % (step, average_loss))
                  glove_loss.append(average_loss)
                  print('data_index:', data_index)
                  average_loss = 0
                
                # Here we compute the top_k closest words for a given validation word
                # in terms of the cosine distance
                # We do this for all the words in the validation set
                # Note: This is an expensive step
                if step % 10000 == 0:
                  sim = self._similarity.eval()
                  for i in range(self.valid_size):
                    valid_word = reverse_dictionary[self.valid_examples[i]]
                    top_k = 8 # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = 'Nearest to "%s":' % valid_word
                    for k in range(top_k):
                      close_word = reverse_dictionary[nearest[k]]
                      log = '%s %s,' % (log, close_word)
                    print(log)
                
            glove_final_embeddings = self._normalized_embeddings.eval()
        
        # We will save the word vectors learned and the loss over time
        # when this information is required later for comparisons.
        np.save('glove_embeddings', glove_final_embeddings)

        with open('glove_loss.csv', 'wt') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(glove_loss)    
        
          