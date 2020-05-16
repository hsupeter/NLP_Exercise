# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:51:40 2020

@author: Peter Hsu
"""
from __future__ import print_function
import math
import numpy as np
import random
import tensorflow as tf
import sys
import csv
import pickle
from generate_batches import genBatch 

class word2VectorModel(object):
    def __init__(self,                 
                 batch_size=128,
                 window_size = 4,
                 num_steps = 100001,                 
                 embedding_size = 128,                 
                 valid_size = 16,
                 valid_window = 50,
                 num_sampled = 32,
                 vocab_size = 25000,
                 data_index = 0
                 ):
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.window_size = window_size
        self.num_steps = num_steps        
        self.vocab_size = vocab_size        
        self.embedding_size = embedding_size # Dimension of the embedding vector.
        self.valid_size = valid_size # Random set of words to evaluate similarity.
        self.valid_window = valid_window
        self.num_sampled = num_sampled # Number of negative examples to sample.        
        self.data_index = data_index
        self._session = None
 
# When selecting valid examples, we select some of the most frequent words as well as
# some moderately rare words as well

    def _build_graph(self, name='build_graph'):
        # ### Defining Inputs and Outputs
        # Here we define placeholders for feeding in training inputs 
        # and outputs (each of size `batch_size`) and a constant tensor 
        # to contain validation examples.
        with tf.name_scope(name='def_IO'):
            '''此處train_input和train_labels是文本數字化的ID，
                也就是上面generate_batch_skip_gram()傳回的batch, labels'''             
            train_dataset = tf.placeholder(tf.int32,
                            shape=[self.batch_size,2*self.window_size])             
            train_labels = tf.placeholder(tf.int32,
                            shape=[self.batch_size, 1])
            embeddings = tf.Variable(tf.random_uniform(
                         [self.vocab_size, self.embedding_size],
                         -1.0, 1.0,dtype=tf.float32))
            softmax_weights = tf.Variable(tf.truncated_normal(
                        [self.vocab_size, self.embedding_size],
                        stddev=1.0 / math.sqrt(self.embedding_size),
                        dtype=tf.float32))
            softmax_biases = tf.Variable(tf.zeros(
                        [self.vocab_size],dtype=tf.float32))
        
        with tf.name_scope(name='create_embeddings'):
            # Model.
            # Look up embeddings for all the context words of the inputs.
            # Then compute a tensor by staking embeddings of all context words
            stacked_embedings = None
            print('Defing %d embed lookup\
             representing each word in the context'%(2*self.window_size))
            for i in range(2*self.window_size):
                embedding_i = tf.nn.embedding_lookup(embeddings,
                                 train_dataset[:,i])        
                x_size,y_size = embedding_i.get_shape().as_list()
                if stacked_embedings is None:
                    stacked_embedings = tf.reshape(embedding_i,
                                           [x_size,y_size,1])
                else:
                    stacked_embedings = tf.concat(axis=2,values=[
                    stacked_embedings,tf.reshape(embedding_i,
                                         [x_size,y_size,1])])
            
            # Make sure the staked embeddings have 2*window_size columns
            assert stacked_embedings.get_shape().as_list()[2]==\
                    2 * self.window_size
            print("Stacked embedding size: %s"
                    %stacked_embedings.get_shape().as_list())            
            # Compute mean embeddings by taking the mean of the tensor 
            # containing the stack of embeddings
            mean_embeddings =  tf.reduce_mean(stacked_embedings,
                                  2,keepdims=False)
            print("Reduced mean embedding size: %s"
                    %mean_embeddings.get_shape().as_list())            
            
        with tf.name_scope(name='loss_optimizer'):
            loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(
                weights=softmax_weights, biases = softmax_biases,
                inputs = mean_embeddings, labels=train_labels,
                num_sampled = self.num_sampled,
                num_classes= self.vocab_size))
            optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)
        
        with tf.name_scope(name='valid_dataset'):            
            valid_examples = np.array(random.sample(
                    range(self.valid_window), self.valid_size))
            valid_examples = np.append(valid_examples, random.sample(
                       range(1000, 1000+self.valid_window),
                       self.valid_size), axis=0)

            '''valid_dataset 隨機選擇用來驗證的 ID dataset被tf.constant固定
             即，每次驗證的valid_examples是同一組，除非重執行valid_examples'''
            valid_dataset = tf.constant(valid_examples, dtype=tf.int32)
        
        with tf.name_scope(name='test_data'):
             
            test_labels = tf.placeholder(tf.int32,
                             shape=[self.batch_size], name='test_dataset')            
            # Used to compute document embeddings by averaging all the word vectors of a 
            # given batch of test data
            mean_batch_embedding = tf.reduce_mean(tf.nn.embedding_lookup(
                    embeddings, test_labels), axis=0)
            
                
        with tf.name_scope(name = 'similarity'):
            # Compute the similarity between minibatch 
            # examples and all embeddings.
            # We use the cosine distance:                       
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings),
                             1, keepdims=True))
            normalized_embeddings = embeddings / norm
            valid_embeddings = tf.nn.embedding_lookup(
                               normalized_embeddings, valid_dataset)
            similarity = tf.matmul(valid_embeddings,
                               tf.transpose(normalized_embeddings))

            
        init = tf.global_variables_initializer()
        saver= tf.train.Saver()
        self._test_labels = test_labels
        self._mean_batch_embedding = mean_batch_embedding
        self._valid_examples = valid_examples
        self._train_dataset = train_dataset
        self._train_labels = train_labels 
        self._optimizer, self._loss = optimizer, loss
        self._similarity = similarity
        self._normalized_embeddings = normalized_embeddings
        self._init, self._saver = init, saver
        
    def close_session(self):
        if self._session:
            self._session.close()
    
    # ## Running the CBOW Algorithm on Document Data
    # 
    # Here we run the CBOW algorithm we defined above. Specifically,
    # we first initialize variables, and then train the algorithm
    # for many steps (`num_steps`). And every few steps we evaluate
    # the algorithm on a fixed validation set and print out the 
    # words that appear to be closest for a given set of words.  

    def fit(self, text_digt, test_text_dict, reverse_dictionary, 
        batch_size, window_size, w2v_model=1, name='fit'):
        if w2v_model == 1:
            print('\n    Skip Gram Model \n')
        elif w2v_model == 2:
            print('\n    CBOW Model \n')
        else:            
            print('\n You have to set w2v_model= ')
            print(' 1 is Skip Gram Model; 2 is CBOW Model \n')
            sys.exit(0)        
        self.close_session()
        self.dataset_size = len(text_digt)               
        loss_value = []
        config=tf.ConfigProto(allow_soft_placement=True)    
        config.gpu_options.allow_growth = True    
        self._graph = tf.Graph()
        with self._graph.as_default():
            self._build_graph()
        config=tf.ConfigProto(allow_soft_placement=True)
        self._session = tf.Session(graph=self._graph, config=config)
        with self._session.as_default() as sess:                               
            self._init.run()            
            print('Initialized')        
            average_loss = 0 
            trn_idx = 0
            # Train the Word2vec model for num_step iterations
            gnBh = genBatch()
            for step in range(self.num_steps):            
                # Generate a single batch of data
                if w2v_model == 1:
                    train_batch, train_labels, trn_idx =\
                        gnBh.train_batch_skip_gram(text_digt,
                        train_batch_size=batch_size,
                        train_window_size = window_size,
                        trn_idx = trn_idx)
                elif w2v_model == 2:
                    train_batch, train_labels, trn_idx =\
                        gnBh.train_batch_cbow(text_digt,
                        train_batch_size=batch_size,
                        train_window_size = window_size,
                        trn_idx = trn_idx)                        
                 # Populate the feed_dict and run the optimizer (minimize loss)
                # and compute the loss
                feed_dict = {self._train_dataset : train_batch,
                             self._train_labels : train_labels}
                _, loss_val = sess.run([self._optimizer, self._loss],
                              feed_dict=feed_dict)
                
                # Update the average loss variable
                average_loss += loss_val
                
                if (step+1) % 2000 == 0:
                    if step > 0:
                        average_loss = average_loss / 2000
                        # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step %d: %f' % (step+1, average_loss))
                    loss_value.append(average_loss)
                    average_loss = 0
                
                # Evaluating validation set word similarities
                if step % 10000 == 0:
                  sim = self._similarity.eval()
                  for i in range(self.valid_size):
                    valid_word = reverse_dictionary[self._valid_examples[i]]
                    top_k = 8 # number of nearest neighbors
                    nearest = (-sim[i, :]).argsort()[1:top_k+1]
                    log = 'Nearest to "%s":' % valid_word
                    for k in range(top_k):
                      close_word = reverse_dictionary[nearest[k]]
                      log = '%s %s,' % (log, close_word)
                    print(log)
       
            # Computing test documents embeddings by averaging word embeddings
            
            # We take batch_size*num_test_steps words from each document
            # to compute document embeddings
            '''
            1.傳入已產生為dict的test_data，每個k(key)為測試項(e.g. tech-56)
                的值test_data[k]
            2.傳入generate_test_batch產生batch_size大小的test_batch_labels
              （字典字的index）
            3.傳入mean_batch_embedding藉由tf.nn.embedding_lookup按已訓練的
             embeddings查表求每column的embedding vector(axis=0)的平均得batch_mean
               （注意,不是每個字(axis=1)的平均）
            4.傳入記錄在topic_mean_batch_embeddings（以axis=0紀錄）
            5.重複1~4步驟num_test_steps次後再取平均放document_embeddings
            6.再改變k再重複1~5步驟記錄不同topic的document_embeddings
            '''
            gnBh_t = genBatch()
            num_test_steps = 100        
            # Store document embeddings
            # {document_id:embedding} format
            document_embeddings = {}
            print('Testing Phase (Compute document embeddings)')
            
            # For each test document compute document embeddings
            for k,v in test_text_dict.items():
                print('\tCalculating mean embedding for document ',
                      k,' with ', num_test_steps, ' steps.')
                test_idx = 0
                topic_mean_batch_embeddings = np.empty((
                        num_test_steps, self.embedding_size)
                        ,dtype=np.float32)            
                # keep averaging mean word embeddings obtained for each step
                for test_step in range(num_test_steps):
                    test_batch_labels, test_idx = gnBh_t.test_batch(
                       test_text_dict[k], batch_size, test_idx = test_idx)
                    batch_mean = sess.run(self._mean_batch_embedding,
                        feed_dict={self._test_labels:test_batch_labels})
                    topic_mean_batch_embeddings[test_step,:] = batch_mean
                document_embeddings[k] = np.mean(
                                    topic_mean_batch_embeddings,axis=0)
    
        output = open('w2v_embeddings.pkl', 'wb')
        pickle.dump(document_embeddings, output)
        output.close() 
        
        with open('loss_value.csv', 'wt') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow(loss_value)    
   