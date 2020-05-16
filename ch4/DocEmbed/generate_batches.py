# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 18:37:23 2020

@author: Peter Hsu
"""

from __future__ import division, print_function, unicode_literals
import collections
import numpy as np
from build_dataset_zip import bbcZipDoc2Dset

class genBatch(object):
    def __init__(self):                 
                 
        """Constructor.
        Args:
            vocabulary_size: maximum  vocabularysize 
        """        
        #self.trn_idx = 0
        #self.vld_idx = 0  
        
    def train_batch_skip_gram(self, text_digt, train_batch_size,
                 train_window_size, trn_idx):       
        
        # trn_idx is updated by 1 everytime we read a data point
              
        # span defines the total window size, where
        # text_digt we consider at an instance looks as follows. 
        # [ skip_window target skip_window ]
        span = 2 * train_window_size + 1
        
        # two numpy arras to hold target words (batch)
        # and context words (labels)
        train_batch =np.ndarray(shape=(train_batch_size,span-1),dtype=np.int32)
        train_labels = np.ndarray(shape=(train_batch_size, 1), dtype=np.int32)
        
        # The buffer holds the text_digt contained within the span
        buffer = collections.deque(maxlen=span)
    
        # Fill the buffer and update the data_index
        for _ in range(span):
            buffer.append(text_digt[trn_idx])
            trn_idx = (trn_idx + 1) % len(text_digt)
    
        # This is the number of context words we sample for a single target word
        num_samples = 2 * train_window_size 
    
        # We break the batch reading into two for loops
        # The inner for loop fills in the batch and labels with 
        # num_samples text_digt points using text_digt contained withing the span
        # The outper for loop repeat this for batch_size//num_samples times
        # to produce a full batch
        for i in range(train_batch_size // num_samples):
          k=0
          # avoid the target word itself as a prediction
          # fill in batch and label numpy arrays
          for j in list(range(train_window_size))+list(
                  range(train_window_size+1, 2*train_window_size+1)):
            train_batch[i * num_samples + k] = buffer[train_window_size]
            train_labels[i * num_samples + k, 0] = buffer[j]
            k += 1 
        
          # Everytime we read num_samples data points,
          # we have created the maximum number of datapoints possible
          # withing a single span, so we need to move the span by 1
          # to create a fresh new span
          buffer.append(text_digt[trn_idx])
          trn_idx = (trn_idx + 1)%len(text_digt)          
         
        assert train_batch.shape[0]==train_batch_size and\
               train_batch.shape[1]== span-1
        
        return train_batch, train_labels, trn_idx     

    def test_batch(self, test_text_dict, test_batch_size, test_idx):
        '''
        Generate a batch of data from the validation data
        This is used to compute the document embedding
        by taking the average of all the words in a document
        '''                 
        
        testbatch = np.ndarray(shape=(test_batch_size,), dtype=np.int32)
        # Get words starting from index 0 to span
        for bi in range(test_batch_size):
            testbatch[bi] = test_text_dict[test_idx]
            test_idx = (test_idx + 1) % len(test_text_dict)
    
        return testbatch, test_idx 
    
    
    def train_batch_cbow(self, text_digt, train_batch_size,
                     train_window_size, trn_idx ):
        # window_size is the amount of words we're looking at from each side of a given word
        # creates a single batch
        
        # trn_idx is updated by 1 everytime we read a set of data point
        # span defines the total window size, where
        # data we consider at an instance looks as follows. 
        # [ skip_window target skip_window ]
        # e.g if skip_window = 2 then span = 5
        span = 2 * train_window_size + 1 # [ skip_window target skip_window ]
    
        # two numpy arras to hold target words (batch)
        # and context words (labels)
        # Note that batch has span-1=2*window_size columns
        batch = np.ndarray(shape=(train_batch_size,span-1), dtype=np.int32)
        labels = np.ndarray(shape=(train_batch_size, 1), dtype=np.int32)
        
        # The buffer holds the data contained within the span
        buffer = collections.deque(maxlen=span)
    
        # Fill the buffer and update the data_index
        for _ in range(span):
            buffer.append(text_digt[trn_idx])
            trn_idx = (trn_idx + 1) % len(text_digt)
    
        # Here we do the batch reading
        # We iterate through each batch index
        # For each batch index, we iterate through span elements
        # to fill in the columns of batch array
        for i in range(train_batch_size):
            target = train_window_size  # target label at the center of the buffer
            #target_to_avoid = [train_window_size] # we only need to know the words around a given word, not the word itself
    
            # add selected target to avoid_list for next time
            col_idx = 0
            for j in range(span):
                # ignore the target word when creating the batch
                if j==span//2:
                    continue
                batch[i,col_idx] = buffer[j] 
                col_idx += 1
            labels[i, 0] = buffer[target]
    
            # Everytime we read a data point,
            # we need to move the span by 1
            # to create a fresh new span
            buffer.append(text_digt[trn_idx])
            trn_idx = (trn_idx + 1) % len(text_digt)
    
        return batch, labels, trn_idx

   
# In[]:
"""
bbcDoc = bbcZipDoc2Dset()
filename = 'bbc-fulltext.zip'

print('Processing training data...')
words = bbcDoc.read_data(filename,expected_bytes=2874078) # expected_bytes to verify
print('\nProcessing validation data...')
valid_words = bbcDoc.read_valid_data(filename)

print('Example words (start): ',words[:10])
print('Example words (end): ',words[-10:])

# Processining training data
text_digt, count, dictionary, reverse_dictionary =\
         bbcDoc.build_dataset(words)

# Processing validation data
valid_text_digt = {}
for k,v in valid_words.items():
    print('Building Validation Dataset for ',k,' topic')
    valid_text_digt[k] =\
    bbcDoc.build_valid_dataset_with_existing_dictionary(
            valid_words[k], dictionary)
    
print('Most common words (+UNK)', count[:5])
print('Sample data', text_digt[:10])
print('valid keys: ', valid_text_digt.keys())
del words  # Hint to reduce memory.
del valid_words

skpGmBh = genBatch()
print('\ntext_digt:', [reverse_dictionary[di] for di in text_digt[:10]])  
'''
for window_size in [1,2]:
    trn_idx = 0          
    train_batch, train_labels, train_index =skpGmBh.train_batch_skip_gram(
        text_digt, train_batch_size=8, train_window_size=window_size)
    print('with window_size = %d:' % (window_size))
    print(' train batch:', [[reverse_dictionary[bii] for bii in bi] 
                            for bi in train_batch])
    print(' train labels:', [reverse_dictionary[li] for li in 
                             train_labels.reshape(8)])
    print('train_index:',train_index)
'''

for window_size in [1,2]:
    trn_idx = 0
    train_batch, train_labels, train_index = skpGmBh.train_batch_cbow(
        text_digt, train_batch_size=8, train_window_size=window_size)
    print('\nwith window_size = %d:' % (window_size))
    print('    train_batch:', [[reverse_dictionary[bii] for bii in bi]
                for bi in train_batch])
    print('    train_labels:', [reverse_dictionary[li] for li in
                                train_labels.reshape(8)])
    print('train_index:',train_index)
    
vld_idx = 0
print()
vtd =valid_text_digt[list(valid_text_digt.keys())[0]]
print('list(valid_text_digt.keys())[0]:',list(valid_text_digt.keys())[0])
print('[list(valid_text_digt.keys())[0]]:',[list(valid_text_digt.keys())[0]])
print('valid_text_digt[list(valid_text_digt.keys())[0]]:', valid_text_digt
        [list(valid_text_digt.keys())[0]])  
valid_batch, valid_index = skpGmBh.valid_batch_skip_gram(valid_text_digt
        [list(valid_text_digt.keys())[0]], valid_batch_size=8)
print('valid_text_digt:\n',[reverse_dictionary[bi] for bi in 
            valid_text_digt[list(valid_text_digt.keys())[0]][:10]])      
print('\nwith window_size = %d:' % (window_size))
print('  labels:', [reverse_dictionary[li] for li in valid_batch.reshape(8)])
print('valid_index:',valid_index)
"""