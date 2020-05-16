# -*- coding: utf-8 -*-
"""
Created on Sun Mar  1 15:40:12 2020

@author: Peter Hsu
"""
# In[1]:

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
#%matplotlib inline
from __future__ import division, print_function, unicode_literals
import collections
import os
import bz2
from math import ceil
from six.moves.urllib.request import urlretrieve
import nltk # standard preprocessing
import pickle

class word2VecDataset(object):
    
# # Dataset
# This code downloads a dataset consisting of several Wikipedia articles 
# totaling up to roughly 61 megabytes. Additionally the code 
# makes sure the file has the correct size after downloading it.
    def __init__(self,
                 url = 'http://www.evanjones.ca/software/',
                 vocabulary_size=50000):
        """Constructor.
        Args:
            vocabulary_size: maximum  vocabularysize 
        """
        self.url = url 
        self._vocabulary_size = vocabulary_size
    
        
    ''' Read Data with Preprocessing with NLTK'''
    ''' Reads data as it is to a string, convert to lower-case and  
     tokenize it using the nltk library.'''
    def read_data(self, filename, expected_bytes):
      """
      Extract the first file enclosed in a zip file as a list of 
      words and pre-processes it using the nltk python library.
      """
      """Download a file if not present, and make sure it's the right size."""
      if not os.path.exists(filename):
        print('Downloading file...')
        filename, _ = urlretrieve(self.url + filename, filename)
      
      statinfo = os.stat(filename)
      if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
      else:
        print(statinfo.st_size)
        raise Exception(
          'Failed to verify '+filename+'. Can you get to it with a browser?')
      
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
    
    #words = read_data(filename)
    #print('Data size %d' % len(words))
    #print('Example words (start):', words[:10])
    #print('Example words (end):', words[-10:])
    
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
    
    # we restrict our vocabulary size to 50000
    #vocabulary_size = 50000
    
    def build_dataset(self, words):
        count = [['UNK', -1]]
        # Gets only the vocabulary_size most common words as the vocabulary
        # All the other words will be replaced with UNK token        
        count.extend(collections.Counter(words).most_common(\
                     self._vocabulary_size-1))
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
        assert len(dictionary) == self._vocabulary_size 
        
        # pkl file適合儲存dictionary        
        output = open('reverse_dict.pkl', 'wb')
        pickle.dump(reverse_dictionary, output)
        output.close()             
        
        return text_digt, count, dictionary, reverse_dictionary

# In[6]:
'''
url = 'http://www.evanjones.ca/software/'
w2v_dataset= word2VecDataset()    
words = w2v_dataset.read_data(\
            '../wikipedia2text-extracted.txt.bz2', 18377035)
print('Data size %d' % len(words))
print('Example words (start):', words[:10])
print('Example words (end):', words[-10:])    
text_digt,count,dictionary,reverse_dictionary =\
        w2v_dataset.build_dataset(words)
print('Most common words (+UNK): ', count[:10])
print('words:', words[:10])
print('Sample text_digitize:', text_digt[:10])
del words'''