# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 16:25:02 2020

@author: Peter Hsu
"""

# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
#%matplotlib inline
from __future__ import division, print_function, unicode_literals
import collections
import os
import zipfile
import numpy as np
from six.moves.urllib.request import urlretrieve
import nltk # standard preprocessing
import pickle

class bbcZipDoc2Dset(object):
    ''' ###Dataset
      This code downloads a dataset consisting of several BBC news
      articles belonging to various categories (e.g. sport, politics,
      etc.). Additionally the code makes sure the file has the 
      correct size after  downloading it.'''
    def __init__(self,
                 url = 'http://mlg.ucd.ie/files/datasets/',
                 files_to_read_for_topic = 250,
                 vocabulary_size = 25000
                 ):
        """Constructor.
        Args:
            vocabulary_size: maximum  vocabularysize 
        """
        self.url = url
        self.files_to_read_for_topic = files_to_read_for_topic
        self.vocabulary_size = vocabulary_size
        
        ''' Read Data with Preprocessing with NLTK'''
        ''' Reads data as it is to a string, convert to lower-case and  
         tokenize it using the nltk library.'''
    
    def read_data(self, filename, expected_bytes):
        
        """Download a file if not present, and make sure it's the right size."""
        if not os.path.exists(filename):
            filename, _ = urlretrieve(self.url + filename, filename)
        
        statinfo = os.stat(filename)
        if statinfo.st_size == expected_bytes:
            print('Found and verified %s' % filename)
        else:
            print(statinfo.st_size)
            raise Exception('Failed to verify ' + filename +
              '. Can you get to it with a browser?')
            
        data = []        
        topics = ['business','entertainment','politics','sport','tech']
        with zipfile.ZipFile(filename) as z:
            parent_dir = z.namelist()[0]
            for t in topics:
                print('\tFinished reading data for topic: ',t)
                for fi in range(1,self.files_to_read_for_topic):
                    with z.open(parent_dir + t + '/'+ format(fi,'03d')+'.txt') as f:
                        file_string = f.read().decode('latin-1')
                        file_string = file_string.lower()
                        file_string = nltk.word_tokenize(file_string)
                        data.extend(file_string)                
        
        return data    
    

    def read_valid_data(self, filename):
      """
      Extract articles up to a given threshold in a zip file as a list of words
      and pre-processes it using the nltk python library
      """
      valid_data = {}      
      topics = ['business','entertainment','politics','sport','tech']
      with zipfile.ZipFile(filename) as z:
        parent_dir = z.namelist()[0]
        for t in topics:
            print('\tFinished reading data for topic: ',t)
                
            for fi in np.random.randint(1,
                self.files_to_read_for_topic,(10)).tolist():
                with z.open(parent_dir + t + '/'+
                            format(fi,'03d')+'.txt') as f:
                    file_string = f.read().decode('latin-1')
                    file_string = file_string.lower()
                    file_string = nltk.word_tokenize(file_string)
                    valid_data[t+'-'+str(fi)] = file_string
                    
      return valid_data
    
    '''Building the Dictionaries
    
    Builds the following. To understand each of these elements,
    let us also assume the text "I like to go to school"
        dictionary: maps a string word to an ID (e.g. 
                    {I:0, like:1, to:2, go:3, school:4})
        reverse_dictionary: maps an ID to a string word (e.g.
                    {0:I, 1:like, 2:to, 3:go, 4:school}
        count: List of list of (word, frequency) elements (e.g.
                    [(I,1),(like,1),(to,2),(go,1),(school,1)]
        data : Contain the string of text we read, where string
               words are replaced with word IDs 
               (e.g. [0, 1, 2, 3, 2, 4])
     It also introduces an additional special token UNK to denote rare words
     to are too rare to make use of.'''
     

    def build_dataset(self, words):
      # Allocate a special token for rare words
      count = [['UNK', -1]]
    
      # Gets only the vocabulary_size most common words as the vocabulary
      # All the other words will be replaced with UNK token
      count.extend(collections.Counter(words).most_common(
                   self.vocabulary_size - 1))
    
      # Create an ID for each word by giving the current length of the dictionary
      # And adding that item to the dictionary
      dictionary = dict()
      for word, _ in count:
        dictionary[word] = len(dictionary)
        
      text_digt = list()
      unk_count = 0
        
      # Traverse through all the text we have and produce a list
      # where each element corresponds to the ID of the word found at that index
      for word in words:
        # If word is in the dictionary use the word ID,
        # else use the ID of the special token "UNK"
        if word in dictionary:
          index = dictionary[word]
        else:
          index = 0  # dictionary['UNK']
          unk_count = unk_count + 1
        text_digt.append(index)
      
      # update the count variable with the number of UNK occurences
      count[0][1] = unk_count
        
      reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys())) 
      # Make sure the dictionary is of size of the vocabulary
      assert len(dictionary) == self.vocabulary_size
      # pkl file適合儲存dictionary
      output = open('bbcDocRevDict.pkl', 'wb')
      pickle.dump(reverse_dictionary, output)
      output.close()
      
      return text_digt, count, dictionary, reverse_dictionary

    def build_valid_dataset_with_existing_dictionary(self, words, dictionary):
        '''
        Here we use this function to convert word strings to IDs
        with a given dictionary
        '''
        valid_text_digt = list()
        for word in words:
            if word in dictionary:
              index = dictionary[word]
            else:
              index = 0  # dictionary['UNK']
            valid_text_digt.append(index)
        return valid_text_digt
     
# In[]:
'''
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
    print('Building validation Dataset for ',k,' topic')
    valid_text_digt[k] =\
    bbcDoc.build_valid_dataset_with_existing_dictionary(
            valid_words[k], dictionary)
    
print('Most common words (+UNK)', count[:5])
print('Sample data', text_digt[:10])
print('validation keys: ',valid_text_digt.keys())
del words  # Hint to reduce memory.
del valid_words
'''
