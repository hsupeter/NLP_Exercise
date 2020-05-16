# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 17:26:01 2020

@author: Peter Hsu
"""
# In[]:
import pickle
from build_dataset_zip import bbcZipDoc2Dset
from generate_batches import genBatch 
from word2vector_model import word2VectorModel
from tsne_plot_kmean import tsnePltKmean

bbcDoc = bbcZipDoc2Dset()
filename = 'bbc-fulltext.zip'

print('Processing training data...')
words = bbcDoc.read_data(filename,expected_bytes=2874078) # expected_bytes to verify
print('\nProcessing validation data...')
test_words = bbcDoc.read_valid_data(filename)

print('Example words (start): ',words[:10])
print('Example words (end): ',words[-10:])

# Processining training data
text_digt, count, dictionary, reverse_dictionary =\
         bbcDoc.build_dataset(words)

# Processing validation data
test_text_dict = {}
for k,v in test_words.items():
    print('Building Validation Dataset for ',k,' topic')
    test_text_dict[k] =\
    bbcDoc.build_valid_dataset_with_existing_dictionary(
            test_words[k], dictionary)
    
print('Most common words (+UNK)', count[:5])
print('Sample data', text_digt[:10])
print('valid keys: ', test_text_dict.keys())
del words  # Hint to reduce memory.
del test_words

skpGmBh = genBatch()
print('\ntext_digt:', [reverse_dictionary[di] for di in text_digt[:10]])  
'''
for window_size in [1,2]:
    trn_idx = 0          
    train_batch, train_labels, train_index =skpGmBh.train_batch_skip_gram(
        text_digt, train_batch_size=8, train_window_size=window_size,
        trn_idx = trn_idx)
    print('with window_size = %d:' % (window_size))
    print(' train batch:', [[reverse_dictionary[bii] for bii in bi] 
                            for bi in train_batch])
    print(' train labels:', [reverse_dictionary[li] for li in 
                             train_labels.reshape(8)])
    print('train_index:',train_index)
'''
trn_idx = 0
for window_size in [1,2]:     
    print('train_index0:', trn_idx)
    train_batch, train_labels, trn_idx = skpGmBh.train_batch_cbow(
        text_digt, train_batch_size=8, train_window_size=window_size,
        trn_idx = trn_idx)
    print('\nwith window_size = %d:' % (window_size))
    print('    train_batch:', [[reverse_dictionary[bii] for bii in bi]
                for bi in train_batch])
    print('    train_labels:', [reverse_dictionary[li] for li in
                                train_labels.reshape(8)])
    print('train_index1:', trn_idx)

print()
test_idx = 0
vtd =test_text_dict[list(test_text_dict.keys())[0]]
#print('list(valid_text_digt.keys())[0]:',list(valid_text_digt.keys())[0])
#print('[list(valid_text_digt.keys())[0]]:',[list(valid_text_digt.keys())[0]])
#print('valid_text_digt[list(valid_text_digt.keys())[0]]:', valid_text_digt
#        [list(valid_text_digt.keys())[0]])  
test_batch, test_index = skpGmBh.test_batch(test_text_dict
        [list(test_text_dict.keys())[0]], test_batch_size=8,
        test_idx = test_idx)
print('valid_text_digt:\n',[reverse_dictionary[bi] for bi in 
            test_text_dict[list(test_text_dict.keys())[0]][:10]])      
print('\nwith window_size = %d:' % (window_size))
print('  labels:', [reverse_dictionary[li] for li in test_batch.reshape(8)])
print('test_index:',test_index)

w2vm= word2VectorModel(num_steps = 100001)           
w2vm.fit(text_digt, test_text_dict, reverse_dictionary,
         batch_size =128, window_size=4, w2v_model=2)


reverse_dictionary = {}
prime_items = {}
pkl_rev_dict = open('bbcDocRevDict.pkl', 'rb')
reverse_dictionary = pickle.load(pkl_rev_dict)
pkl_rev_dict.close()
print('Reverse Dictionary Done')
pkl_w2v_embed = open('w2v_embeddings.pkl', 'rb')
w2v_embed = pickle.load(pkl_w2v_embed)
pkl_w2v_embed.close()
print('Reverse Dictionary Done')

ts_pl_km = tsnePltKmean()
twodembed, sel_ids = ts_pl_km.tsne_select_points(
        prime_items=w2v_embed,num_points = 1000)
ts_pl_km.plot(twodembed, sel_ids)
ts_pl_km.k_means(w2v_embed, n_clusters=5)
 # In[]:
    