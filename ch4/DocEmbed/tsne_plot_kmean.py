# -*- coding: utf-8 -*-
"""
Created on Wed Mar  4 16:05:38 2020

@author: Peter Hsu
"""

# In[1]:
from __future__ import division, print_function, unicode_literals
import numpy as np
from matplotlib import pylab
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import matplotlib.cm as cm

class tsnePltKmean(object):   
    
    def tsne_select_points(self, prime_items, num_points=1000):
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)    
        print('Fitting items to T-SNE (Non_Prun)')
        selected_ids, doc_embeddings = zip(*prime_items.items())
        two_d_items = tsne.fit_transform(doc_embeddings)
        print('\tDone')
        return two_d_items, selected_ids    
   
    ### Plotting the t-SNE Results with Matplotlib
    def plot(self, embeddings, labels):        
          n_clusters = 5 # number of clusters
            
          # automatically build a discrete set of colors, each for cluster
          cmap = cm.get_cmap("Spectral")
          label_colors = [cmap(float(i) /n_clusters) for i in range(n_clusters)]
          label_markers = ['o','^','d','s','x']
          # make sure the number of document embeddings is same as
          # point labels provided
          assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
          
          pylab.figure(figsize=(15,15))  # in inches
        
          def get_label_id_from_key(key):
            '''
            We assign each different category a cluster_id
            This is assigned based on what is contained in the point label
            Not the actual clustering results
            '''
            if 'business' in key:
                return 0
            elif 'entertainment' in key:
                return 1
            elif 'politics' in key:
                return 2
            elif 'sport' in key:
                return 3
            elif 'tech' in key:
                return 4
                    
          # Plot all the document embeddings and their corresponding words
          for i, label in enumerate(labels):
            x, y = embeddings[i,:] #傳入經TSNE後的2D embedding值作為座標 x, y值
            print('i:', i,'\tlabel:', label)
            print('embeddings[i,:]:', x,'\t', y)
            pylab.scatter(x, y, c=label_colors[get_label_id_from_key(label)],
                s=50, marker=label_markers[get_label_id_from_key(label)])    
            
            # Annotate each point on the scatter plot
            pylab.annotate(label, xy=(x, y), xytext=(5, 2),
                       textcoords='offset points',
                       ha='right', va='bottom',fontsize=16)
          
          # Set plot title
          pylab.title('Document Embeddings visualized with t-SNE',fontsize=24)
          
          # Use for saving the figure if needed
          pylab.savefig('document_embeddings.png')
          pylab.show()
          
    def k_means(self, document_embeddings, n_clusters):
        # Create and fit K-means
        kmeans = KMeans(n_clusters=n_clusters, random_state=43643,
                max_iter=10000, n_init=100, algorithm='elkan')
        kmeans.fit(np.array(list(document_embeddings.values())))
        
        # Compute items fallen within each cluster
        '''kmeans.labels_是cluste完的label值，此處為0~4，再把
         document_embeddings的key值（內含test文件代號，e.g. tech-56）
         與此label值zip起來，再依label值把此key值分類集合
         (document_classes[lbl] = [inp])並顯示'''
        document_classes = {}
        for inp, lbl in zip(list(document_embeddings.keys()),
                            kmeans.labels_):            
            if lbl not in document_classes:
                document_classes[lbl] = [inp] #要用[inp]形成list才能用append               
            else:
                document_classes[lbl].append(inp)                
        for k,v in document_classes.items():    
            print('\nDocuments in Cluster ',k)
            print('\t',v)
                

# In[2]:
'''
if __name__ == '__main__':
                
    # read python dict back from the file
    #  pkl file適合儲存dictionary
    reverse_dictionary = {}
    pkl_file = open('reverse_dict.pkl', 'rb')
    reverse_dictionary = pickle.load(pkl_file)
    pkl_file.close()   
        
    
    print('reverse_dictionary', reverse_dictionary)
    #print('reverse_dictionary shape', reverse_dictionary)
        
    prime_items = np.load('glove_embeddings.npy')
    em_ts_pl = embedTsnePlt()
    twodembed, sel_ids = em_ts_pl.tsne_select_points(prime_items,
                                             num_points = 1000)
    words = [reverse_dictionary[i] for i in sel_ids]    
    em_ts_pl.tsne_plot(twodembed, words)
'''    