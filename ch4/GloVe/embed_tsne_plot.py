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
import pickle

class embedTsnePlt(object):
    
    def __init__(self,                
                 distance_threshold = 0.25,
                 sample_threshold = 10
                 ):
        self.distance_threshold = distance_threshold
        self.sample_threshold = sample_threshold
        """Constructor.
        Args:
            istance_threshold: maximum distance between two points 
                              to qualify as neighbors
            sample_threshold: number of neighbors required to be 
                              considered a cluster. 
        """        
          
    
    def _find_clustered_embeddings(self, embeddings):
                       
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
        cosine_sim = np.dot(embeddings,np.transpose(embeddings))
        norm = np.dot(np.sum(embeddings**2,axis=1).reshape(-1,1),
             np.sum(np.transpose(embeddings)**2,axis=0).reshape(1,-1))
        # assert <test>, <message> 其中test是狀態測試，
        # message是斷言失敗時呈現的訊息。若無message則返回AssertionError
        assert cosine_sim.shape == norm.shape # 僅 assert <test>
        cosine_sim /= norm  # cosine_sim = cosine_sim/norm
         # make all the diagonal entries -1.0 
         # otherwise this will be picked as highest
        np.fill_diagonal(cosine_sim, -1.0) # 矩陣對角線填-1.0    
        argmax_cos_sim = np.argmax(cosine_sim, axis=1) # 返回axis軸最大值的索引
        mod_cos_sim = cosine_sim 
           
        ''' foor loop 用來把每一row最大值index設為-1，意味排除相似性  
            此用意應該是排除中心詞附近的相似性，盡量找text內的其他字的相似性
            
            註：雖然指令是設定mod_cos_sim(row, argmax index)=-1，
               但因為mod_cos_sim和cosine_sim是相同指摽，
               所以mod_cos_sim和cosine_sim矩陣值同時改'''
        for _ in range(self.sample_threshold-1):
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
        return np.where(max_cosine_sim>self.distance_threshold)[0]

    def tsne_prune(self, glove_embed_load, num_points = 1000):
        #The num_points is sample space which we use cosine similarity to prune.
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
        
        print('Fitting embeddings to T-SNE. This can take some time ...')
        # get the T-SNE manifold
        selected_embeddings = glove_embed_load[:num_points, :]
        two_d_embeddings = tsne.fit_transform(selected_embeddings)
        
        print('Pruning the T-SNE embeddings')
        # prune the embeddings by getting ones only more than 
        # n-many sample above the similarity threshold
        # this unclutters the visualization
        selected_ids = self._find_clustered_embeddings(
                selected_embeddings)
        two_d_embeddings = two_d_embeddings[selected_ids,:]        
        print('Out of ',num_points,' samples, ', selected_ids.shape[0],' samples were selected by pruning')
        
        return two_d_embeddings, selected_ids
    
    ### Plotting the t-SNE Results with Matplotlib
    def tsne_plot(self, two_d_embed, labels):
      
          n_clusters = 20 # number of clusters
          # automatically build a discrete set of colors, each for cluster
          cmap = cm.get_cmap("Spectral") 
          #label_colors = [pylab.cm.spectral(float(i) /n_clusters) for i in range(n_clusters)]
          label_colors = [cmap(float(i) /n_clusters) for i in range(n_clusters)]
          
          assert two_d_embed.shape[0] >= len(labels), 'More labels than embeddings'
          
          # Define K-Means
          kmeans = KMeans(n_clusters=n_clusters, init='k-means++',
                          random_state=0).fit(two_d_embed)
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

# In[2]:     
if __name__ == '__main__':
                
    # read python dict back from the file
    #  pkl file適合儲存dictionary
    reverse_dictionary = {}
    pkl_file = open('reverse_dict.pkl', 'rb')
    reverse_dictionary = pickle.load(pkl_file)
    pkl_file.close()   
        
    
    print('reverse_dictionary', reverse_dictionary)
    #print('reverse_dictionary shape', reverse_dictionary)
        
    glove_embed_load = np.load('glove_embeddings.npy')
    em_ts_pl = embedTsnePlt()
    twodembed, sel_ids = em_ts_pl.tsne_prune(glove_embed_load,
                                             num_points = 1000)
    words = [reverse_dictionary[i] for i in sel_ids]    
    em_ts_pl.tsne_plot(twodembed, words)
    