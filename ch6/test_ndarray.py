
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf


# In[4]:


a1= np.array([[0., 1., 2., 3., 4.],  [5., 6., 7., 8., 9.]])
a2= np.array([[10.,12.,14.,16.,18.], [20.,22.,24.,26.,28.]])
a3= np.array([[ 6., 8.,10.,12.,14.], [16.,18.,20.,22.,24.]])
a = [a1,  a2,  a3]
print('a1:\n', a1)
print('a2:\n', a2)
print('a3:\n', a3)
print('a:\n', a)


# In[6]:


b1= np.array([[50.,52.,54.,56.,58.], [60., 62., 64., 66., 68.]])
b2= np.array([[30.,32.,34.,36.,38.], [40., 42., 44., 46., 48.]])
b3= np.array([[12.,14.,16.,18.,20.], [22., 24., 26., 28., 30.]])
b = [b1,  b2,  b3]
print('b1:\n', b1)
print('b2:\n', b2)
print('b3:\n', b3)
print('b:\n', b)


# In[7]:


a_array_325= np.array(a)
b_array_325= np.array(b)
print('a_array_325:\n', a_array_325)
print('b_array_325:\n', b_array_325)


# In[8]:


a_array_235 = np.reshape(np.hstack(a_array_325), (2, 3, 5))
b_array_235 = np.reshape(np.hstack(b_array_325), (2, 3, 5))
print('a_array_235:\n', a_array_235)
print('b_array_235:\n', b_array_235)


# In[11]:


a235_m_b235 = (a_array_235 * b_array_235)
print('a235_m_b235:', a235_m_b235.shape,'\n', a235_m_b235)


# In[13]:


ab235_avg = tf.reduce_sum(a235_m_b235)/(2*3)
sess = tf.Session()
ab235_avg_v = sess.run(ab235_avg)
print('ab235_avg:\n', ab235_avg_v)
sess.close()


# In[15]:


rev_a_325 = np.reshape(np.hstack(a_array_235), (3, 2, 5))
rev_b_325 = np.reshape(np.hstack(b_array_235), (3, 2, 5))
print('rev_a_325 ',rev_a_325.shape, '\n', rev_a_325)                       
print('rev_b_325 ',rev_b_325.shape, '\n', rev_b_325)


# In[17]:


print('a_array_325 ',a_array_325.shape, '\n', a_array_325)
print('b_array_325 ',b_array_325.shape, '\n', b_array_325)


# In[18]:


a325_m_b325_rev = rev_a_325 * rev_b_325
print('a325_m_b325_rev:',a325_m_b325_rev.shape, '\n', a325_m_b325_rev)


# In[20]:


print('a235_m_b235:',a235_m_b235.shape, '\n', a235_m_b235)


# In[40]:


ab325_avg = tf.reduce_sum(a325_m_b325_rev)/(2*3)
sess = tf.Session()
ab325_avg_v = sess.run(ab325_avg)
print('ab325_avg:\n', ab325_avg_v)
sess.close()


# In[23]:


a25_list = []
b25_list = []
for ui in range(3):
    aui = np.array(rev_a_325[ui])
    a25_list.append(aui)
    bui = np.array(rev_b_325[ui])
    b25_list.append(bui)
print('a25_list:\n', a25_list)
print('b25_list:\n', b25_list)


# In[24]:


print('a25_list:\n', a25_list)
print()
print('a:\n', a)


# In[25]:


print('b25_list:\n', b25_list)
print()
print('b:\n', b)


# In[41]:


concat_a25_list = tf.concat(a25_list, 0)
concat_b25_list = tf.concat(b25_list, 0)
a25_m_b25_avg =tf.reduce_sum(concat_a25_list * concat_b25_list)/(3*2) 
sess = tf. Session()
cona25, conab25, a25mb25avg =sess.run([concat_a25_list,                               concat_b25_list,a25_m_b25_avg])
print('concat_a25_list:\n', cona25)
print('concat_b25_list:\n', conab25)
print('a25_m_b25_avg: ', a25mb25avg)
sess.close()


# In[30]:


a325_hst = np.hstack(a_array_325)
a235 = np.reshape(a325_hst, (2, 3, 5))
print('a_array_325:', a_array_325.shape, '\n', a_array_325)
print()
print('a325_hst:', a325_hst.shape, '\n', a325_hst)
print()
print('a235:', a235.shape, '\n', a235)


# In[32]:


a235_hst = np.hstack(a235)
a325 = np.reshape(a235_hst, (3, 2, 5))
print('a235_hst:', a325_hst.shape, '\n', a235_hst)
print()
print('a325:', a325.shape, '\n', a325)
print()
print('a_array_325:', a_array_325.shape, '\n', a_array_325)


# In[34]:


print('a325 shape:', a325.shape)
print('a325 size axis 0:', np.size(a325 ,0))
print('a325 size axis 1:', np.size(a325 ,1))
print('a325 size axis 2:', np.size(a325 ,2))


# In[46]:


# a25_m_b25_avg =tf.reduce_sum(concat_a25_list * concat_b25_list)/(3*2)
n_roll = len(a25_list)
batch_size = np.size(a25_list[0], 0)
axis_1 = np.size(a25_list[0], 1)
print('n_roll:', n_roll)
print('batch_size:', batch_size)
print('axis_1:', axis_1)


# In[45]:


a25mb25avg=tf.reduce_sum(concat_a25_list*concat_b25_list)/(n_roll*batch_size)
with tf.Session() as sess:
    print('a25mb25avg:', a25mb25avg.eval())


# In[47]:


a25_list_1= [np.array(rev_a_325[ui]) for ui in range(n_roll)]
b25_list_1= [np.array(rev_b_325[ui]) for ui in range(n_roll)]
print('a25_list_1:\n', a25_list_1)
print('b25_list_1:\n', b25_list_1)

print('a25_list:\n', a25_list)
print('b25_list:\n', b25_list)


# In[77]:


x1 = tf.constant([[0., 1., 2., 3., 4.], [5., 6., 7., 8., 9.]])
x2 = tf.constant([[10., 12., 14., 16., 18.], [20., 22., 24., 26., 28.]])
x3 = tf.constant([[ 6.,  8., 10., 12., 14.], [16., 18., 20., 22., 24.]])                 
y1 = tf.constant([[50., 52., 54., 56., 58.], [60., 62., 64., 66., 68.]])
y2 = tf.constant([[30., 32., 34., 36., 38.], [40., 42., 44., 46., 48.]])
y3 = tf.constant([[12., 14., 16., 18., 20.], [22., 24., 26., 28., 30.]])
x = tf.stack([x1, x2, x3], axis=0)
x_list = [x1, x2, x3]
y = tf.stack([y1, y2, y3], axis=0)
y_list = [y1, y2, y3]
x_r325 = tf.reshape(x, [2, 3, 5])

#x_hs = tf.concat(x, 2)
#x_hs = tf.stack(tf.concat(x, 1), axis=1)
x_hs = tf.reshape(x, [-1, 5])
x_hs_235 = tf.reshape(x_hs, [2, 3, 5])


# In[78]:


with tf.Session() as sess:
    x_v, x_list_v, y_v, y_list_v, x_r325_v,x_hs_v,x_hs_235_v = sess.run(
        [x, x_list, y, y_list, x_r325, x_hs, x_hs_235])
    print('x shape:', x_v.shape, 'x:\n', x_v)
    #print('x list:\n', x_list_v)
    #print('y shape:', y_v.shape, 'y:\n', y_v)
    #print('y list:\n', y_list_v)
    print('x_r325:\n', x_r325_v)
    print('x_hs:\n', x_hs_v)
    print('x_hs_235:\n', x_hs_235_v)


# In[107]:


x_us1_list = tf.unstack(x, axis=1)
x_us1_array = tf.stack(x_us1, axis=0)
x_us1_soft_list = [tf.nn.softmax(x_us1_list[ui]) for ui in range(2)]


# In[108]:


with tf.Session() as sess:
    x_us1_list_v, x_us1_array_v, x_us1_soft_list_v = sess.run(
        [x_us1_list, x_us1_array, x_us1_soft_list])
        
    #print('x_us0:\n',x_us0_v)
    print('x_us1_list type:',type(x_us1_list_v),
          'x_us1_list[0] type:',x_us1_list_v[0].shape,'x_us1:\n',
          x_us1_list_v)
    print('x_us1_array type:',type(x_us1_array_v),
          'x_us1s shape:', x_us1_array_v.shape,'\n',x_us1_array_v)
    print('x_us1_soft_list type:',type(x_us1_soft_list_v),
          'x_us1_soft_list[0] shape:', x_us1_soft_list_v[0].shape,'\n',
          x_us1_soft_list_v)
    #print('x_us2:\n',x_us2_v)


# In[112]:


a1= np.array([[0., 1., 2., 3., 4.],  [5., 6., 7., 8., 9.]])
text_digt =[5, 7, 4, 8, 0]
text_digt[np.random.randint(0,5)]


# In[118]:


#[np.random.randint(0, 100)]
predPhase_word = np.zeros((1, 1, 10),dtype=np.float32) 
predPhase_word[0, 0, text_digt[np.random.randint(0, 5)]] = 1.0
print(predPhase_word)


# In[127]:




