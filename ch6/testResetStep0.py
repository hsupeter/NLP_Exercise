
# coding: utf-8

# In[1]:
import numpy as np
import tensorflow as tf
from tensorflow.python.ops import variable_scope as vs

# In[]:
'''with tf.variable_scope('zerostate', reuse = True):
     X= tf.Variable(3, dtype=tf.float32, name='X')
     Y= tf.Variable(4, dtype=tf.float32, name='Y')'''
X = tf.placeholder(dtype =tf.float32, name='X')  
Y = tf.placeholder(dtype =tf.float32, name='Y')     
f = X * Y
init = tf.global_variables_initializer()
# In[]:
with tf.Session() as sess:
    init.run()    
    print(f.eval(feed_dict={X:6, Y:7}))
    print(f.eval(feed_dict={X:9, Y:15}))
# In[2]:

'''
reset_predPhase_state_op = tf.assign(prev_predPhase_state,
        tf.truncated_normal([predPhase_batch_size, n_hidden],
        stddev=0.01,dtype=tf.float32))'''    

# 创建RNNCell的初始状态
#initial_state = tf.assign(rnn_cell.zero_state, tf.truncated_normal(
#    [batch_size, state_size], stddev=0.01, dtype=tf.float32))

batch_size_1 = 32 # batch大小
time_steps_1 = 10

batch_size_2 = 1 # batch大小
time_steps_2 = 1 # 隐藏状态ht维度

input_size = 100 # 输入向量xt维度
state_size = 128 # 隐藏状态ht维度
#batch_size = tf.placeholder(tf.int32)
#time_steps = tf.placeholder(tf.int32)
# 首先构造输入 shape为(batch_size, time_steps, input_size)
'''inputs_1 = tf.random_normal(shape=[batch_size_1, time_steps_1, input_size],
                          dtype=tf.float32)
inputs_2 = tf.random_normal(shape=[batch_size_2, time_steps_2, input_size],
                          dtype=tf.float32)'''
inputs_1 = np.random.rand(batch_size_1, time_steps_1, input_size)
inputs_2 = np.random.rand(batch_size_2, time_steps_2, input_size)
# inputs = tf.placeholder(dtype = tf.float32, shape = (batch_size, time_steps, input_size))
print('inputs_1.shape:', inputs_1.shape)

# inputs = tf.placeholder(dtype = tf.float32, shape = (batch_size, time_steps, input_size))
print('inputs_2.shape:',inputs_2.shape)
# 创建一个RNNCell对象
rnn_cell = tf.nn.rnn_cell.BasicRNNCell(num_units = state_size)
print('rnn_cell.state_size:', rnn_cell.state_size)

with tf.variable_scope('zerostate_scope', reuse = True):
    zero_state = rnn_cell.zero_state(batch_size_1, dtype = tf.float32)
    
reset_state_op1 = tf.assign(zero_state,
       tf.zeros([batch_size_1, state_size],
                dtype=tf.float32))
reset_state_op2 = tf.assign(zero_state,
      tf.truncated_normal([batch_size_2, state_size],
      stddev=0.01,dtype=tf.float32))

print('zero_state.shape:', zero_state.shape)
print('zero_state:\n', zero_state)

print('reset_state_op1.shape:', reset_state_op1.shape)
print('reset_state_op1:\n', reset_state_op1)

print('reset_state_op2.shape:', reset_state_op2.shape)
print('reset_state_op2:\n', reset_state_op2)
# 创建dynamic_rnn
inputs = tf.placeholder(tf.float32, [None, None, input_size]) 
outputs, state = tf.nn.dynamic_rnn(rnn_cell, inputs,
                           initial_state = zero_state)

print('inputs:', inputs)
print('outputs:', outputs)
print('state:', state)
# In[]:
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    #zero_state = zero_state_1
    for _ in range(2):        
        print('\n\n')
        output_1, final_state_1 = sess.run(
            [outputs,state], feed_dict={inputs:inputs_1})    
        print('final_state_1.shape:',final_state_1.shape)
        print('final_state_1[0]:',final_state_1[0])
        print('output_1.shape:', output_1.shape)
        print('output_1:', output_1)
        sess.run(reset_state_op1)        
          
    for _ in range(2):
        print('\n\n')
        output_2, final_state_2 = sess.run(
                [outputs, state], feed_dict={inputs:inputs_2})
    #output_2, final_state_2 = sess.run([outputs, state],
    #   feed_dict={inputs:inputs_2})
        print('\n')
        print('final_state_1.shape:',final_state_1.shape)
        print('final_state_1[0]:',final_state_1[0])
        print('output_1.shape:', output_1.shape)
        print('output_1:', output_1)
        sess.run(reset_state_op1)
        
# In[]:
[v.name for v in tf.trainable_variables()]
# In[]:
[v.name for v in tf.global_variables()] 
for op in tf.get_default_graph().get_operations():
    print(op)
# In[]:
'''    
output_size = 10
batch_size = 32
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=output_size)
print(cell.output_size)
input = tf.placeholder(dtype=tf.float32,shape=[batch_size,150])
h0 = cell.zero_state(batch_size=batch_size,dtype=tf.float32)
output,h1 = cell.call(input,h0)
print(output)
print(h1)
# In[]:
output_size = 4
batch_size = 3
dim = 5
cell = tf.nn.rnn_cell.BasicRNNCell(num_units=output_size)
input = tf.placeholder(dtype=tf.float32, shape=[batch_size, dim])
h0 = cell.zero_state(batch_size=batch_size, dtype=tf.float32)
output, h1 = cell.call(input, h0)

x = np.array([[1, 2, 1, 1, 1], [2, 0, 0, 1, 1], [2, 1, 0, 1, 0]])


scope = vs.get_variable_scope()
with vs.variable_scope(scope, reuse=True) as outer_scope:
    weights = vs.get_variable(
        "kernel", [9, output_size],
        dtype=tf.float32,
        initializer= None)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    a,b,w= sess.run([output,h1,weights],feed_dict={input:x})
    print('output:')
    print(a)
    print('h1:')
    print(b)
    print("weights:")
    print(w)# shape = (9,4)

state = np.zeros(shape=(3,4))# shape = (3,4)
all_input = np.concatenate((x,state),axis=1)# shape = (3,9)
result = np.tanh(np.matmul(all_input,w))
print('result:')
print(result)'''   

