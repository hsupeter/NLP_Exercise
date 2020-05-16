
# coding: utf-8

# In[1]:
from __future__ import division, print_function, unicode_literals
import tensorflow as tf
from matplotlib import pylab
import numpy as np
# Required for Data downaload and preparation
import struct
import gzip
import os
from six.moves.urllib.request import urlretrieve

# In[2]:
def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X)//batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches): #把rnd_idx分割成n_batches個陣列，每個為batch_size(50)大小
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch
# In[3]:
# to make this notebook's output stable across runs
def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

# To plot pretty figures
import matplotlib as mpl
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

# Where to save the figures
PROJECT_ROOT_DIR = "."
#CHAPTER_ID = "cnn"
channels = 1  
# Number of different digits we have images for (i.e. classes)
n_classes = 10 
n_train = 55000 # Train dataset size
n_valid = 5000 # Validation dataset size
n_test = 10000 # Test dataset size

height = 28
width = 28

def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images",
                        fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)

def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")

# ## Visualizing MNIST Results
# Here we define a function to collect correctly and  
# incorrectly classified samples to visualize later.  
# Visualizing such samples will help us to understand why  
# the CNN incorrectly classified certain samples.
# Makes sure we only collect 10 samples for each 
correct_fill_index, incorrect_fill_index = 0,0
# Visualization purposes
correctly_pred_img = np.empty(shape=(10,28,28,1),
                               dtype=np.float32)
correct_predictions = np.empty(shape=(10,n_classes),
                               dtype=np.float32)
incorrectly_pred_img = np.empty(shape=(10,28,28,1),
                                 dtype=np.float32)
incorrect_predictions = np.empty(shape=(10,n_classes),
                                 dtype=np.float32)

def collect_samples(test_batch_predictions,test_images,
                    test_labels):
    global correctly_pred_img, correct_predictions
    global incorrectly_pred_img, incorrect_predictions
    global correct_fill_index, incorrect_fill_index
    
    correct_indices = np.where(np.argmax(
                        test_batch_predictions,
                        axis=1)==test_labels)[0]
                    
    
    incorrect_indices = np.where(np.argmax(
                          test_batch_predictions,
                          axis=1)!=test_labels)[0]
    
    if correct_indices.size>0 and correct_fill_index<10:
        #print('\nCollecting Correctly Predicted Samples')
        chosen_index = np.random.choice(correct_indices)
        correctly_pred_img[correct_fill_index,:,:,:] = \
                    test_images[chosen_index,:].reshape(1,
                    height,width, channels)
        correct_predictions[correct_fill_index,:] = \
                    test_batch_predictions[chosen_index,:]
        correct_fill_index += 1

    if incorrect_indices.size>0 and incorrect_fill_index < 10:
        #print('Collecting InCorrectly Predicted Samples')
        chosen_index = np.random.choice(incorrect_indices)
        incorrectly_pred_img[incorrect_fill_index,:,:,:] = \
                    test_images[chosen_index,:].reshape(1, 
                    height, width, channels)
        incorrect_predictions[incorrect_fill_index,:]= \
            test_batch_predictions[chosen_index,:]
        incorrect_fill_index += 1
   
# In[4]:
# ## Lolading Data
# 
# Here we download (if needed) the MNIST dataset and, perform reshaping and normalization. Also we conver the labels to one hot encoded vectors.

# batch_size = 100 # This is the typical batch size we've been using
# image_size = 28 # This is the width/height of a single image
 
# Number of color channels in an image. These are black and white images 

 
def maybe_download(url, filename, expected_bytes, force=False):
   """Download a file if not present, and make sure it's the right size."""
   if force or not os.path.exists(filename):
     print('Attempting to download:', filename) 
     filename, _ = urlretrieve(url + filename, filename)
     print('\nDownload Complete!')
   statinfo = os.stat(filename)
   if statinfo.st_size == expected_bytes:
     print('Found and verified', filename)
   else:
     raise Exception(
       'Failed to verify ' + filename +\
         '. Can you get to it with a browser?')
   return filename
 
 
def read_mnist(fname_img, fname_lbl, one_hot=True):
     print('\nReading files %s and %s'%(fname_img, fname_lbl))
     
     # Processing images
     with gzip.open(fname_img) as fimg:        
         magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
         print(num,rows,cols)
         img = (np.frombuffer(fimg.read(num*rows*cols),\
                  dtype=np.uint8).reshape(num, rows, cols,1)).\
                  astype(np.float32)
         print('(Images) Returned a tensor of shape ',img.shape)
         
         img = (img - np.mean(img)) /np.std(img)
         #img *= 1.0 / 255.0
     
     # Processing labels
     with gzip.open(fname_lbl) as flbl:
         # flbl.read(8) reads upto 8 bytes
         magic, num = struct.unpack(">II", flbl.read(8))               
         lbl = np.frombuffer(flbl.read(num), dtype=np.int8)
         if one_hot:
             one_hot_lbl = np.zeros(shape=(num,10),dtype=np.float32)
             one_hot_lbl[np.arange(num),lbl] = 1.0
         print('(Labels) Returned a tensor of shape: %s'%lbl.shape)
         print('Sample labels: ',lbl[:10])
     
     if  one_hot:
        return img, one_hot_lbl
     else:
        return img, lbl    
    


train_inputs, train_labels = read_mnist(\
                            'train-images-idx3-ubyte.gz',\
                            'train-labels-idx1-ubyte.gz',one_hot= False)
X_train, y_train = train_inputs[:n_train,:,:,:], train_labels[:n_train]
X_valid, y_valid = train_inputs[n_train:,:,:,:], train_labels[n_train:]
print('Train Inputs Shape: ', X_train.shape)
print('Train labels Shape: ', y_train.shape)
print('Valid Inputs Shape: ', X_valid.shape)
print('Valid labels Shape: ', y_valid.shape)
X_test, y_test = read_mnist('t10k-images-idx3-ubyte.gz',\
                            't10k-labels-idx1-ubyte.gz',one_hot=False)
print()
print('Test Inputs Shape: ', X_test.shape)
print('Test labels Shape: ', y_test.shape)

# In[5]:
n_inputs = height * width

conv1_fmaps = 32 # 輸出的feature map數
conv1_ksize = 3  # kernal_size為兩整數的tuple表捲積窗的高和寬
                 # 若為整數則高和寬相同
conv1_stride = 1
conv1_pad = "SAME"

conv2_fmaps = 64
conv2_ksize = 3
conv2_stride = 2
conv2_pad = "SAME"

pool3_fmaps = conv2_fmaps

n_fc1 = 64
n_outputs = 10

reset_graph()

with tf.name_scope("inputs"):
    X = tf.placeholder(tf.float32, shape=[None, height,
                          width, channels], name="X")    
    y = tf.placeholder(tf.int32, shape=[None], name="y")
    conv1 = tf.layers.conv2d(X, filters=conv1_fmaps,
                             kernel_size=conv1_ksize,
                             strides=conv1_stride,
                             padding=conv1_pad,
                             activation=tf.nn.relu, name="conv1")
    conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps,
                             kernel_size=conv2_ksize,
                             strides=conv2_stride,
                             padding=conv2_pad,
                             activation=tf.nn.relu, name="conv2")

with tf.name_scope("pool3"):
    pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
    pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7])
    '''此處7 * 7是與圖尺寸28*28的1/4大小
    pool3_fmaps * 7 * 7 = 64*7*7 = 3136 = 4*28*28'''

with tf.name_scope("fc1"):
    fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")

with tf.name_scope("output"):
    logits = tf.layers.dense(fc1, n_outputs, name="output")
    Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

# In[6]:
n_epochs = 2
batch_size = 100

max_checks_without_progress = 5
checks_without_progress = 0
best_loss = np.infty
config = tf.ConfigProto(allow_soft_placement=True)
# Good practice to use this to avoid any surprising errors thrown by TensorFlow
config.gpu_options.allow_growth = True
# Making sure Tensorflow doesn't overflow the GPU
 
config.gpu_options.per_process_gpu_memory_fraction = 0.9 
#session = tf.InteractiveSession(config=config)
with tf.Session(config=config) as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            
        print('\nepoch ',epoch)
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        
        loss_val,acc_val, i = 0, 0, 0
        for X_val_batch, y_val_batch in shuffle_batch(
                        X_valid, y_valid, batch_size):
            los_bat, acc_bat = sess.run([loss, accuracy],
                        feed_dict={X: X_val_batch, y:  y_val_batch})        
            loss_val+= los_bat
            acc_val += acc_bat
            i += 1
        loss_val, acc_val = loss_val/i, acc_val / i        
        print('Last batch accuracy:{:.3f} \nValid accuracy: {:.2f}%'
              .format(acc_batch, acc_val* 100))              
        print('Valid Loss: %.3f' %loss_val) 
        if loss_val < best_loss:
            save_path = saver.save(sess, "./mnist_model.ckpt")
            best_loss = loss_val
            checks_without_progress = 0
        else:
            checks_without_progress += 1
            if checks_without_progress > max_checks_without_progress:
                print("Early stopping!")
                break
        print("Best loss:{:.3f}".format(best_loss))
        
    save_path = saver.save(sess, "./mnist_model")
        
with tf.Session() as sess:
    saver.restore(sess, "./mnist_model.ckpt")
    acc_test, i = 0, 0
    for X_tst_batch, y_tst_batch in shuffle_batch(
                    X_test, y_test, batch_size):
        acc_bat = accuracy.eval(feed_dict={
                    X: X_tst_batch, y: y_tst_batch})
        test_batch_pred_prob = Y_proba.eval(
                feed_dict={X: X_tst_batch})
        test_batch_predict=np.argmax(test_batch_pred_prob,axis=1)
        #print('\ni:', i+1)
        #print('acc_bat.shape:',acc_bat.shape)
        #print('tst_bat_pred.shape:',test_batch_pred.shape)
        #print('bat_pred.shape:',test_batch_predict.shape)        
        acc_test += acc_bat
        i += 1
        collect_samples(test_batch_pred_prob, X_tst_batch, y_tst_batch)
    #print('tst_bat_pred:',test_batch_pred)
    print('Test Label:  \n',y_tst_batch[:20])
    print('Test Predict:\n',test_batch_predict[:20])
    acc_test = acc_test / i
    print("\nFinal Test accuracy: {:.2f}%".format(acc_test * 100))
    #image_test = X_tst_batch[20]

# In[7]:
digit_images = []
for image_idx in range (20):  
    print('\n',image_idx,' Test Label[10]: ',
          y_tst_batch[image_idx])
    print('Test Predict:',test_batch_predict[image_idx])
    digit_image = X_tst_batch[image_idx].reshape(28, 28)    
    digit_images.append(digit_image)    
    plt.imshow(digit_image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")    
    #save_fig("test_digit_plot_%d" %image_idx)
    plt.show()

# In[8]:
print('Test Label:  \n',y_tst_batch[:20])
print('Test Predict:\n',test_batch_predict[:20])
plt.figure(figsize=(9,9))
plot_digits(digit_images, images_per_row=10)
save_fig("more_digits_plot")
plt.show()

# In[9]:
# Defining the plot related settings
pylab.figure(figsize=(25,20))  # in inches
pl_width=0.5 # Width of a bar in the barchart
pl_padding = 0.05 # Padding between two bars
pl_labels = list(range(0,10)) # Class labels

# Defining X axis
x_axis = np.arange(0,10)

# We create 4 rows and 7 column set of subplots

# We choose these to put the titles in
# First row middle
pylab.subplot(4, 7, 4)
pylab.title('Correctly Classified Samples',fontsize=24)

# Second row middle
pylab.subplot(4, 7,11)
pylab.title('Softmax Predictions for Correctly Classified Samples',\
            fontsize=24)

# For 7 steps
for sub_i in range(7):
    # Draw the top row (digit images)
    pylab.subplot(4, 7, sub_i + 1)        
    pylab.imshow(np.squeeze(correctly_pred_img
            [sub_i].reshape(28, 28)),cmap='gray')
                     
    pylab.axis('off')
    
    # Draw the second row (prediction bar chart)
    pylab.subplot(4, 7, 7 + sub_i + 1)        
    pylab.bar(x_axis + pl_padding, correct_predictions[sub_i], pl_width)
    pylab.ylim([0.0,1.0])    
    pylab.xticks(x_axis, pl_labels)

# Set titles for the third and fourth rows
pylab.subplot(4, 7, 18)
pylab.title('Incorrectly Classified Samples',fontsize=26)
pylab.subplot(4, 7,25)
pylab.title('Softmax Predictions for Incorrectly Classified Samples',\
            fontsize=24)

# For 7 steps
for sub_i in range(7):
    # Draw the third row (incorrectly classified digit images)
    pylab.subplot(4, 7, 14 + sub_i + 1)
    pylab.imshow(np.squeeze(incorrectly_pred_img
                [sub_i].reshape(28, 28)),cmap='gray')                 
    pylab.axis('off')
    
    # Draw the fourth row (incorrect predictions bar chart)
    pylab.subplot(4, 7, 21 + sub_i + 1)        
    pylab.bar(x_axis + pl_padding, incorrect_predictions[
            sub_i], pl_width)
    pylab.ylim([0.0,1.0])
    pylab.xticks(x_axis, pl_labels)

# Save the figure
pylab.savefig('mnist_result.png')
pylab.show()
