import tensorflow as tf
import pandas as pd
import numpy as np

class Data():
    def __init__(self):
        self.size = 0

    def get_xdata(self,x_data_path):
        df = pd.read_csv(x_data_path, sep=',', header=None)
        a = np.array(df).astype(int)
        self.size = len(df)
        a = a.reshape(self.size,28,28)
        self.x_data = a
        return self.x_data

    def get_ydata(self,y_data_path):
        df = pd.read_csv(y_data_path,sep = ',',header = None)
        b = np.array(df).astype(int)
        b = b.reshape(len(df),10)
        self.y_data = b
        return self.y_data

    def get_rand_batch(self,batch_size = None):
        if batch_size is None:
            b_size = 128
        else:
            b_size = batch_size

        rand_indices = np.random.choice(self.size, b_size, replace=False)
        x_batch = self.x_data[rand_indices]
        self.x_batch = x_batch.reshape(b_size, 28, 28, 1)
        self.y_batch = self.y_data[rand_indices]


gph = tf.Graph()
with gph.as_default():
    x = tf.placeholder('float',shape = [None,28,28,1],name = "x")
    y_true = tf.placeholder('float',shape = [None,10],name = "y_true")
    y_true_cls = tf.argmax(y_true,axis =1,name = "y_true_cls")

    ls_kern_count = [16,32,128,10]
    kern_size = [[5,5],[7,7]]

    kern_init = tf.random_normal_initializer(mean = 0.0,stddev = 0.01)
    bias_init = tf.zeros_initializer()

    conv1 = tf.layers.conv2d(x,filters = ls_kern_count[0],kernel_size=kern_size[0],
                             strides = [2,2],padding ="same",activation = tf.nn.relu,use_bias = False,
                             kernel_initializer=kern_init,trainable=True,name = "conv1")
    conv2 = tf.layers.conv2d(conv1, filters=ls_kern_count[1], kernel_size=kern_size[0],
                             strides=[2, 2], padding="same", activation=tf.nn.relu, use_bias=False,
                             kernel_initializer=kern_init, trainable=True,name = "conv2")

    flat_tensor = tf.layers.flatten(conv2,name = "flat_tensor")

    fc1 = tf.layers.dense(flat_tensor,units = ls_kern_count[2],activation = tf.nn.relu,
                          use_bias = True,kernel_initializer=kern_init,bias_initializer=bias_init,
                          trainable=True,name = "fc1")
    logits = tf.layers.dense(fc1,ls_kern_count[3],activation= None,use_bias = True,
                             kernel_initializer=kern_init,bias_initializer=bias_init,
                             trainable=True,name = "logits")

    y_pred = tf.nn.softmax(logits,name = "y_pred")
    y_pred_cls = tf.argmax(y_pred,axis = 1,name = "y_pred_cls")

    correct_pred = tf.equal(y_pred_cls,y_true_cls,name = "correct_pred")
    accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32),name = 'accuracy')

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = logits,labels=y_true),name = "loss")
    opt = tf.train.AdamOptimizer(1e-4).minimize(loss)

    saver = tf.train.Saver(max_to_keep=100)

save_dir = 'checkpoints/'
def get_save_path(net_numb):
    return save_dir+'network'+str(net_numb)

batch_size = 128
epochs = 10000

with tf.Session(graph=gph) as sess:
    sess.run(tf.global_variables_initializer())

    train_data = Data()
    train_data.get_xdata("data/x_train.csv")
    train_data.get_ydata("data/y_train.csv")

    for i in range(5):              # 5 denotes the number of networks in our ensemble

        print("\nNetwork: ",str(i))
        # initializing global variabale for each network
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):  #no. of epochs for each network in ensemble
            train_data.get_rand_batch(batch_size)
            x_batch = train_data.x_batch
            y_batch = train_data.y_batch
            feed_dict = {x:x_batch,y_true: y_batch}
            sess.run(opt,feed_dict)
            if epoch%100 == 0:
                acc = sess.run(accuracy,feed_dict)
                print("Iteration: ",str(epoch),"\tacc_on_train: ",acc)
        saver.save(sess = sess,save_path=get_save_path(i))
