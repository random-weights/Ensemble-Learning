import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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


# def main():
#     train_data = Data()
#     train_data.get_xdata("data/x_train.csv")
#     train_data.get_ydata("data/y_train.csv")
#     train_data.get_rand_batch(1)
#     x_batch = train_data.x_batch.reshape(28,28)
#     y_batch = train_data.y_batch
#     print(y_batch)
#     plt.imshow(x_batch,cmap = 'binary')
#     plt.show()
# main()

gph = tf.Graph()
with gph.as_default():
    x = tf.placeholder('float',shape = [None,28,28,1],name = "x")
    y_true = tf.placeholder('float',shape = [None,10],name = "y_true")
    y_true_cls = tf.argmax(y_true,axis =1,name = "y_true_cls")

    kern1 = tf.Variable(tf.random_normal(shape = [5,5,1,16],mean = 0.0,stddev=0.01),name = "kern1")
    kern2 = tf.Variable(tf.random_normal(shape = [7,7,16,32],mean = 0.0,stddev=0.01),name = "kern2")

    conv1 = tf.nn.conv2d(x,kern1,[1,2,2,1],'SAME',name = "conv1")
    conv2 = tf.nn.conv2d(conv1,kern2,[1,2,2,1],'SAME',name = "conv2")
    #shape = [1,7,7,32]

    flat_tensor = tf.reshape(conv2,[-1,1568])

    w1 = tf.Variable(tf.random_normal(shape = [1568,128],mean = 0.0,stddev=0.01),name = "w1")
    w2 = tf.Variable(tf.random_normal(shape = [128,10],mean = 0.0,stddev=0.01),name = "w2")
    b1 = tf.Variable(tf.constant(0.0,shape = [128]),name = "b1")
    b2 = tf.Variable(tf.constant(0.0,shape = [10]),name = "b2")

    fc1 = tf.nn.relu(tf.matmul(flat_tensor,w1)+b1,name = "fc1")
    logits = tf.matmul(fc1,w2) + b2

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

with tf.Session(graph=gph) as sess:
    sess.run(tf.global_variables_initializer())

    train_data = Data()
    train_data.get_xdata("data/x_train.csv")
    train_data.get_ydata("data/y_train.csv")

    for i in range(5):              # 5 denotes the number of networks in our ensemble

        print("\nNetwork: ",str(i))
        # initializing global varibale for each network
        sess.run(tf.global_variables_initializer())

        for epoch in range(10000):  #no. of epochs for each ensemble
            train_data.get_rand_batch(batch_size)
            x_batch = train_data.x_batch
            y_batch = train_data.y_batch
            feed_dict = {x:x_batch,y_true: y_batch}
            sess.run(opt,feed_dict)
            if epoch%100 == 0:
                acc = sess.run(accuracy,feed_dict)
                print("Iteration: ",str(epoch),"\tacc_on_train: ",acc)
        saver.save(sess = sess,save_path=get_save_path(i))