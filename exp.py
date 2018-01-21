import tensorflow as tf
import numpy as np
import pandas as pd


save_dir = 'checkpoints/'

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

def get_save_path(net_number):
    return save_dir + 'network' + str(net_number)

ac = []
preds = []
sess = tf.Session()
for i in range(5):
    saver = tf.train.import_meta_graph(get_save_path(i)+'.meta')
    saver.restore(sess =sess, save_path = get_save_path(i))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")

    test_data = Data()
    test_data.get_xdata("data/x_test.csv")
    test_data.get_ydata("data/y_test.csv")
    x_test = test_data.x_data.reshape(10000,28,28,1)
    y_test = test_data.y_data

    feed_dict = {x: x_test,y_true:y_test}
    temp_pred = sess.run('y_pred_cls:0',feed_dict)

    #list of predictions of 5 networks on test set
    preds.append(list(temp_pred))


sess.close()


