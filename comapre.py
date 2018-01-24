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

def wrong_indices(pred_cls,true_cls):
    comp = pred_cls - true_cls
    indices = []
    for i in range(len(comp)):
        if comp[i]!=0:
            indices.append(i)
    return indices


ls_acc = []             # list of prediction accuracies
ls_preds = []           #list of predictions as softmax output
ls_preds_cls = []       #list of predictions as a class number
sess = tf.Session()

#test data
test_data = Data()
test_data.get_xdata("data/x_test.csv")
test_data.get_ydata("data/y_test.csv")
x_test = test_data.x_data.reshape(10000,28,28,1)
y_test = test_data.y_data
y_test_cls = np.argmax(y_test)

for i in range(5):      #iterating over 5 networks
    saver = tf.train.import_meta_graph(get_save_path(i)+'.meta')
    saver.restore(sess =sess, save_path = get_save_path(i))

    graph = tf.get_default_graph()
    x = graph.get_tensor_by_name("x:0")
    y_true = graph.get_tensor_by_name("y_true:0")

    feed_dict = {x: x_test,y_true:y_test}
    acc, temp_pred = sess.run(['accuracy:0','y_pred:0'],feed_dict)
    #extracted accuracy and softmax predictions of each network

    #appending to list of predictions of 5 networks on test set
    ls_preds.append(list(temp_pred))
    temp_pred_cls = np.argmax(temp_pred,axis = 1)
    ls_preds_cls.append(temp_pred_cls)

    #list of accuracies of 5 networks
    ls_acc.append(acc)


sess.close()
#finding the best performing network
max_acc = max(ls_acc)
network_numb = ls_acc.index(max_acc)
print("Best peroforming network: ",network_numb)

# compute mean of predictions of 5 networks and do argmax for y_pred from ensemble
ls_preds = np.array(ls_preds)
sum_preds = ls_preds[0]
for i in range(1,5):
    sum_preds += np.add(sum_preds,ls_preds[i])

mean_preds = 0.2 * sum_preds
mean_preds_class = np.argmax(mean_preds,axis = 1)

network_pred = ls_preds[network_numb]
network_pred_cls = np.argmax(network_pred,axis  = 1)

network_diff_indices = wrong_indices(network_pred_cls,y_test_cls)
ensemble_diff_indices = wrong_indices(mean_preds_class,y_test_cls)

print("Wrong indices on best network: ",len(network_diff_indices))
print("Wrong indices on ensemble: ",len(ensemble_diff_indices))