

```python
%matplotlib inline
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
```

### Helper function for retrieving data from csv




```python
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
```

### Helper function to plot images and  show the true label, predicted,ensembled label on the bottom
adapted from Hvass laboratories


```python
def plot_images(images,y_true_cls,y_pred_cls = None,y_ensemb_cls = None):
    
    # Create figure with 3x3 sub-plots.
    fig, axes = plt.subplots(3, 3)

    # Adjust vertical spacing if we need to print ensemble and best-net.
    if y_ensemb_cls is None:
        hspace = 1
    else:
        hspace = 1.0
    fig.subplots_adjust(hspace=hspace, wspace=0.3)

    # For each of the sub-plots.
    for i, ax in enumerate(axes.flat):

        # There may not be enough images for all sub-plots.
        if i < len(images):
            # Plot image.
            ax.imshow(images[i].reshape(28,28), cmap='binary')

            # Show true and predicted classes.
            if (y_ensemb_cls is None):
                if (y_pred_cls is None):
                    xlabel = "True: {0}".format(y_true_cls[i])
                else:
                    msg = "True: {0} \nPredicted: {1}"
                    xlabel = msg.format(y_true_cls[i],y_pred_cls[i])
            else:
                msg = "True: {0}\nNetwork: {1}\nEnsemble: {2}"
                xlabel = msg.format(y_true_cls[i],y_pred_cls[i],y_ensemb_cls[i])

            # Show the classes as the label on the x-axis.
            ax.set_xlabel(xlabel)
        
        # Remove ticks from the plot.
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Ensure the plot is shown correctly with multiple plots
    # in a single Notebook cell.
    plt.show()
```

## Now we will define our compuational graph

Placeholders for input to graph


```python
x = tf.placeholder('float',shape = [None,28,28,1],name = "x")
y_true = tf.placeholder('float',shape = [None,10],name = "y_true")
```


```python
y_true_cls = tf.argmax(y_true,axis = 1)
```
