import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


def double_plot(x_list, y_list, y1_list, filename, x_label, y_label, plot_name, legend):
    fig = plt.figure()
    sb = fig.add_subplot(1, 1, 1)
    plt.plot(x_list, y_list, linewidth=2, color='r',label=legend[0])
    plt.plot(x_list, y1_list, linewidth=2,color='b', label=legend[1])
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    sb.yaxis.set_major_formatter(FuncFormatter('{0:.0%}'.format))
    header = plot_name + " Accuracy: " + filename
    plt.title(header)
    plt.legend(loc="upper right")
    plt.savefig(header)
    plt.close(fig)


# Defining the distance measures for KNN
euclidean = lambda x1, x2: np.sqrt(np.sum((x1 - x2) ** 2, axis=-1))
manhattan = lambda x1, x2: np.sum(np.abs(x1 - x2), axis=-1)
hamming = lambda x1, x2 : np.sum(x1 != x2, axis=-1)


#computes misclassification cost by subtracting the maximum probability of any class
def cost_misclassification(labels):
    counts = np.bincount(labels)
    class_probs = counts / np.sum(counts)
    #you could compress both the steps above by doing class_probs = np.bincount(labels) / len(labels)
    return 1 - np.max(class_probs)


#computes entropy of the labels by computing the class probabilities
def cost_entropy(labels):
    class_probs = np.bincount(labels) / len(labels)
    class_probs = class_probs[class_probs > 0]              #this steps is remove 0 probabilities for removing numerical issues while computing log
    return -np.sum(class_probs * np.log(class_probs))       #expression for entropy -\sigma p(x)log[p(x)]


#computes the gini index cost
def cost_gini_index(labels):
    class_probs = np.bincount(labels) / len(labels)
    return 1 - np.sum(np.square(class_probs))               #expression for gini index 1-\sigma p(x)^2


def replace_miss(data):
    d_na = data.replace(to_replace='?', value=np.nan, inplace=False)
    filt = data.nunique() >5 # Adhoc filter for categorical variables
    d_num = d_na[d_na.columns[list(filt)]]
    d_cat = d_na[d_na.columns[list(~filt)]]
    d_num = d_num.fillna(d_num.mean())
    d_cat = d_cat.fillna(d_num.mode())
    d_res = d_num.join(d_cat)
    return d_res

