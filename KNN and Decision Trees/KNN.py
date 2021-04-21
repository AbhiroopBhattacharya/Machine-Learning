import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utils_MP1 import hamming, euclidean, manhattan


def calc_distanc(x, t, dist):
    dist_cat, w1 = 0, 0
    x_num = x.select_dtypes(exclude="category").reset_index(drop=True)
    t_num = t.select_dtypes(exclude="category").reset_index(drop=True)
    ax_num = x_num.to_numpy().astype(float)
    tx_num = t_num.to_numpy().astype(float)
    dist_num = dist(ax_num[None, :, :], tx_num[:, None, :])

    if any(t for t in x.dtypes == "category"):
        x_cat = x.select_dtypes(include="category").reset_index(drop=True)
        t_cat = t.select_dtypes(include="category").reset_index(drop=True)
        x_cat_mod = pd.get_dummies(x_cat)
        t_cat_mod = pd.get_dummies(t_cat)
        ax_cat = x_cat_mod.to_numpy().astype(int)
        tx_cat = t_cat_mod.to_numpy().astype(int)
        dist_cat = hamming(ax_cat[None, :, :], tx_cat[:, None, :])

        # As per Nishant's suggestion, calculating w as per ratio of maximum distances
        w1 = np.max(dist_num)/np.max(dist_cat)

    result_dist = w1 * dist_cat + (1 - w1) * dist_num
    return result_dist


class KNN:
    def __init__(self, K, dist):
        self.K=K
        self.dist=dist
        return

    def fit(self,x,y):
        self.x=x
        self.y=y
        self.C= np.max(y)+1
        return self

    def predict(self, x_test):
        ''' Makes a prediction using the stored training data and the test data given as argument'''
        num_test = x_test.shape[0]
        # calculate distance between the training & test samples and returns an array of shape [num_test, num_train]
        ty=self.y.to_numpy().astype(int)
        # distances= calc_dist1(self.x,x_test,dist=self.dist) # Trying Eucl for all
        distances = calc_distanc(self.x,x_test,dist=self.dist)
        # ith-row of knns stores the indices of k closest training samples to the ith-test sample
        knns = np.zeros((num_test, self.K), dtype=int)
        # ith-row of y_prob has the probability distribution over C classes
        y_prob = np.zeros((num_test, self.C))
        # print('y_probs.{} ty.{}'.format(y_prob.shape,y.shape))
        for i in range(num_test):
            knns[i, :] = np.argsort(distances[i])[:self.K]
            y_prob[i, :] = np.bincount(ty[knns[i, :]], minlength=self.C)
        y_prob /= self.K
        return y_prob, knns

    def eval(self, x_test, y_test, y_pred):
        correct = y_test == y_pred
        incorrect = np.logical_not(correct)

        # visualization of Misclassification of data points
        plt.scatter(self.x[:, 0], self.x[:, 1], c=self.y, marker='o', alpha=.2, label='train')
        plt.scatter(x_test[correct, 0], x_test[correct, 1], marker='.', c=y_pred[correct], label='correct')
        plt.scatter(x_test[incorrect, 0], x_test[incorrect, 1], marker='x', c=y_test[incorrect], label='misclassified')
        df_confusion = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

        # Evaluation Metrics and ROC Curve
        fp = np.sum(df_confusion, axis=0) - np.diag(df_confusion)
        tp = np.diag(df_confusion)
        fn = np.sum(df_confusion, axis=1) - np.diag(df_confusion)
        tn = np.sum(df_confusion) - np.sum(fp,tp,fn)
        recall = tp/ (tp+fn)
        precision = tp/(tp+fp)
        fpr = fp/(fp+tn)
        roc=plt.figure()
        plt.plot(fpr,recall, linewidth=2, c='r')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(" ROC Curve")
        plt.close(roc)

