import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Utils_MP1 import cost_entropy, cost_gini_index, cost_misclassification


def greedy_test(node, cost_fn):

    #initialize the best parameter values
    best_cost = np.inf
    best_feature, best_value = None, None
    num_instances, num_features = node.data.shape
    #sort the features to get the test value candidates by taking the average of consecutive sorted feature values
    data_sorted = np.sort(node.data[node.data_indices],axis=0)
    test_candidates = (data_sorted[1:] + data_sorted[:-1]) / 2.
    for f in range(num_features):
        #stores the data corresponding to the f-th feature
        data_f = node.data[node.data_indices, f]
        for test in test_candidates[:,f]:
            #Split the indices using the test value of f-th feature
            left_indices = node.data_indices[data_f <= test]
            right_indices = node.data_indices[data_f > test]
            #we can't have a split where a child has zero element
            #if this is true over all the test features and their test values  then the function returns the best cost as infinity
            if len(left_indices) == 0 or len(right_indices) == 0:
                continue
            #compute the left and right cost based on the current split
            left_cost = cost_fn(node.labels[left_indices])
            right_cost = cost_fn(node.labels[right_indices])
            num_left, num_right = left_indices.shape[0], right_indices.shape[0]
            #get the combined cost using the weighted sum of left and right cost
            cost = (num_left * left_cost + num_right * right_cost)/num_instances
            #update only when a lower cost is encountered
            if cost < best_cost:
                best_cost = cost
                best_feature = f
                best_value = test
    return best_cost, best_feature, best_value


class Node:
    def __init__(self, data_indices, parent):
        self.data_indices = data_indices                    #stores the data indices which are in the region defined by this node
        self.left = None                                    #stores the left child of the node
        self.right = None                                   #stores the right child of the node
        self.split_feature = None                           #the feature for split at this node
        self.split_value = None                             #the value of the feature for split at this node
        if parent:
            self.depth = parent.depth + 1                   #obtain the dept of the node by adding one to dept of the parent
            self.num_classes = parent.num_classes           #copies the num classes from the parent
            self.data = parent.data                         #copies the data from the parent
            self.labels = parent.labels                     #copies the labels from the parent
            class_prob = np.bincount(self.labels[data_indices], minlength=self.num_classes) #this is counting frequency of different labels in the region defined by this node
            self.class_prob = class_prob / np.sum(class_prob)  #stores the class probability for the node
            #note that we'll use the class probabilites of the leaf nodes for making predictions after the tree is built


class DecisionTree:
    def __init__(self, num_classes=None, max_depth=3, cost_fn=cost_misclassification, min_leaf_instances=1):
        self.max_depth = max_depth  # maximum dept for termination
        self.root = None  # stores the root of the decision tree
        self.cost_fn = cost_fn  # stores the cost function of the decision tree
        self.num_classes = num_classes  # stores the total number of classes
        self.min_leaf_instances = min_leaf_instances  # minimum number of instances in a leaf for termination

    def fit(self, x, y):
        self.data = x
        self.labels = y
        if self.num_classes is None:
            self.num_classes = np.max(y)+1
        # below are initialization of the root of the decision tree
        self.root = Node(np.arange(x.shape[0]), None)
        self.root.data = x.to_numpy().astype(float)
        self.root.labels = y.to_numpy().astype(int)
        self.root.num_classes = self.num_classes
        self.root.depth = 0
        # to recursively build the rest of the tree
        self._fit_tree(self.root)
        return self

    def _fit_tree(self, node):
        # This gives the condition for termination of the recursion resulting in a leaf node
        if node.depth == self.max_depth or len(node.data_indices) <= self.min_leaf_instances:
            return
        # greedily select the best test by minimizing the cost
        cost, split_feature, split_value = greedy_test(node, self.cost_fn)
        # if the cost returned is infinity it means that it is not possible to split the node and hence terminate
        if np.isinf(cost):
            return
        # print(f'best feature: {split_feature}, value {split_value}, cost {cost}')
        # to get a boolean array suggesting which data indices corresponding to this node are in the left of the split
        test = node.data[node.data_indices, split_feature] <= split_value
        # store the split feature and value of the node
        node.split_feature = split_feature
        node.split_value = split_value
        # define new nodes which are going to be the left and right child of the present node
        left = Node(node.data_indices[test], node)
        right = Node(node.data_indices[np.logical_not(test)], node)
        # recursive call to the _fit_tree()
        self._fit_tree(left)
        self._fit_tree(right)
        # assign the left and right child to present child
        node.left = left
        node.right = right

    def predict(self, data_test):
        y_np=data_test.to_numpy().astype(float)
        class_probs = np.zeros((data_test.shape[0], self.num_classes))
        for n, x in enumerate(y_np):
            node = self.root
            # loop along the dept of the tree looking region where the present data sample fall in based on the split feature and value
            while node.left:
                if x[node.split_feature] <= node.split_value:
                    node = node.left
                else:
                    node = node.right
            # the loop terminates when you reach a leaf of the tree and the class probability of that node is taken for prediction
            class_probs[n, :] = node.class_prob
        return class_probs

    def eval(self, x_test, y_test, y_pred):
        correct = y_test == y_pred
        incorrect = np.logical_not(correct)

        # visualization of Misclassification of data points
        plt.scatter(self.data[:, 0], self.data[:, 1], c=self.labels, marker='o', alpha=.2, label='train')
        plt.scatter(x_test[correct, 0], x_test[correct, 1], marker='.', c=y_pred[correct], label='correct')
        plt.scatter(x_test[incorrect, 0], x_test[incorrect, 1], marker='x', c=y_test[incorrect], label='misclassified')
        df_confusion = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)

        # Evaluation Metrics and ROC Curve
        fp = np.sum(df_confusion, axis=0) - np.diag(df_confusion)
        tp = np.diag(df_confusion)
        fn = np.sum(df_confusion, axis=1) - np.diag(df_confusion)
        tn = np.sum(df_confusion) - np.sum(fp, tp, fn)
        recall = tp / (tp + fn)
        precision = tp / (tp + fp)
        fpr = fp / (fp + tn)
        roc = plt.figure()
        plt.plot(fpr, recall, linewidth=2, c='r')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(" ROC Curve")
        plt.close(roc)

