import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
from KNN_MP1 import KNN
from DT_MP1 import DecisionTree
from Utils_MP1 import *

# Setting the seed
np.random.seed(1234)


def eda_hepatitis(data, target_name):

    # Feature list : Class AGE  SEX  STEROID ANTIVIRALS FATIGUE MALAISE ANOREXIA LIVER_BIG LIVER_FIRM SPLEEN_PALPABLE
    #                SPIDERS ASCITES VARICES BILIRUBIN ALK_PHOSPHATE SGOT  ALBUMIN  PROTIME HISTOLOGY

    # Handling Categorical Variables
    mask_cat = [True, False, True, True, True, True, True, True, True, True, True, True, True, True, False, False,False,
                False, False, True]
    mask_num = [not b for b in mask_cat]
    d_cat = data[data.columns[mask_cat]].reset_index(drop=True)
    d_cat = d_cat.drop(target_name, axis=1, inplace=False)
    d_cat = d_cat.astype("category")


    # Handling numerical variables
    d_num = data[data.columns[mask_num]].reset_index(drop=True)
    d_num = d_num.apply(pd.to_numeric)
    t1 = pd.DataFrame(d_num.describe()).transpose()
    cor_num = d_num.corr()
    t1.rename(columns={t1.columns[0]: "Features"}, inplace=True)
    t1.sort_values(by=["Features"], inplace=True)
    # t1.to_csv("Data Report-{}".format(filename))

    # Standardizing the numerical variables because they have different scales
    d_norm = (d_num - d_num.mean()) / d_num.std()


    cor = data.astype(float).corr()[target_name].drop(target_name, axis=0, inplace=False)
    cor.rename("Features", inplace=True)
    cor.sort_values()
    # cor.to_csv("Target Correlation Report-{}".format(filename))
    # cor_num.to_csv("Feature Correlation Report-{}".format(filename))
    # print(f'Correlation of Features with Target {cor}')

    # Optional- Converting Feature labels into 0,1 from 1,2
    data[target_name] = data[target_name].astype("category").cat.codes
    x, y = d_norm.join(d_cat), data[target_name]

    # Print Statements
    # print("Distribution of Class variable", data[target_name].value_counts())
    # print(f"Correlation of numerical variable {cor_num}")
    # print(f"Description of numerical variable {t1}")
    # print("Categorical Features: {} Categorical corr:{}".format(d_cat.columns.tolist, cor_cat1))
    return x, y


def eda_cancer(data, target_name):

    data = data.drop("id", axis=1, inplace=False)
    x, y = data.drop(target_name, axis=1, inplace=False), data[target_name]
    t1 = pd.DataFrame(x.describe()).transpose()
    t1.rename(columns={t1.columns[0]: "Features"}, inplace=True)
    t1.sort_values(by=["Features"], inplace=True)
    cor = data.corr(method="spearman")[target_name][:].drop(target_name, axis=0, inplace=False)
    cor.sort_values()
    t1["Class_Correlation"] = cor

    # Print Statements
    # print("Distribution of Class variable", data[target_name].value_counts())
    # print(f"Description of features {t1}")
    # t1.to_csv("Data Report-{}".format("filename"))

    return x, y


def visualize(data, target_name, filename):  # Future Work
    grr = pd.plotting.scatter_matrix(data, figsize=(15, 15))
    plt.savefig("Scatter Plot: {}".format(filename))
    # Plotting Heatmap
    # corrmat = data.corr()
    # f, ax = plt.subplots(figsize=(9, 8))
    # sns.heatmap(corrmat, ax=ax, cmap="YlGnBu", linewidths=0.6) 


def read_data(filename, target_name, replace_missing=False):
    d = pd.read_csv(filename)
    # Removing Rows with missing values
    if replace_missing:
        d_clean = replace_miss(d)
    else:
        d_clean = d[~d.eq('?').any(1)]  # Query

    d_clean = d_clean.drop_duplicates(keep='first', inplace=False)

    if filename == 'hepatitis.csv':
        x, y = eda_hepatitis(d_clean, target_name)
    else:
        x, y = eda_cancer(d_clean, target_name)
    return x, y


def train_test_split(x, y, p=0.8):
    num1 = list(np.random.choice(range(len(x)), int(p * len(x)), replace=False))
    msk = x.index.isin(num1)
    x_train, x_test = x[msk], x[~msk]
    y_train, y_test = y[msk], y[~msk]
    return x_train, y_train, x_test, y_test


def train_valid_test_split(x_total, y_total, train_percent=.6, validate_percent=.2):
    x = x_total.reset_index(drop=True)
    y = y_total.reset_index(drop=True)
    perm = np.random.permutation(x.index)
    train_length = int(train_percent*len(x.index))
    validate_length = int(validate_percent*len(x.index)) + train_length
    x_train, y_train = x.iloc[perm[:train_length]], y.iloc[perm[:train_length]]
    x_validate, y_validate = x.iloc[perm[train_length:validate_length]], y.iloc[perm[train_length:validate_length]]
    x_test, y_test = x.iloc[perm[validate_length:]], y.iloc[perm[validate_length:]]

    print(f'X_train {x_train.shape} Y_train {y_train.shape}')
    print(f'X_validate {x_validate.shape} Y_validate {y_validate.shape}')
    print(f'X_test {x_test.shape} Y_test {y_test.shape}')
    print(f'Y_train {np.bincount(y_train)} Y_test {np.bincount(y_test)} Y_validate {np.bincount(y_validate)}')
    return x_train, y_train, x_validate, y_validate, x_test, y_test


def test_KNN(x_train, y_train, x_test, y_test, K, dist):
    model = KNN(K=K, dist=dist)
    y_prob_test, knns = model.fit(x_train, y_train).predict(x_test)
    y_prob_train, knns = model.fit(x_train, y_train).predict(x_train)
    y_pred_test = np.argmax(y_prob_test, axis=-1)
    y_pred_train = np.argmax(y_prob_train, axis=-1)
    accuracy_test = np.sum(y_pred_test == y_test) / y_test.shape[0]
    accuracy_train = np.sum(y_pred_train == y_train) / y_train.shape[0]
    # df_confusion = pd.crosstab(y_test, y_pred_test, rownames=['Actual'], colnames=['Predicted'], margins=True)
    # print(f'Test accuracy is {accuracy_test * 100:.1f}.')
    # print(f'Train accuracy is {accuracy_train * 100:.1f}.')
    # print(df_confusion)
    return accuracy_test, accuracy_train


def test_decision_tree(x_train, y_train, x_test, y_test, max_depth, cf):
    tree = DecisionTree(max_depth=max_depth, cost_fn=cf)
    model_tree = tree.fit(x_train, y_train)
    probs_train = model_tree.predict(x_train)
    probs_test = model_tree.predict(x_test)
    y_pred_train = np.argmax(probs_train, 1)
    y_pred_test = np.argmax(probs_test, 1)
    accuracy_test = np.sum(y_pred_test == y_test) / y_test.shape[0]
    accuracy_train = np.sum(y_pred_train == y_train) / y_train.shape[0]
    # print(f'Test accuracy is {accuracy_test * 100:.1f}.')
    # print(f'Train accuracy is {accuracy_train * 100:.1f}.')
    return accuracy_test, accuracy_train


def cross_validate(x_train, y_train, L, method, dist, d=20, cf=cost_gini_index, K=5):
    x_t = x_train.reset_index(drop=True)
    y_t = y_train.reset_index(drop=True)
    shuffle = np.random.permutation(x_t.index)
    split = round(len(shuffle) / L)
    parts = list([shuffle[i:i + split] for i in range(0, len(shuffle), split)])
    val_acc_train, val_acc_test = [], []
    for p in parts:
        msk = x_t.index.isin(p)
        x_val, y_val = x_t.iloc[p], y_t.iloc[p]
        x_tr, y_tr = x_t.loc[~msk], y_t.loc[~msk]
        # print(" p {} x_val{},y_val{},x_tr{},y_tr{}".format(p, x_val.shape,y_val.shape,x_tr.shape,y_tr.shape))
        if method == "KNN":
            a_test, a_train = test_KNN(x_tr, y_tr, x_val, y_val, K=K, dist=dist)
        else:
            a_test, a_train = test_decision_tree(x_tr, y_tr, x_val, y_val, d, cf=cf)
        val_acc_train.append(a_train)
        val_acc_test.append(a_test)
    return np.mean(val_acc_test), np.mean(val_acc_train)


def hyperparameter_tuning(x_train, y_train, x_test, y_test, filename,x_label, hp_range, method, dist=euclidean,
                          cf=cost_gini_index):
    acc_k_list_test, hp_list = [], []
    acc_k_list_train = []
    for hp in range(1, hp_range):
        if method == 'KNN':
            a_test, a_train = test_KNN(x_train, y_train, x_test, y_test, K=hp, dist=dist)
        else:
            a_test, a_train = test_decision_tree(x_train, y_train, x_test, y_test, hp, cf)
        acc_k_list_test.append(a_test)
        acc_k_list_train.append(a_train)
        hp_list.append(hp)
    double_plot(hp_list, acc_k_list_train, acc_k_list_test, filename,
                x_label=x_label, y_label="Accuracy", plot_name=method, legend=["Train Accuracy","Validation Accuracy"] )


def CV_tuning(x_train, y_train, filename, x_label, l, hp_range, method, dist=euclidean, cf=cost_gini_index):
    acc_k_list_test, hp_list = [], []
    acc_k_list_train = []
    for hp in range(1, hp_range):
        a_test, a_train = cross_validate(x_train, y_train, L=l, method=method, dist=dist, d=hp, cf=cf, K=hp)
        acc_k_list_test.append(a_test)
        acc_k_list_train.append(a_train)
        hp_list.append(hp)
    double_plot(hp_list, acc_k_list_train, acc_k_list_test, filename,
                x_label=x_label, y_label="Accuracy", plot_name=method, legend=["Train Accuracy","Val Accuracy"])


def decision_boundary(x_input, y_input, x_train, y_train, filename, method,var1, var2, k=4, depth=10):

    x = x_input[[var1, var2]].to_numpy().astype(float)
    y_input = y_input.astype("category").cat.codes
    y = y_input.to_numpy().astype(int)
    min1, max1 = x[:, 0].min() - 1, x[:, 0].max() + 1  # 1st feature
    min2, max2 = x[:, 1].min() - 1, x[:, 1].max() + 1  # 2nd feature
    x1_scale = np.arange(min1, max1, 0.1)
    x2_scale = np.arange(min2, max2, 0.1)
    x_grid, y_grid = np.meshgrid(x1_scale, x2_scale)
    x_g, y_g = x_grid.flatten(), y_grid.flatten()
    x_g, y_g = x_g.reshape((len(x_g), 1)), y_g.reshape((len(y_g), 1))
    grid = np.hstack((x_g, y_g))
    if method == 'KNN':
        # make predictions for the grid
        model = KNN(K=k, dist=euclidean).fit(x_train[[var1, var2]], y_train)
        # predict the probability
        p_pred, _ = model.predict(pd.DataFrame(grid))
        p_pred = p_pred[:, 0]
    else:
        tree = DecisionTree(max_depth=depth, cost_fn=cost_gini_index)
        model_tree = tree.fit(x_train[[var1, var2]], y_train)
        p_pred = model_tree.predict(pd.DataFrame(grid))[:, 0]

    # reshaping the results
    pp_grid = p_pred.reshape(x_grid.shape)
    # plot the grid of x, y and z values as a surface
    surface = plt.contourf(x_grid, y_grid, pp_grid, cmap='Pastel1')
    plt.xlim(x_grid.min(), x_grid.max())
    plt.ylim(y_grid.min(), y_grid.max())
    plt.colorbar(surface)
    # create scatter plot for samples from each class
    for class_value in range(2):
        # get row indexes for samples with this class
        row_ix = np.where(y == class_value)
        # create scatter of these samples
        plt.scatter(x[row_ix, 0], x[row_ix, 1], cmap='Pastel1', label=class_value)
    # show the plot
    plt.title('KNN Decision Boundary Dataset:{}'.format(k, filename))
    plt.xlabel(var1)
    plt.ylabel(var2)
    plt.legend()
    plt.savefig(f"Decision Boundary KNN {filename}")


if __name__ == "__main__":
    # Name of the Input file
    filename = 'hepatitis'

    # Name of the Target variable
    target_name = 'Class'

    # Reading the input data and cleaning the data
    # To Replace missing values with mean/mode put replace missing =True
    x, y = read_data(filename + '.csv', target_name)

    # visualize(x, target_name,filename)

    # Splitting the data into test and training datasets
    #x_train, y_train, x_test, y_test = train_test_split(x, y, 0.8)

    # Plotting the decision boundary for KNN
    # decision_boundary(x,y, x_train,y_train, filename=filename, method='KNN', var1="ALBUMIN,", var2="PROTIME")

    # Plotting the decision boundary for Decision Tree
    # decision_boundary(x, y, x_train, y_train,filename, method='DT',depth=10, var1="ALBUMIN,", var2="PROTIME")

    # Splitting the data into train, validate and test datasets
    # x_train, y_train, x_validate, y_validate, x_test, y_test = train_valid_test_split(x, y, train_percent=0.7, validate_percent=0.15)

    # 1) Running K Nearest Neighbour algorithm with a specific value of K and distance
    # a_test, a_train = test_KNN(x_train, y_train, x_test, y_test, K=5, dist=euclidean)
    # print(f"Test Accuracy {a_test} Train Accuracy {a_train}")

    # 2) Using a decision tree classifier with a specific cost function and maximum depth
    # a_test, a_train = test_decision_tree(x_train, y_train, x_test, y_test, max_depth=8, cf=cost_gini_index)
    # print(f"Test Accuracy {a_test} Train Accuracy {a_train}")

    # 3) KNN Hyperparameter tuning -Value of K . Plots training and test accuracy curves
    # hyperparameter_tuning(x_train, y_train, x_validate, y_validate, filename,x_label="K Value", hp_range=20, method='KNN', dist=manhattan)

    # 4) Decision tree Hyperparameter tuning -Value of Max Depth . Plots training and test accuracy curves
    # hyperparameter_tuning(x_train, y_train, x_validate, y_validate, filename, x_label= "Maximum Depth", hp_range=20, method='DT', cf=cost_misclassification)

    # 5) KNN Hyperparameter tuning using Cross Validation. Plots average of training and test accuracy over L folds.
    # CV_tuning(x, y, filename,x_label="K Value",l=4, hp_range=20, method='KNN', dist=euclidean)

    # 6) Decision Tree Hyperparameter tuning using Cross Validation. Plots average of training and test accuracy over L folds.
    # CV_tuning(x, y, filename, x_label="Maximum Depth",l=4, hp_range=20, method='DT', cf=cost_gini_index)

