import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import learning_curve
from matplotlib import pyplot as plt

def preprocess(input_size, raw_data, downsample='random', 
               padding='end', random_seed=11):
    """
    Preprocess each data point of raw_data with certain rules given input_size.
    If input_size < raw_data[i].shape[0], downsample with 'random' or 'equal_space';
    If input_size > raw_data[i].shape[0], pad with zeros to the 'beginning' or 'end' or 'both'
    As raw_data[i] has dim of m*3, padding [0, 0, 0] to it
    """
    sampled_data = []
    
    for elem in raw_data:
        if input_size < elem.shape[0]:
            if downsample == 'random':
                np.random.seed(random_seed)
                # randomly sampling the indices for each matrix
                random_idx = np.random.randint(0, elem.shape[0], input_size)
                # sort the indices to match the timestamps
                random_idx.sort()
            else:
                # add "equal_space"
                random_idx = range(input_size)
                pass
            sampled_data.append(np.array(elem[random_idx, :]))
        elif input_size > elem.shape[0]:
            padding_size = input_size - elem.shape[0]
            padding_matrix = np.zeros(dtype=np.float32, 
                                      shape=(padding_size, elem.shape[1]))
            #padding_matrix = np.asarray([0.0]*elem.shape[1]*padding_size, 
                                        #dtype=np.float32).reshape(-1, elem.shape[1])
            if padding == 'end':
                sampled_data.append(np.vstack([elem, padding_matrix]))
            elif padding == 'begin':
                sampled_data.append(np.vstack([padding_matrix, elem]))
            else:
                # need to implement adding to both ends
                sampled_data.append(np.vstack([elem, padding_matrix]))
                pass
        else:
            sampled_data.append(elem)
    
    return sampled_data


# Downsampling + Zero padding
def preprocess_v2(input_size, raw_data, downsample='random', shift_factor=2,
               padding='begin', norm=False, random_seed=11):
    """
    Preprocess each data point of raw_data with certain rules given input_size.
    If input_size < raw_data[i].shape[0], downsample with 'random' or 'equal_space';
    If input_size > raw_data[i].shape[0], pad with zeros to the 'beginning' or 'end' or 'both'
    As raw_data[i] has dim of m*3, padding [0, 0, 0] to it
    norm is implemented in this version not for the demo. SVM performs better with norm=True.
    Neural networks in the demo doesn't need normalized data. 
    """
    
    sampled_data = []
    
    for elem in raw_data:
        if norm:
            for col in range(elem.shape[1]):
                col_mean, col_std = np.mean(elem[:, col]), np.std(elem[:, col])
                elem[:, col] = (elem[:, col] - col_mean) / col_std
                
                
        if input_size < elem.shape[0]:
            if downsample == 'random':
                np.random.seed(random_seed)
                # randomly sampling the indices for each matrix
                random_idx = np.random.randint(0, elem.shape[0], input_size)
                # sort the indices to match the timestamps
                random_idx.sort()
                sampled_data.append(np.array(elem[random_idx, :]))
            elif downsample == 'shift':
                diff = elem.shape[0] - input_size
                shift = int(diff // shift_factor)
                sampled_data.append(np.array(elem[shift:shift+input_size, :]))
            else:
                # add "equal_space"
                random_idx = range(input_size)
                sampled_data.append(np.array(elem[random_idx, :]))
                pass
            
        elif input_size > elem.shape[0]:
            padding_size = input_size - elem.shape[0]
            padding_matrix = np.zeros(dtype=np.float32, 
                                      shape=(padding_size, elem.shape[1]))
            #padding_matrix = np.asarray([0.0]*elem.shape[1]*padding_size, 
                                        #dtype=np.float32).reshape(-1, elem.shape[1])
            if padding == 'end':
                sampled_data.append(np.vstack([elem, padding_matrix]))
            elif padding == 'begin':
                sampled_data.append(np.vstack([padding_matrix, elem]))
            else:
                # need to implement adding to both ends
                sampled_data.append(np.vstack([elem, padding_matrix]))
                pass
        else:
            sampled_data.append(elem)
    
    return sampled_data



def split_data(sampled_data, label, num_classes=10, train_idx=240,
               shuffle=True, feature='linear', degree=3, 
               norm=False, random_seed=11):  
    """
    sampled_data is a list of matrices
    The function shuffles X+y
    splits from train_idx
    extend features
    return 4 matrices
    """
    sample_size = len(sampled_data)
    y = np.eye(num_classes)[np.array(label)]
    if feature == 'linear':
        X = np.asarray(sampled_data, dtype=np.float32).reshape(sample_size, -1)
    elif feature == 'poly':
        tmp_sampled_poly = []
        for elem in sampled_data:
            poly = PolynomialFeatures(degree=degree)
            tmp_sampled_poly.append(poly.fit_transform(elem))
        X = np.asarray(tmp_sampled_poly, dtype=np.float32).reshape(sample_size, -1)
    
    #print(X.shape, y.shape)
    tmp = np.hstack([X, y])
    np.random.seed(random_seed)
    np.random.shuffle(tmp)
    X = tmp[:, :-num_classes]
    y = tmp[:, -num_classes:]
    
    X_train, X_test = X[:train_idx, :], X[train_idx:, :]
    y_train, y_test = y[:train_idx, :], y[train_idx:, :]
    
    return X_train, X_test, y_train, y_test


def mean_std_acc(acc_list, num_seeds=6):
    acc_matrix = np.array(acc_list).reshape(-1, num_seeds)
    acc_mean = acc_matrix.mean(axis=1)
    acc_std = acc_matrix.std(axis=1)
    return acc_mean, acc_std


#from Hongsup
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, shuffle=False, random_state=1,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5), score_metric='accuracy'):
    
    plt.figure(figsize=(8,5))
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel(score_metric)
    
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, shuffle=shuffle,
                                                            random_state=random_state, n_jobs=n_jobs, 
                                                            train_sizes=train_sizes, scoring=score_metric)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    colors = ["#348ABD", "#A60628"]    
    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color=colors[0])
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color=colors[1])
    plt.plot(train_sizes, train_scores_mean, 'o-', color=colors[0],
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color=colors[1],
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    
    return plt