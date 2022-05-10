import pickle
import numpy as np
from tqdm import tqdm, trange


def train_test_split(X, y, test_size = .33, random_state = 420):
	"""Utility function to split a dataset into train and test folds.
	
	Args:
        X (ndarray): Array of unlabeled data
        y (ndarray): Array of labels for X
        test_size (float): Float between 0 and 1. Determines the size of the test set
        random_state(int): Random state, for reproducibility, or generating different splits
	
	Returns:
        X_train, X_test, y_train, y_test(ndarrrays): Arrays of train/test split Xs and ys
	"""
	data = np.concatenate([X, y[:,None]], axis = 1)
	np.random.seed(random_state)
	np.random.shuffle(data)
	test_data = data[:int(data.shape[0] * test_size), :]
	train_data = data[int(data.shape[0] * test_size):, :]
	X_test = test_data[:, :-1]
	y_test = test_data[:,-1]
	X_train = train_data[:,:-1]
	y_train = train_data[:,-1]
	return X_train, X_test, y_train, y_test

def gaussian_kernel(x_i, x_j, sigma):
    """Computes the Gaussian kernel for the given x_i and x_j.

    Args:
        x_i, x_j (ndarrays): vectors of samples
        sigma (float): hyperparameter that governs the width of the Gaussian kernel

    Returns:
        kernel (float): value of the Gaussian kernel for the given x_i, x_j
    """
    kernel = np.exp(-((np.linalg.norm(x_i - x_j)**2) / (2 * sigma**2)))
    return kernel

def poly_kernel(x_i, x_j, d):
    """Computes the Polynomial kernel for the given x_i and x_j.

    Args:
        x_i, x_j (ndarrays): vectors of samples
        d (int): degree of the polynomial kernel to compute

    Returns:
        kernel (float): value of the Polynomial kernel for the given x_i, x_j
    """
    kernel = (x_i @ x_j) ** d
    return kernel

def gaussian_kernel_matrix(X1, X2, gamma):
    """Given a dataset and a kernel function, returns the kernel matrix
    to be used in regression. Depends on the Gaussian Kernel function.
    
    Args:
        X1, X2 (ndarrays): arrays of data, of shape (n_samples1, n_features), (n_samples2, n_features) 
        gamma (float): Value of sigma to compute the kernel function with.
    
    Returns:
        kernel_matrix (ndarray): Kernel matrix of shape (n_samples1, n_samples2)
    """

    # Take the squared norms of each row
    squared_norm1 = np.sum(X1** 2, axis = 1)
    squared_norm2 = np.sum(X2 ** 2, axis = 1)
    
    # Exploiting the identity: ||x-y||^2 = ||x||^2 + ||y||^2 - 2 * x^T * y  and numpy broadcasting
    K = np.exp( -gamma * (squared_norm1[:, None] + squared_norm2[None, :] - 2 * X1 @ X2.T))
    
    return K

def poly_kernel_matrix(X1, X2, d):
    """Given a dataset and a kernel function, returns the kernel matrix
    to be used in regression. Depends on the Polynomial Kernel function.
    
    Args:
        X1, X2 (ndarrays): arrays of data, of shape (n_samples1, n_features), (n_samples2, n_features)
        d (int): Degree of the polynomial kernel to compute the kernel matrix with
    
    Returns:
        kernel_matrix (ndarray): Kernel matrix of shape (n_samples1, n_samples2)
    """
    
    # First compute gram matrix for the X values
    gram = X1 @ X2.T
    
    # Raise to appropriate power to get polynomial kernel
    return gram ** d


def cross_validation_split(data, k_splits):
    """Splits the dataset provided into k_splits of indices to perform cross validation_indexes
    
    Args:
        data (ndarray): The dataset to split
        k_splits (int): The number of cross validation folds to generate indices for
    
    Returns:
        list_of_folds: list of lists containing n = k_folds (train_indices, validation_indices) tuples of indices.
                        Can be used to index the data when performing cross validation"""
    # Initialise indices
    indexes = np.array(list(range(data.shape[0])))

    # Shuffle indices
    np.random.shuffle(indexes)

    # Split into k folds
    split_list = np.array_split(indexes, 5)

    # Initialise list of lists
    list_of_folds = []

    # Fill list of folds with tuples of (train_indices, validation_indices)
    for i in range(k_splits):
        temp = split_list.copy()
        validation_indexes = list(split_list[i])
        temp.pop(i)
        train_indexes = list(np.concatenate(temp, axis = 0))
        list_of_folds.append((train_indexes, validation_indexes))
    return list_of_folds

def test():
    print('test passed')
    
    
def sign(x):
    """A conservative version of the signum function that handels the = 0 case by assigning -1"""
    if x <= 0:
        return -1
    else: 
        return 1
    
def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)