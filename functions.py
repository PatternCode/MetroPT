import numpy as np
def MulVarGauss(mean_vec,cov_mat,num_samples):
    # Generate num_samples random samples according to a multivariate
    # Gaussian distribution with mean vector mean_vec, and covariance
    # matrix cov_mat. It uses the Cholesky decomposion method to map
    # standard Gaussian samples to the desired samples

    num_features = len(mean_vec)
    standard_normal_samples = np.random.randn(num_samples, num_features)
    
    # Compute the Cholesky decomposition of the covariance matrix
    cholesky_decomposition = np.linalg.cholesky(cov_mat)
    
    # Transform standard normal samples into samples with desired mean and covariance
    return np.dot(cholesky_decomposition, standard_normal_samples.T).T + mean_vec

def hotelling(samples,n_components):
    # This function performes Principal Component Analysis (PCA) on 'samples' and extracts 
    # 'num_of_comp' principal components. It returns a 1-dimensional numpy array of sorted
    # eigenvalues and an 2-dimensional numpy array of eigen vectores of their corresponding
    # eigenvectors
    
    # Center the data by subtracting the mean of each feature
    mean = np.mean(samples, axis=0)
    centered_data = samples - mean
    
    # Calculate the covariance matrix
    covariance = np.cov(centered_data.T)
    
    # Compute eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(covariance)
    
    # Sort eigenvalues and eigenvectors in descending order of eigenvalues
    sorted_index = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_index]
    eigenvectors = eigenvectors[:, sorted_index]
    
    pcs = eigenvectors[:, :n_components]  # Select top n_components eigenvectors
    explained_variance = eigenvalues[:n_components] / np.sum(eigenvalues)  # Explained variance ratio
    return pcs,explained_variance


def normalized(data):
    # Calculate mean and standard deviation for each row
    row_means = np.mean(data, axis=1, keepdims=True)
    row_stds = np.std(data, axis=1, keepdims=True)
    # Normalize each row
    return (data - row_means) / row_stds

def select_random_rows(array, n):
    """
    Randomly selects n rows from a NumPy array.

    Parameters:
        array (numpy.ndarray): Input NumPy array.
        n (int): Number of rows to select.

    Returns:
        numpy.ndarray: NumPy array containing the selected rows.
    """
    num_rows = array.shape[0]
    
    # Generate random indices to select rows
    random_indices = np.random.choice(num_rows, n, replace=False)
    
    # Select rows based on random indices
    selected_rows = array[random_indices]
    
    return selected_rows
    
    
    
    
    
    
    
    
    

