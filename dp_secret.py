import numpy as np
def reconstruct_psd_matrix(matrix):
    """
    Reconstruct a matrix via eigenvalue decomposition to ensure it is positive semi-definite (PSD).
    This is the key step to resolve the NaN issue.
    """
    # 1. Compute eigenvalues and eigenvectors. eigh is specifically for symmetric matrices.
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)

    # 2. "Clip" all negative eigenvalues to a tiny positive number.
    #    This ensures the positive semi-definiteness of the matrix.
    eigenvalues[eigenvalues < 0] = 1e-8

    # 3. Reconstruct the diagonal matrix using the clipped eigenvalues
    diag_matrix = np.diag(eigenvalues)

    # 4. Recombine into a new matrix that is guaranteed to be positive semi-definite
    #    Formula: New_Matrix = Eigenvectors * Clipped_Eigenvalues_Diag * Eigenvectors_Transpose
    psd_matrix = eigenvectors @ diag_matrix @ eigenvectors.T

    return psd_matrix
# We continue using the previous manual mean function
def dp_mean_manual(features, epsilon, bounds):
    n_samples, n_features = features.shape
    sensitivity = np.array([(b[1] - b[0]) / n_samples for b in bounds])
    scale = sensitivity / epsilon
    true_mean = np.mean(features, axis=0)
    noise = np.random.laplace(loc=0, scale=scale, size=n_features)
    return true_mean + noise


def dp_covariance_manual_robust(features, epsilon, bounds):
    """
    A robust manual covariance implementation with integrated PSD repair step.
    """
    n_samples, n_features = features.shape

    # (Steps for computing sensitivity and noise remain unchanged)
    ranges = np.array([b[1] - b[0] for b in bounds])
    sensitivity_matrix = np.outer(ranges, ranges) / n_samples
    scale_matrix = sensitivity_matrix / epsilon
    true_cov = np.cov(features, rowvar=False)
    noise = np.random.laplace(loc=0, scale=scale_matrix, size=(n_features, n_features))
    noisy_cov = true_cov + noise
    symmetric_noisy_cov = (noisy_cov + noisy_cov.T) / 2

    robust_noisy_cov = reconstruct_psd_matrix(symmetric_noisy_cov)

    return robust_noisy_cov


def get_dp_stats_fully_manual_robust(features, epsilon, bounds):
    """
    The final, robust, fully manual wrapper function.
    """
    epsilon_for_mean = epsilon*0.2
    epsilon_for_cov = epsilon*0.8

    noisy_mean = dp_mean_manual(features, epsilon_for_mean, bounds)
    noisy_cov = dp_covariance_manual_robust(features, epsilon_for_cov, bounds)

    return noisy_mean, noisy_cov
