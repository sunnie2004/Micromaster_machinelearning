"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture
def Gaussian(x,mu,sigma):
    d = x.shape[0]
    gaussian_component = 0
    gaussian_component = np.power(2*np.pi*sigma,-d/2) * np.exp(np.linalg.norm((x-mu),ord=2) **2 /(-2*sigma)) #norm--two vector's distance
    return gaussian_component


def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    ############################################
    #EM ---normal situation
    n, _ = X.shape
    K, _ = mixture.mu.shape
    log_likelihood = 0
    post = np.zeros((n,K))
    probability_X = 0
    for i in range(n):
        probability_X = 0                   #p(j|i)=p(z=k|xi)=P_k*N(mu_k,sigma_k)/sum(P_k *N(mu_k,sigma_k))
        for j in range(K):
            post[i][j] = mixture.p[j] * Gaussian(X[i],mixture.mu[j],mixture.var[j])
            probability_X += post[i][j]
        log_likelihood += np.log(probability_X)    #log_likelihood = sum(n) log(sum(k))
        for j in range(K):
            post[i][j] /= probability_X
    print(log_likelihood)
    return post,log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # EM ---normal situation
    n, K = post.shape
    d =  X.shape[1]
    N_post_mu = np.zeros(K)
    var = np.zeros(K)
    p = np.ones(K)
    mu =np.zeros((K,d))
    numerator = 0

    mu = np.dot(np.transpose(post), X)

    for j in range(K):
        N_post_mu[j] = sum(post[:,j])       #N(k)
        mu[j,:] /= N_post_mu[j]
        p[j] =  N_post_mu[j] / n

        numerator = np.dot(np.dot((X-mu[j,:]),np.transpose(X-mu[j,:])).diagonal(),post[:,j])   #variance numerator
        var[j] = numerator/(d *N_post_mu[j])

    new_mixture = GaussianMixture(mu, var, p)
    return new_mixture
    ########################################


def run(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    ###############################################################

    old_log_likelihood = None
    new_log_likelihood = None
    while (old_log_likelihood is None or (new_log_likelihood - old_log_likelihood) > 1e-6 * np.abs(new_log_likelihood)):
        old_log_likelihood = new_log_likelihood
        post,new_log_likelihood = estep(X, mixture)
        mixture = mstep(X, post)

    return mixture, post, new_log_likelihood
    ####################################################################
