"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
import common



def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """

   ###################################################
    # raise NotImplementedError
    #GMM matrix, most data X = 0
    # n, d = X.shape
    # K, _ = mixture.mu.shape
    # log_likelihood = 0
    # post = np.zeros((n,K))
    # dimension=0
    # probability_x =0
    # for i in range(n):
    #     probability_x = 0
    #     dimension = np.sum(X[i,:] > 0)
    #     for j in range(K):
    #         for u in range(d):     #do not count unobserved data X
    #             if X[i][u] != 0:
    #                 post[i][j] += (X[i][u] - mixture.mu[j][u]) **2/(2*mixture.var[j])   #Xu-mu distance
    #         if post[i][j] !=0:
    #             post[i][j] = np.log((mixture.p[j] + 1e-16) * np.power(2*np.pi*mixture.var[j],-dimension/2)) - post[i][j] #f(u,j)
    #      log_likelihood += logsumexp(post[i,:])
    #         u=0
    #
    #     for j in range(K):
    #         if post[i][j] !=0:
    #             post[i][j] = np.exp(post[i][j]-logsumexp(post[i,:]))
    # print(log_likelihood)
    # return post,log_likelihood
    # ############################
    n, d = X.shape
    K, _ = mixture.mu.shape
    log_likelihood = 0
    post = np.zeros((n,K))
    dimension=0
    X_C= np.where(X!=0,X,np.nan)
    for u in range(n):
        dimension =  np.sum(X[u] !=0)
        # if dimension > 0:
        for j in range(K):
            post[u][j]= np.log(mixture.p[j]+1e-16)-dimension/2 *np.log(2*np.pi*mixture.var[j])-\
                        np.nansum((X_C[u]-mixture.mu[j])**2)/(2*mixture.var[j])
        log_likelihood +=logsumexp(post[u,:])
        post[u,:] = np.exp(post[u,:] - logsumexp(post[u,:]))
    # print(log_likelihood)
    return post, log_likelihood
    ##################################################


def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    # raise NotImplementedError
    ############################################
   #GMM matrix most data X =0
    n, K = post.shape
    d =  X.shape[1]
    var = np.zeros(K)
    p = np.ones(K)
    mu =np.zeros((K,d))
    numerator = 0
    denominator = 0
    denominator_mu = 0
    Cu = ()

    for k in range(K):
        p[k] =  sum(post[:,k])/n     #p
        for l in range(d):
            C = np.where(X[:,l]!=0,1,0)        #if x!=0 ,x>0, c=1;otherwise c=0
            denominator_mu = np.dot(np.transpose(post[:, k]), C)   #mu-denominator=sum(post[;,k]),when c=1
            if  denominator_mu >=1:               #check update mu
                mu[k][l] = np.dot(np.transpose(post[:,k]),X[:,l])/denominator_mu
            else:                      #keep same number as the input
                mu[k][l] = mixture.mu[k][l]
        numerator= 0                   #var-numerator
        denominator = 0                #var-denominator
        Cu = ()                         #X>0, index of X
        for u in range(n):
            Cu = np.where(X[u]!=0)      #return tuple
            length = np.sum(X[u] !=0)    #the number of X, when X>0

            numerator += post[u][k] * (np.linalg.norm((X[u][Cu] - mu[k][Cu]),ord=2)**2)
            denominator += post[u][k] *length

        var[k] = max(numerator/denominator, min_variance)      #check var>min_var

    new_mixture = GaussianMixture(mu, var, p)
    return  new_mixture
   ##########################################

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
    # raise NotImplementedError

    #GMM- matrix ,most data X is 0
    old_log_likelihood = None
    new_log_likelihood = None
    while (old_log_likelihood is None or (new_log_likelihood - old_log_likelihood) > 1e-6 * np.abs(new_log_likelihood)):
        old_log_likelihood = new_log_likelihood
        post,new_log_likelihood = estep(X, mixture)
        mixture = mstep(X, post, mixture,min_variance=0.25)

    return mixture, post, new_log_likelihood


    #####################################################################


def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    n, d = X.shape
    K, _ = mixture.mu.shape
    log_likelihood = 0
    post = np.zeros((n,K))
    dimension=0
    X_C= np.where(X!=0,X,np.nan)
    X_pred = np.zeros((n,d))
    mu = np.zeros((K,d))
    p =np.zeros(K)
    for u in range(n):
        dimension =  np.sum(X[u] !=0)
        # if dimension > 0:
        for j in range(K):
            post[u][j]= np.log(mixture.p[j]+1e-16)-dimension/2 *np.log(2*np.pi*mixture.var[j])-\
                        np.nansum((X_C[u]-mixture.mu[j])**2)/(2*mixture.var[j])

        post[u,:] = np.exp(post[u,:] - logsumexp(post[u,:]))

    X_pred = np.dot(post,mixture.mu)
    X_pred = np.where(X!=0,X,X_pred)

    return X_pred
