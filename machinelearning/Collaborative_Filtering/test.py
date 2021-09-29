import numpy as np
import em
import common

# X = np.loadtxt("test_incomplete.txt")
# X_gold = np.loadtxt("test_complete.txt")
#
# K = 4
# n, d = X.shape
# seed = 0

##############
# X = np.array([[2,5,3,0,0],
#  [3, 5, 0, 4,3],
#  [2, 0, 3, 3, 1],
#  [4,0, 4, 5, 2],
#  [3, 4, 0, 0, 4],
#  [1, 0,4,5, 5],
#  [2,5,0,0,1],
#  [3,0,5, 4,3],
#  [0, 5,3,3,3],
#  [2, 0, 0,3,3],
#  [3, 4,3,3, 3],
#  [1, 5,3,0,1],
#  [4, 5, 3, 4,3],
#  [1, 4, 0,5,2],
#  [1,5,3, 3, 5],
#  [3,5,3,4,3],
#  [3, 0, 0, 4,2],
#  [3, 5, 3,5,1],
#  [2,4,5, 5, 0],
#  [2,5,4,4,2]])
# K=4
# Mu=np.array(
# [[2,4,5,5, 0],
#  [3, 5, 0,4,3],
#  [2, 5,4, 4,2],
#  [0,5,3,3,3]])
# Var=np.array([5.93,4.87,3.99,4.51])
# P=np.array([0.25,0.25,0.25,0.25])
#
# mixture = common.GaussianMixture(Mu,Var,P)
# em.fill_matrix(X,mixture)
############################################
X = np.loadtxt("netflix_incomplete.txt")
X_gold = np.loadtxt("netflix_complete.txt")

for seed_random in range(5):
    mixture, post = common.init(X,K=12,seed= seed_random)
    mixture,post,log= em.run(X,mixture,post)
    X_pred = em.fill_matrix(X,mixture)
    print(common.rmse(X_gold,X_pred))
# TODO: Your code here
