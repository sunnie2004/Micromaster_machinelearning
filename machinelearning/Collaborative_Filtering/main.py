import numpy as np
import kmeans
import common
import naive_em
import em

# X = np.loadtxt("toy_data.txt")
#kmean test min cost from different K and seeds
# cost_list = []
#
# for i in range(4):
#     for seed_random in range(5):
#         mixture, post = common.init(X,K=i+1,seed= seed_random)
#         mixture,post,cost = kmeans.run(X,mixture,post)
#         cost_list.append(cost)
#     print(np.min(cost_list))
#     cost_list.clear()

#test estep and mstep by k=3,seed=0,maxlog=-1388.0818004
# mixture, post = common.init(X,K=3,seed=0)
# naive_em.estep(X,mixture)
# naive_em.run(X,mixture,post)

#kmean test max log_likelihood from different K and seeds
# log_list = []
#
# for i in range(4):
#     for seed_random in range(5):
#         mixture, post = common.init(X,K=i+1,seed= seed_random)
#         mixture,post,log= em.run(X,mixture,post)
#         log_list.append(log)
#     print(np.max(log_list))
#     log_list.clear()

# compare EM and K-means by K cluters
# for i in range(4):
#     # mixture, post = common.init(X, K=i + 1, seed=0)
#     # mixture, post, cost = kmeans.run(X, mixture, post)
#     # common.plot(X,mixture,post,title='K-mean')
#     mixture, post = common.init(X, K=i + 1, seed=10)
#     mixture, post, log = em.run(X, mixture, post)
#     common.plot(X, mixture, post, title='EM')

# BIC comparison under K in EM algorithm
# BIC_list = []
# log_list = []
#
# for i in range(4):
#     for seed_random in range(5):
#         mixture, post = common.init(X,K=i+1,seed=seed_random)
#         mixture,post,log= em.run(X,mixture,post)

#test likelihood by GMM k=1,k=12 for seeds=0,1,2,3,4
X = np.loadtxt("netflix_incomplete.txt")
log_list = []

for seed_random in range(5):
    mixture, post = common.init(X,K=1,seed= seed_random)
    mixture,post,log= em.run(X,mixture,post)
    log_list.append(log)
print(np.max(log_list))
log_list.clear()
for seed_random in range(5):
    mixture, post = common.init(X,K=12,seed= seed_random)
    mixture,post,log= em.run(X,mixture,post)
    log_list.append(log)
print(np.max(log_list))
log_list.clear()
# # TODO: Your code here
