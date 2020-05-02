# Code for analysing RBM
This repository contains codes of differnt algorithms of RBM. The code is in python and most of them are shown in jupyter notebook.
  
## Statistic mechanics of RBM
    RBM is a kind of energy based model and it can complete some unsupervised learning task. Gernerally, maximizing log-likelihood is used to train RBM. However, computing the free-energy or the thermal average term of a RBM require O(2^(N+M)) time complexity. The work [''Advanced mean-ﬁeld theory of the restricted Boltzmann machine''
](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.91.050101) sovles the problems by applying Bethe approximation. The code within the folder ['Statistic mechanics of RBM']() is a realization of the algorithm.
