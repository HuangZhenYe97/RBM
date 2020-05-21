# Code for analysing RBM
This repository contains codes of differnt algorithms of RBM. The code is in python and most of them are shown in jupyter notebook.
  
##  Statistic mechanics of RBM
RBM is a kind of energy based model and it can complete some unsupervised learning task. Gernerally, maximizing log-likelihood is used to train RBM. However, computing the free-energy or the thermal average term of a RBM require O(2^(N+M)) time complexity. The work [''Advanced mean-ﬁeld theory of the restricted Boltzmann machine''
](https://journals.aps.org/pre/abstract/10.1103/PhysRevE.91.050101) sovles the problems by applying Bethe approximation.   
The code within the folder ['Statistic mechanics of RBM'](https://github.com/HuangZhenYe97/RBM/tree/master/Statistic%20mechanics%20of%20RBM) is a realization of the algorithm. It repeats the experiments of fig.2 and fig.3 in the original paper. One can just open the jupyter file and view the codes and the results.
  
  ## RBM with binary synapse   
  As binary synapses are underivable, the common algorithms, like CD algorithm, fail to train RBM with binary synapse.  In the work ['How data, synapses and neurons interact with each other: a variational principle marrying gradient ascent and message passing'](https://arxiv.org/abs/1911.07662), an algorithm marrying variational method and massages passing can complete such a task. This code is a realization of the algorithom.  
  One can read the code in ipynb file. It contains the detail of code and some experiments reesults. What's more, you can just run:
  ```
  python experiments.py --random_seed=1
  ```
Then experiments results can be obtained after sometime. Note that Pytorch is needed to load data of MNIST.
