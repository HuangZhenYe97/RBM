#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 21 15:00:42 2020

@author: hzy
"""


import numpy as np
import matplotlib.pyplot as plt
import time
import torch
import torchvision
import torchvision.transforms as transforms
from binary_RBM import *
import argparse





def experiment_of_two_hidden():
    np.random.seed(0)
    n_v = 100
    n_h = 2
    beta = 1
    q = 0.3
    
    xi = np.random.choice([-1,1],[n_v,n_h])
    sigma = gibbs_sampling(beta, xi, alpha = 5)
    rbm = RBM_binary(n_v,n_h, beta, lr = 0.4)
    xi1 = np.random.choice([-1,1],[n_v,1])
    xi2 = sign( (q+1)/2 - np.random.rand(n_v,1))*xi1
    xi_p = np.column_stack((xi1,xi2))
    sigma_p = gibbs_sampling(beta, xi_p, alpha = 5)
    rbm_p = RBM_binary(n_v,n_h, beta, lr = 0.4)
    
    ol = []
    kl = []
    LB = []
    t1 = time.time()
    for i in range(200):
        rbm.adam_update(sigma)
        rbm_p.adam_update(sigma_p)
        ol.append([overlap(np.abs((rbm.ld*1).T.dot(xi))/100),overlap(np.abs((rbm_p.ld*1).T.dot(xi_p))/100)])
        kl.append([rbm.KL,rbm_p.KL])
        LB.append([rbm.LB,rbm_p.LB])
        if i % 20 == 0:
            rbm.lr *= 1
        
    ol = np.asarray(ol).squeeze()
    A = plt.figure(figsize=(6,12))
    marks = ['r', 'b', 'g','b']
    label = [r'q=0, $Q^{1}$',r'q=0, $Q^{2}$',r'q=0.3, $Q^{1}$',r'q=0.3, $Q^{2}$']
    plt.subplot(3,1,1)
    for i in range(2):
        plt.plot([i for i in range(ol.shape[0])], ol[:,i,0],marks[i], label = label[i])
        plt.plot([i for i in range(ol.shape[0])], ol[:,i,1],marks[i+2], label = label[i+2])
    plt.legend()
    plt.xlabel('learning steps')
    plt.ylabel('overlap')

    kl = np.asarray(kl)
    plt.subplot(3,1,2)
    plt.plot([i for i in range(kl.shape[0])],kl[:,0],'b',label = 'q=0')
    plt.plot([i for i in range(kl.shape[0])],kl[:,1],'r',label = 'q=0.3')
    plt.xlabel('learning steps')
    plt.ylabel('KL')
    plt.legend()
    LB = np.asarray(LB)
    plt.subplot(3,1,3)
    plt.plot([i for i in range(LB.shape[0])], LB[:,0],'b',label = 'q=0')
    plt.plot([i for i in range(LB.shape[0])], LB[:,1],'r',label = 'q=0.3')
    plt.legend()
    plt.xlabel('learning steps')
    plt.ylabel('LB')
    plt.show()

def experiment_of_three_hidden():
    np.random.seed(args.random_seed)
    n_v = 100
    n_h = 3
    beta = 1
    
    xi = np.random.choice([-1,1],[n_v,n_h])
    sigma = gibbs_sampling(beta, xi, alpha = 5)
    rbm = RBM_binary(n_v,n_h, beta, lr = 0.4)
    ol = []
    kl = []
    LB = []
    t1 = time.time()
    for i in range(200):
        rbm.adam_update(sigma)
        t2 = time.time()
        
        t1=t2*1
        ol.append(overlap(np.abs((rbm.ld*1).T.dot(xi))/100))
        kl.append(rbm.KL)
        LB.append(rbm.LB)
        if i % 20 == 0:
            rbm.lr *= 1
        
    ol = np.asarray(ol).squeeze()
    A = plt.figure(figsize=(6,12))
    marks = ['r', 'b', 'g']
    label = [r'$Q^{1}$',r'$Q^{2}$',r'$Q^{3}$']
    plt.subplot(3,1,1)
    for i in range(3):
        plt.plot([i for i in range(ol.shape[0])], ol[:,i],marks[i], label = label[i])
    plt.legend()
    plt.xlabel('learning steps')
    plt.ylabel('overlap')

    kl = np.asarray(kl)
    plt.subplot(3,1,2)
    plt.plot([i for i in range(kl.shape[0])],kl,'b',label = 'KL')
    plt.xlabel('learning steps')
    plt.ylabel('KL')
    plt.legend()
    LB = np.asarray(LB)
    plt.subplot(3,1,3)
    plt.plot([i for i in range(LB.shape[0])], LB,'b',label = 'LB')
    plt.legend()
    plt.xlabel('learning steps')
    plt.ylabel('LB')
    plt.show()


def experiment_of_different_alpha():
    np.random.seed(args.random_seed)
    n_v = 100
    n_h = 3
    beta = 1
    q = 0.3
    ol_10 = []
    kl_10 = []
    LB_10 = []
    for i in range(10):
        ol = []
        kl = []
        LB = []
        for j in range(5):
            alpha = j+1
            xi_3 = np.random.choice([-1,1],[n_v,n_h])
            sigma_3 = gibbs_sampling(beta, xi_3, alpha = alpha)
            rbm_3 = RBM_binary(n_v,n_h, beta, lr = 0.4)
            n_h = 2
            xi1 = np.random.choice([-1,1],[n_v,1])
            xi2 = sign( (q+1)/2 - np.random.rand(n_v,1))*xi1
            xi = np.column_stack((xi1,xi2))
            sigma = gibbs_sampling(beta, xi, alpha = alpha)
            xi_p = np.random.choice([-1,1],[n_v,n_h])
            sigma_p = gibbs_sampling(beta, xi_p, alpha = alpha)
            rbm = RBM_binary(n_v,n_h, beta, lr = 0.1)
            rbm_p = RBM_binary(n_v,n_h, beta, lr = 0.1)
            for k in range(100):
                rbm_3.adam_update(sigma_3)
                rbm.adam_update(sigma)
                rbm_p.adam_update(sigma_p)
            t2 = time.time()
            
            t1=t2*1
            ol.append([overlap(np.abs(sign(rbm_3.ld*1).T.dot(xi_3))/100).mean(),
                        overlap(np.abs(sign(rbm.ld*1).T.dot(xi))/100).mean(), 
                       overlap(np.abs(sign(rbm_p.ld*1).T.dot(xi_p))/100).mean(),
                      ])
            kl.append([rbm_3.KL*1,rbm.KL*1,rbm_p.KL*1])
            LB.append([rbm_3.LB*1,rbm.LB*1,rbm_p.LB*1])
            
        ol_10.append(ol)
    ol_10 = np.asarray(ol_10)
    label = ['q=0, P=3', 'q=0, P=2', 'q=0.3, P=2']
    for i in range(3):
        plt.errorbar([i+1 for i in range(ol_10.shape[1])], ol_10[:,:,i].mean(0), ol_10[:,:,i].std(0), label = label[i])
    plt.xlabel(r'$\alpha$')
    plt.ylabel('mean overlap with ground true')
    plt.show()

def experiment_of_MNIST():
    np.random.seed(args.random_seed)
    mnist_train =  torchvision.datasets.MNIST('./', train=True, 
                                               transform=torchvision.transforms.ToTensor(), download=True)
    data_loader = torch.utils.data.DataLoader(mnist_train, batch_size=2000)
    for data, label in data_loader:
        sigma = (data.reshape(-1,784).numpy())
        break
    n_v = 784
    n_h = 4
    beta = 1
    rbm = RBM_binary(n_v,n_h, beta, lr = 0.4)
    kl= []
    LB = []
    t1 = time.time()
    for i in range(200):
        rbm.adam_update(sigma)
        t2 = time.time()
        
        t1 = t2*1
        kl.append(rbm.KL*1)
        LB.append(rbm.LB*1)
    plt.subplot(1,2,1)
    plt.plot([i for i in range(len(kl))],kl,label = 'KL')
    plt.legend()
    plt.subplot(1,2,2)
    plt.plot([i for i in range(len(LB))],LB, label = 'LB')
    plt.legend()
    plt.show()

if __name__ == "__main__":    

    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--random_seed', type=int, default = 0)
    args = parser.parse_args()        
        
print('experiment of two hidden node')
experiment_of_two_hidden()

print('experiment of three hidden node')
experiment_of_three_hidden()

print('experiment of different alpha')
experiment_of_different_alpha()

print('experiment of MNIST')
experiment_of_MNIST()
