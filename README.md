# How-KNN-Improve-the-Reliability-of-DNN-

In this project, we explore how KNN improves the reliability of neural networks, including two parts:
* how model performs on data samples with different KNN distances
* whether the KNN distances of out-of-distribution dataset exhibits different distribution compared with in-distribution 

## setups
We use CIFAR10 and CIFAR100 as in-distribution datasets on which we conduct experiments to answer the first problem. We 
solve the second problem on a common out-of-distribution dataset: SVHN. 

## running
#### CIFAR10
```
bash scripts/cifar10.sh 
```

#### CIFAR100
```
bash scripts/cifar100.sh 
```

#### SVHN
```
bash scripts/svhn.sh 
```
