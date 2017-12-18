# GR5242 Final Project

## Building Neural Networks for CIFAR-10 Images Classification: Comparing models developed with advanced techniques

![alt text](https://github.com/nkx199611/GR5242finalproject/blob/master/images/cifar_10.png)

### Introduction: 

This project contains Images Classification models built using Neural Networks. User can run the notebooks for each different model and compare their performances with test. Each part of codes is commented detaily. 

**Team Members**

* Wanhua He

* Yuqi Shi 

* Kexin Nie

* Youyang Cao


### Structure ###

The main contents of this project include:

* main.pdf: This is a pdf report giving the backgrund information for our model construction. It also includes analysis for the results.The report is a good place for users who are looking for explanations. The significant outputs and models comparisons are also included here.

* model_[].ipynb: Those are the ipython notebook for each model. In these notebooks, codes are well explained. Notebooks include the full process of data import, training, validation and testing processing, it is easy for users to reproduce the results step by step. 

* data/: This is the python version for our 60000 [CIFAR-10](http://rodrigob.github.io/are_we_there_yet/build/classification_datasets_results.html#43494641522d3130) images. Those images are from 10 different categories with dimension of 32x32x3

### Special Instructions 

* GPU: The notebooks results are mainly computed by GPUs, if users want to rerun the notebook, make sure run with tensorflow-gpu or other GPU devices. We used floyd cloud GPU services for this project.

* Data import: When downloading the data folder, please make sure it is in the same directory with the ipynb. Also, user should always check the path of data

* Tensorboard: Since we run notebook on floyd, we have a cload tensorboard, the user should set up and open local port for tensorboard if running notebook on local machine.

