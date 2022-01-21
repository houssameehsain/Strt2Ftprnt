# Strt2Ftprnt
Predicting building footprints given a street network using deep computer vision.
     
![A gif animation featuring 3 images (input - ground truth - output) displaying learning progress of the DNN](images/strt2ftprnt_train.gif)

The model predicts the existence (or non-existence) of building footprint for each pixel through a sigmoid output (far right), this animation depicts the progress the model shows during training from random predictions to pretty accurate ones.  

The following are samples from the model's prediction of building footprints (far right) given various starting street network images (far left) and a ground truth image (middle):

![An image showing a sample from the model's prediction of building footprints given various starting street network images](images/strt2ftprnt_resultSample.jpg)

The models could quickly learn to predict the position of the various building masses relative to the streets and relative to each other, but as could be seen through these images, it fails to predict other details such as courtyard, shaft positions, it only manages to learn those if it overfits the dataset. 

A hyperparameter search (bayesian search) was done using Weights&Biases Sweeps prior to training: 
![](images/strt2ftprnt_pix2pixGAN_sweeps.png)
The best performing hyperparameter combinations are visualized here ..
![](images/strt2ftprnt_pix2pixGAN_best-sweeps.png)
