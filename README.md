# Single-cell classification based on label-free high-resolution optical data of cell adhesion kinetics

This is the supplementary repository for our paper titled "Single-cell classification based on label-free high-resolution optical data of cell adhesion kinetics". This was originally forked from the [DL-4-TSC](https://github.com/hfawaz/dl-4-tsc) repository for their time series classification [paper](https://link.springer.com/article/10.1007%2Fs10618-019-00619-1) titled "Deep learning for time series classification: a review". The codebase was modified to accomodate our dataset for training and valiadtion. Additionally, it contains our evaluation codebase which can be used to generate the results discussed in our paper.

The single-cell time-series dataset and the paper results are available [here](https://nc.ek-cer.hu/index.php/s/wqE3LHxZdCbsngz). The results contain the trained model parameters and metric valies for every model type and cross validation iteration.

## Prerequisites
All python packages needed are listed in [pip-requirements.txt](https://github.com/hfawaz/dl-4-tsc/blob/master/utils/pip-requirements.txt) file and can be installed simply using the pip command. 
The code now uses Tensorflow 2.0.
The results in the paper were generated using the Tensorflow 1.14 implementation which can be found [here](https://github.com/hfawaz/dl-4-tsc/commit/7ab94a02aedf3a9688e248603bd43c5d405f039b). 
Using Tensorflow 2.0 should give the same results.  
Now [InceptionTime](https://github.com/hfawaz/InceptionTime) is included in the mix, feel free to send a pull request to add another classifier. 

* [numpy](http://www.numpy.org/)  
* [pandas](https://pandas.pydata.org/)  
* [sklearn](http://scikit-learn.org/stable/)  
* [scipy](https://www.scipy.org/)  
* [matplotlib](https://matplotlib.org/)  
* [tensorflow-gpu](https://www.tensorflow.org/)  
* [keras](https://keras.io/)  
* [h5py](http://docs.h5py.org/en/latest/build.html)
* [keras_contrib](https://www.github.com/keras-team/keras-contrib.git)

## Reference

If you re-use our work or the dataset, please cite:

and also cite the original DL-4-TSC paper if you use the time-series classification codebase.