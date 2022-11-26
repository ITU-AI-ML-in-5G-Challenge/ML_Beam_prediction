## Description of dataset

* ITU_test_dataset.hdf5 -- It contains filtered test dataset.

## Description of .py file

* evaluation.py -- It is used to load the pre-trained model, evaluate data and generate output .csv file

## Description of .pt file

* model_x.pt -- It saves the entire model for scenario x.
* model_param_x.pt -- It only saves the model training parameters for scenario x.
* x can be 31, 32, 33, 34

## Description of .csv file
* test_results.csv -- It contains the results of the test dataset requested in the challenge.

## Software Versions
* python 3.8.10
* CUDA Version: 11.2
* pytorch 1.11.0+cu102