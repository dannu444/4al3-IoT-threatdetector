# 4al3-IoT-threatdetector

## General Info & Running the model

1) To preprocess the dataset, run ```python preprocess.py <file_path> <input_dest_path> <target_dest_path>```, where ```file_path``` is the relative path to the dataset (```data/RT_IOT2022.csv```), and ```input_dest_path``` & ```target_dest_path``` are the relative paths to where you wish to save the input and target files.

2) ```train.py``` contains the definitions for the neural network, the baseline logistic regression model, the train loop, and helper functions to calculate loss and accuracy. A main function that takes in the previously defined paths to input data and target data is included as an example instance of the model.

3) To run hyperparameter tuning and produce performance plots for both the baseline and the model, run ```python testing.py <input_path> <target_path>```, where ```input_path``` and ```target_path``` are the relative paths to the previously created input and target files from ```preprocess.py```. 

4) To produce a report that profiles the dataset, run ```python profile_dataset.py <file_path> -c <correlation_threshold>``` (flag -p to profile with ydata_profiling), where ```file_path``` is the relative path to the dataset and ```correlation_threshold``` indicates the threshold where the script will check correlation of features above.