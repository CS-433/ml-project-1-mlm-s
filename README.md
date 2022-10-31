
# Machine Learning Project 1  : Higgs Boson Challenge

This repository was created to store the the first project of the **CS-433 Machine Learning course**, in which we solve the Higgs Boson Challenge. 

The objective was to perform binary classification on the data, to predict whether a data entry corresponds to a Higgs Boson signal or to background noise. To build the model, we use `train.csv` and to test it, `test.csv`.

To tackle the problem, we started by cleaning and processing the raw data, applied feature engineering and selected an optimal regression technique **ridge regression**. This allowed us to obtain a final accuracy of 0.824 and an F1-score of 0.727 or AICrowd. For more details covering the strategy, please refer to our `report.pdf`.


## Structure

```
├── dataset
│   ├── submission.csv
│   ├── test.csv
│   └── train.csv
├── README.md
├── report.pdf
└── script
    ├── cross_validation.py
    contains the necessary functions to perform cross-validation
    
    ├── data_cleaning.py
    contains the necessary functions to clean the data
    
    ├── data_processing.py
    contains the functions used and replace null values and standardize the data
    
    ├── helper_functions.py
    contains functions used for regression methods, building polynomials and computing accuracies
    
    ├── helpers.py    
    contains the functions used to load the .csv files and build the .csv submission
    
    ├── implementations.py
    contains six regression methods functions used to find the best fit for our model
    
    └── run.py
    central file which the user can directly run to build the optimal model and create the output file
```

## Set up

 `Numpy` is necessary to run our code. It can be installed with the following command: ```pip3 install --user numpy```. No other packages were used.

 You need to **download `train.csv` and `test.csv`** and place them in the  `dataset` folder. These files can be found [here](https://www.kaggle.com/c/higgs-boson/data).

 To **run the code**, you need to execute `run.py`, which is in the `script` folder. This will train the optimal model and provide the output predictions in the `submission.csv`, which will be stored in the `dataset` folder.


## Team

Our team **mlm-s** is composed of three members: Hugo Witz, Maria Tager, Joana Malvar.
