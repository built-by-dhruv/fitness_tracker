import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import scipy
from sklearn.neighbors import LocalOutlierFactor  # pip install scikit-learn

plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (20,5)
plt.rcParams['figure.dpi'] = 100
# --------------------------------------------------------------
# Load data
# --------------------------------------------------------------


data = pd.read_pickle("../../data/interim/01_data_processed.pkl")

# --------------------------------------------------------------
# Plotting outliers
# --------------------------------------------------------------

def plot_binary_outliers(dataset, col, outlier_col, reset_index):
    """ Plot outliers in case of a binary outlier score. Here, the col specifies the real data
    column and outlier_col the columns with a binary value (outlier or not).

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): Column that you want to plot
        outlier_col (string): Outlier column marked with true/false
        reset_index (bool): whether to reset the index for plotting
    """

    # Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/util/VisualizeDataset.py

    dataset = dataset.dropna(axis=0, subset=[col, outlier_col])
    dataset[outlier_col] = dataset[outlier_col].astype("bool")

    if reset_index:
        dataset = dataset.reset_index()

    fig, ax = plt.subplots()

    plt.xlabel("samples")
    plt.ylabel("value")

    # Plot non outliers in default color
    ax.plot(
        dataset.index[~dataset[outlier_col]],
        dataset[col][~dataset[outlier_col]],
        "go",
        # markersize=10 #Increases the size of the plus signs.
    )
    # Plot data points that are outliers in red
    ax.plot(
        dataset.index[dataset[outlier_col]],
        dataset[col][dataset[outlier_col]],
        "ro",
        # markersize=10 #Increases the size of the plus signs.
    )

    plt.legend(
        ["no outlier " + col, "outlier " + col],
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
    )
    plt.show()

def mark_outliers_iqr(dataset, col):
    """Function to mark values as outliers using the IQR method.

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()

    Q1 = dataset[col].quantile(0.25)
    Q3 = dataset[col].quantile(0.75)
    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    dataset[col + "_outlier"] = (dataset[col] < lower_bound) | (
        dataset[col] > upper_bound
    )

    return dataset



outlier_columns = list(data.columns[:6])
data[outlier_columns[:3]+['label']].boxplot(by="label" , figsize=(20,10), layout=(1,3))
data[outlier_columns[3:]+['label']].boxplot(by="label" , figsize=(20,10), layout=(1,3))

dataset = mark_outliers_iqr(data , 'acc_x')

col= 'acc_x' 
outlier_col= f'{col}_outlier' 

    
plot_binary_outliers(
    dataset= dataset ,
    reset_index=False,
    col= col ,
    outlier_col= outlier_col ,
    )

for col in outlier_columns:
    dataset = mark_outliers_iqr(data , col)
    outlier_col= f'{col}_outlier' 
    plot_binary_outliers(
        dataset= dataset ,
        reset_index=True,
        col= col ,
        outlier_col= outlier_col ,
        )


# --------------------------------------------------------------
# Chauvenets criteron (distribution based)
# --------------------------------------------------------------

import seaborn as sns
    # data[outlier_columns[3:]+['label']].plot.hist(by="label" , figsize=(20,10), layout=(3,3))




# Insert Chauvenet's function
def mark_outliers_chauvenet(dataset, col, C=2):
    """Finds outliers in the specified column of datatable and adds a binary column with
    the same name extended with '_outlier' that expresses the result per data point.
    
    Taken from: https://github.com/mhoogen/ML4QS/blob/master/Python3Code/Chapter3/OutlierDetection.py

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        C (int, optional): Degree of certainty for the identification of outliers given the assumption 
                           of a normal distribution, typicaly between 1 - 10. Defaults to 2.

    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column 
        indicating whether the value is an outlier or not.
    """

    dataset = dataset.copy()
    # Compute the mean and standard deviation.
    mean = dataset[col].mean()
    std = dataset[col].std()
    N = len(dataset.index)
    criterion = 1.0 / (C * N)

    # Consider the deviation for the data points.
    deviation = abs(dataset[col] - mean) / std

    # Express the upper and lower bounds.
    low = -deviation / math.sqrt(C)
    high = deviation / math.sqrt(C)
    prob = []
    mask = []

    # Pass all rows in the dataset.
    for i in range(0, len(dataset.index)):
        # Determine the probability of observing the point
        prob.append(
            1.0 - 0.5 * (scipy.special.erf(high[i]) - scipy.special.erf(low[i]))
        )
        # And mark as an outlier when the probability is below our criterion.
        mask.append(prob[i] < criterion)
    dataset[col + "_outlier"] = mask
    return dataset

def mark_outliers_lof(dataset, columns, n=20):
    """Mark values as outliers using LOF

    Args:
        dataset (pd.DataFrame): The dataset
        col (string): The column you want apply outlier detection to
        n (int, optional): n_neighbors. Defaults to 20.
    
    Returns:
        pd.DataFrame: The original dataframe with an extra boolean column
        indicating whether the value is an outlier or not.
    """
    
    dataset = dataset.copy()

    lof = LocalOutlierFactor(n_neighbors=n)
    data = dataset[columns]
    outliers = lof.fit_predict(data)
    X_scores = lof.negative_outlier_factor_

    # dataset["outlier_lof"] = outliers == -1
    dataset[f'{columns[0]}_outlier' ] = outliers == -1
    return dataset, outliers, X_scores



for i in range(0, len(outlier_columns)):
    data[outlier_columns[i:i+1]+['label']].plot.hist(by="label" , figsize=(20,10), layout=(3,3))
    # sns.histplot(data[outlier_columns[i:i+1]+['label']], x=outlier_columns[i], hue='label', kde=True)



    

col= 'acc_x' 
outlier_col= f'{col}_outlier' 
label = "row"
dataset, a,b  = mark_outliers_lof(data[data["label"] == label ] , ['acc_x'])
dataset['acc_x_outlier'].value_counts()

# Filter to get only the rows where the outlier flag is True (outliers are marked as -1)
outliers_data = dataset[dataset["acc_x_outlier"] == True]

# Assuming outliers are marked as True in "outlier_lof" or -1
dataset.loc[dataset[outlier_col] == True, col] = np.nan
dataset.head(50)
# dataset = mark_outliers_chauvenet(data[data["label"] == label] , 'acc_x')
# dataset = dataset.loc[dataset[col+'_outlier']  , col] == np.nan

plot_binary_outliers(
    dataset= dataset ,
    reset_index=True,
    col= col ,
    outlier_col= outlier_col ,
    )


outlier_columns = list(data.columns[:6])
bimodal_outlier_ = ['acc_x', 'acc_y', 'acc_z' , 'row', 'rest']

outlierdf_semiresolved = data.copy()
for col in outlier_columns:
    for label in data['label'].unique():
        print(col,label)

        if col and label in bimodal_outlier_:
            outlierdf , a,b  = mark_outliers_lof(data[data["label"] == label ] , [col])
            # print( 'bm', outlierdf.columns)
        else:
            outlierdf = mark_outliers_chauvenet(data[data["label"] == label] , col)
            # print(outlierdf.columns)
        outlier_col= f'{col}_outlier' 

        # print
        outlierdf.loc[outlierdf[outlier_col] == True, col] = np.nan
        outlierdf_semiresolved.loc[outlierdf_semiresolved["label"] == label, col] = outlierdf[col]
        n_outliers = len(outlierdf) - len(outlierdf[col].dropna())    


        print(f'Found {n_outliers} outliers in {col} for label {label}')
        


outlierdf_semiresolved.info()

outlierdf_semiresolved.to_pickle("../../data/interim/02_data_outliers_removed.pkl")