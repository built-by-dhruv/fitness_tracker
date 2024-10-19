##############################################################
#                                                            #
#    Mark Hoogendoorn and Burkhardt Funk (2017)              #
#    Machine Learning for the Quantified Self                #
#    Springer                                                #
#    Chapter 4                                               #
#                                                            #
##############################################################

# Updated by Dave Ebbelaar on 22-12-2022

import numpy as np
import scipy.stats as stats

# Class to abstract a history of numerical values we can use as an attribute.
class NumericalAbstraction:

    # This function aggregates a list of values using the specified aggregation
    # function (which can be 'mean', 'max', 'min', 'median', 'std')
    def aggregate_value(self, aggregation_function):
        # Compute the values and return the result.
        if aggregation_function == "mean":
            return np.mean
        elif aggregation_function == "max":
            return np.max
        elif aggregation_function == "min":
            return np.min
        elif aggregation_function == "median":
            return np.median
        elif aggregation_function == "std":
            return np.std
        else:
            return np.nan

    # Abstract numerical columns specified given a window size (i.e. the number of time points from
    # the past considered) and an aggregation function.
    def abstract_numerical(self, data_table, cols, window_size, aggregation_function) -> tuple:

        # Create new columns for the temporal data, pass over the dataset and compute values
        new_columns = []
        for col in cols:
            column_name = col + "_temp_" + aggregation_function + "_ws_" + str(window_size)
            new_columns.append(column_name)
            data_table[column_name] = (
                data_table[col]
                .rolling(window_size)
                .apply(self.aggregate_value(aggregation_function))
            )

        # print("New columns: ", new_columns)
        return data_table , new_columns


import pandas as pd
if __name__ == "__main__":
    # Create a dataset
    data = np.random.rand(100, 2)
    data_table = pd.DataFrame(data, columns=["abz", "bbz"])

    # Create the numerical abstraction object
    NBS = NumericalAbstraction()

    # Abstract the data
    data_table, new_columns = NBS.abstract_numerical(data_table, ["abz", "bbz"], 10, "mean")

    # Print the result
    print(data_table.head())
    print(new_columns)
    print("Done")