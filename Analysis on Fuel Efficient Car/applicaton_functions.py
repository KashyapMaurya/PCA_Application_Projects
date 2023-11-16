# %%
import streamlit as st 
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing


def pca_maker(data_import):

    numerical_column_list = []
    categorical_columns_list = []

    for i in data_import.columns:
        if data_import[i].dtype == np.dtype("float64") or data_import[i].dtype == np.dtype("int64"):
            numerical_column_list.append(data_import[i])
        else:
            categorical_columns_list.append(data_import[i])

    numerical_data = pd.concat(numerical_column_list, axis=1)
    categorical_data = pd.concat(categorical_columns_list, axis=1)

    ## Apply method from pandas is used to apply a formula function to an coloumn or an row 
    ## What we did here is used lambda f(x) which execute in one line 
    #  "x" is what we are passing in every columns of our database
    ##  and we fill the Null values with the mean value of the that entire column 

    numerical_data = numerical_data.apply(lambda x: x.fillna(np.mean(x)))


    scaler = StandardScaler()

    scaled_values = scaler.fit_transform(numerical_data)


    pca = PCA()

    pca_data = pca.fit_transform(scaled_values)
    pca_data = pd.DataFrame(pca_data)

    pca_data

    new_columns_names =["PCA_" + str(i) for i in range (1, len(pca_data.columns) +1)]

    columns_mapper = dict(zip(list(pca_data.columns), new_columns_names))

    pca_data = pca_data.rename(columns=columns_mapper)


    output = pd.concat([data_import, pca_data], axis=1)

    return output, list(categorical_data.columns), new_columns_names 