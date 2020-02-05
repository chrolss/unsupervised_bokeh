from sklearn.cluster import KMeans
import pandas as pd
import numpy as np


def clean_columns(dataset):
    dataset.columns = dataset.columns.str.replace('.', '')
    dataset.columns = dataset.columns.str.replace(' ', '')
    dataset.columns = dataset.columns.str.lower()

    return dataset


def generate_dataset(dataset, keys, keep_columns='ALL', remove_columns='None'):
    if remove_columns != 'None':
        for dropcolumn in remove_columns:
            dataset = dataset.drop([dropcolumn], axis=1)
    else:
        pass

    if keep_columns == 'ALL':
        unsupervised_cols = dataset.columns
        for key in keys:
            unsupervised_cols.remove(key)
    else:
        unsupervised_cols = keep_columns

    temp_df = dataset

    if keys[1] == 'None':
        keys = keys[0]

    if type(keys) == str:
        temp_df.index = temp_df[keys]
    else:
        temp_df['new_index_col'] = temp_df.apply(lambda row: str(row[keys[0]]) + ' - ' + str(row[keys[1]]), axis=1)
        temp_df.index = temp_df.new_index_col
        temp_df = temp_df.drop(['new_index_col'], axis=1)

    temp_df = temp_df.drop(keys, axis=1)

    numerical_features = [cat for cat in unsupervised_cols if dataset[cat].dtype != 'object']
    categorical_featurs = [cat for cat in unsupervised_cols if dataset[cat].dtype == 'object']

    for catf in categorical_featurs:
        temp_df = pd.concat([temp_df, pd.get_dummies(temp_df[catf])], axis=1)
        temp_df.drop([catf], axis=1, inplace=True)

    return temp_df, numerical_features, categorical_featurs


def create_analytics_dataframe(dataset, keys, keep_columns='ALL', remove_columns='None'):
    if remove_columns != 'None':
        for dropcolumn in remove_columns:
            dataset = dataset.drop([dropcolumn], axis=1)
    else:
        pass

    if keep_columns == 'ALL':
        unsupervised_cols = dataset.columns.copy()
    else:
        unsupervised_cols = keep_columns.copy()
        for key in keys:
            if key != 'None':
                unsupervised_cols.append(key)
            else:
                pass

    if keys[1] == 'None':
        keys = keys[0]

    temp_df = dataset[unsupervised_cols]
    if type(keys) == str:
        temp_df.index = temp_df[keys]
        unsupervised_cols.remove(keys)

    else:
        temp_df['new_index_col'] = temp_df.apply(lambda row: str(row[keys[0]]) + ' - ' + str(row[keys[1]]), axis=1)
        temp_df.index = temp_df.new_index_col
        temp_df = temp_df.drop(['new_index_col'], axis=1)
        for key in keys:
            unsupervised_cols.remove(key)

    temp_df = temp_df.drop(keys, axis=1)

    categorical_featurs = [cat for cat in unsupervised_cols if temp_df[cat].dtype == 'object']

    for catf in categorical_featurs:
        temp_df = pd.concat([temp_df, pd.get_dummies(temp_df[catf])], axis=1)
        temp_df.drop([catf], axis=1, inplace=True)

    return temp_df


def find_optimal_clusters(dataframe, kmin=1, kmax=10):
    # Takes the dataset and fits it to different KMeans models where n_clusters vary from kmin to kmax
    # Returns two lists: nr of clusters and model inertia which can be plotted
    inertias = []
    clusters = [i for i in range(kmin, kmax+1)]
    for i in range(kmin, kmax+1):
        model = KMeans(n_clusters=i)
        model.fit(dataframe)
        inertias.append(model.inertia_)

    return clusters, inertias