from .data import load_data, pca_on_data
from sklearn.model_selection import train_test_split
import pandas as pd

def data_pipeline(
            path_X_train, 
            path_y_train,
            path_X_test,
            path_y_test,
            data_path)

    # configs
    normalize = True
    val_split_size = 0.2
    pca_components = 100

    # load data, normalize
    df_X_train, df_y_train = load_data(path_X_train, path_y_train, normalize=False)
    df_X_test, df_y_test = load_data(path_X_test, path_y_test, normalize=False)
    
    # save normalization parameters
    df_X_mean = df_X_train.mean()
    df_X_std = df_X_train.std()
    
    # normalize sets
    df_X_train = (df_X_train - df_X_mean) / df_X_std
    df_X_test = (df_X_test - df_X_mean) / df_X_std
    
    # save normalization parameters
    df_X_mean.to_csv(data_path + "processed/X_train_mean.csv", index=False)
    df_X_std.to_csv(data_path + "processed/X_train_std.csv", index=False)
    
    # split the data
    X_train, X_val, y_train, y_test = train_test_split(df_X_train, df_y_train, test_size=val_split_size, random_state=42)
    
    # perform PCA only on training data
    X_train, pca = pca_on_data(X_train, n_components=pca_components)
    
    # transform the validation and test data
    X_val = pca.transform(X_val)
    X_test = pca.transform(df_X_test.values)
    
    # save the data as csv
    pd.DataFrame(X_train).to_csv(data_path + "processed/X_train.csv", index=False)
    pd.DataFrame(X_val).to_csv(data_path + "processed/X_val.csv", index=False)
    pd.DataFrame(X_test).to_csv(data_path + "processed/X_test.csv", index=False)
    pd.DataFrame(y_train).to_csv(data_path + "processed/y_train.csv", index=False)
    pd.DataFrame(y_test).to_csv(data_path + "processed/y_test.csv", index=False)
    pd.DataFrame(df_y_test).to_csv(data_path + "processed/y_test_full.csv", index=False)
    
    return
    
    
    
    
    