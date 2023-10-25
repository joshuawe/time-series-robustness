import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Tuple

def load_data(
            path_X = "../data/UCI HAR Dataset/train/X_train.txt", 
            path_y = "../data/UCI HAR Dataset/train/y_train.txt",
            normalize = False,
            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the data from the given paths.

    Parameters
    ----------
    path_X : str, optional
        Path to training data X, by default "../data/UCI HAR Dataset/train/X_train.txt"
    path_y : str, optional
        Path to labels for training data X, by default "../data/UCI HAR Dataset/train/y_train.txt"
    normalize : bool, optional
        Whether to normalize the data, by default False

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        The training data X and labels y.
    """
    # read data
    df_X = pd.read_csv(path_X, sep='\s+', header=None)
    df_y = pd.read_csv(path_y, sep='\s+', header=None)
    
    # make sure there is no missingness
    assert df_X.isnull().sum().sum() == 0
    assert df_y.isnull().sum().sum() == 0
    
    # normalize data
    if normalize:
        df_X = (df_X - df_X.mean()) / df_X.std()
    
    return df_X, df_y


def create_corr_plot(df: pd.DataFrame, save_path:str=None) -> None:
    """
    Create a correlation plot for the given dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        The dataframe to plot.
    save_path : str, optional
        The path to where plot can be stored, by default None
    """
    corr = df.corr()
    fig = plt.figure(figsize=(10,9))
    ax = sns.heatmap(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    txt = ax.set_title("Correlation Matrix", fontsize=16)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
    return