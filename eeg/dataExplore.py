import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
from sklearn.decomposition import PCA

def preprocessEEG(matrix, window, plot = False):
    """ Takes a data matrix, z-scales the data, performs 1st degree
        PCA on the data, calculates the running variance of the reduced dataset
        with the specified window size. The result of PCA and running variance
        will then be plotted (with the PCA Min-Max scaled to the running 
        variance's min and max) if desired. Returns a tuple containing the 
        result of PCA and running variance.
    """
    df = (matrix - np.mean(matrix, 0)) / np.std(matrix, 0)
    pca = PCA(1)
    # df_r is the 1st principal component
    df_r = pca.fit(df).transform(df)
    df_v = pd.rolling_std(df_r[:,0], window)
    if plot:
        df_mm = ((df_r-np.min(df_r))/(np.max(df_r)-np.min(df_r)))*(np.nanmax(df_v)-np.nanmin(df_v))-np.nanmin(df_v)
        plt.plot(df_mm)
        plt.plot(df_v)
        plt.show()

    return (df_r, df_v)