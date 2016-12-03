import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataExplore import *

#direct: directory in which the following files reside
#e.g. G:/Research/Eugene/Research/vtcri/ieeg data/train_1/

def preprocessDemo(direct):
    df = pd.read_csv(direct + '1_1_0.csv', header=None).as_matrix()
    pre = preprocessEEG(df, 10000, False)
    df_r = pre[0]
    df_v = pre[1]
    df_mm = ((df_r-np.min(df_r))/(np.max(df_r)-np.min(df_r)))*(np.nanmax(df_v)-np.nanmin(df_v))-np.nanmin(df_v)
    plt.plot(df_mm, label = "PCA 1_1_0", color = "blue")
    plt.plot(df_v, label = "Var 1_1_0", color = "blue")
    
    df2 = pd.read_csv(direct + '1_1_1.csv', header=None).as_matrix()
    pre2 = preprocessEEG(df2, 10000, False)
    df_r2 = pre2[0]
    df_v2 = pre2[1]
    df_mm2 = ((df_r2-np.min(df_r2))/(np.max(df_r2)-np.min(df_r2)))*(np.nanmax(df_v2)-np.nanmin(df_v2))-np.nanmin(df_v2)
    plt.plot(df_mm2, label = "PCA 1_1_1", color = "green")
    plt.plot(df_v2, label = "Var 1_1_1", color = "green")
    
    df3 = pd.read_csv(direct + '1_2_0.csv', header=None).as_matrix()
    pre3 = preprocessEEG(df3, 10000, False)
    df_r3 = pre3[0]
    df_v3 = pre3[1]
    df_mm3 = ((df_r3-np.min(df_r3))/(np.max(df_r3)-np.min(df_r3)))*(np.nanmax(df_v3)-np.nanmin(df_v3))-np.nanmin(df_v3)
    plt.plot(df_mm3, label = "PCA 1_2_0", color = "red")
    plt.plot(df_v3, label = "Var 1_2_0", color = "red")
    
    df4 = pd.read_csv(direct + '1_2_1.csv', header=None).as_matrix()
    pre4 = preprocessEEG(df4, 10000, False)
    df_r4 = pre4[0]
    df_v4 = pre4[1]
    df_mm4 = ((df_r4-np.min(df_r4))/(np.max(df_r4)-np.min(df_r4)))*(np.nanmax(df_v4)-np.nanmin(df_v4))-np.nanmin(df_v4)
    plt.plot(df_mm4, label = "PCA 1_2_1", color = "black")
    plt.plot(df_v4, label = "Var 1_2_1", color = "black")
    
    plt.legend(bbox_to_anchor=(0, 1), loc='upper right', ncol=1)
    plt.show()