# -*- coding: utf-8 -*-
"""

@author: akito
"""

import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
df1 = pd.read_csv(r"C:\Users\akito\Desktop\subjectE.csv")
df2 = pd.read_csv(r"C:\Users\akito\Desktop\subjectF.csv")

def print_cmx(y_true, y_pred):
    labels = sorted(list(set(y_true)))
    cmx_data = confusion_matrix(y_true, y_pred, labels=labels)
    
    df_cmx = pd.DataFrame(cmx_data, index=labels, columns=labels)
    #plt.figure(figsize = (10,7))
    #sns.heatmap(df_cmx, annot=True)
    plt.show()
    return df_cmx
print("subject-E")
output1 = print_cmx(df1["Emotion"], df1["predict_Emotion"])
print(output1)
print("subject-F")
output2 = print_cmx(df2["Emotion"], df2["predict_Emotion"])
print(output2)