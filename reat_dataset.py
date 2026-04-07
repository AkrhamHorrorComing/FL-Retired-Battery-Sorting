import os.path

import pandas as pd
import ast
def read_data(random_seed=42):
    df = pd.read_csv("split_dataset//"+str(random_seed)+".csv",sep="\t")
    df["train"] = df["train"].apply(ast.literal_eval)  # Safe conversion method
    df["test"] = df["test"].apply(ast.literal_eval)
    data_all = pd.read_csv('mat_data.csv',sep="\t",index_col=0)

    for conditon in df["name"]:
        data = data_all[data_all["condition"] == conditon]
        if not os.path.exists("data\\train"+str(random_seed)+".csv"):
            data_train = data[data["No."].isin(df[df["name"] == conditon]["train"].values[0])]
            data_train.to_csv("data\\train"+str(random_seed)+".csv", index=False,sep="\t",header=1,mode="a")
            print("{} is saved".format(conditon))
        else:
            data_train = data[data["No."].isin(df[df["name"] == conditon]["train"].values[0])]
            data_train.to_csv("data\\train"+str(random_seed)+".csv", index=False,sep="\t",header=None,mode="a")
            print("{} is saved".format(conditon))

        if not os.path.exists("data\\test"+str(random_seed)+".csv"):
            data_test = data[data["No."].isin(df[df["name"] == conditon]["test"].values[0])]
            data_test.to_csv("data\\test"+str(random_seed)+".csv", index=False, sep="\t", header=1,mode="a")
        else:
            data_test = data[data["No."].isin(df[df["name"] == conditon]["test"].values[0])]
            data_test.to_csv("data\\test"+str(random_seed)+".csv",index=False,sep="\t",header=None,mode="a")

read_data()
