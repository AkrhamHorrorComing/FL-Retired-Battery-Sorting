# Split training and test sets at 8:2 ratio
import os.path
import pandas as pd
import numpy as np
import random
import ast


def split_dataset(random_seed=42):

    df = pd.read_csv("mat_data.csv",sep="\t",header=0,index_col=0)

    condition = df["condition"].unique()

    for c in condition:
        df_c = df[df["condition"] == c]
        ID = df_c["No."].unique()

        random.seed(random_seed)

        # Randomly shuffle indices
        indices = np.random.permutation(ID)
        # Calculate split point
        split_point_1 = int(len(ID) * 0.8)
        # split_point_2 = int(len(ID) * 0.8)

        # Split into training and test sets
        train_ids = np.sort(indices[:split_point_1])
        # valid_ids = np.sort(indices[split_point_1:split_point_2])
        test_ids = np.sort(indices[split_point_1:])

        DF = pd.DataFrame({
            "name": [c],
            "train": [train_ids.tolist()],  # Use the entire array as one element
            # "valid": [valid_ids.tolist()],  # Use the entire array as one element
            "test": [test_ids.tolist()]  # Also use the entire array as one element

        })
        flag = os.path.exists("split_dataset//"+str(random_seed)+".csv")
        DF.to_csv("split_dataset//"+str(random_seed)+".csv",index=False,header=not flag,mode='a',sep='\t')

def read_data(random_seed=42):
    df = pd.read_csv("split_dataset//"+str(random_seed)+".csv",sep="\t")
    df["train"] = df["train"].apply(ast.literal_eval)  # Safely convert string to list
    # df["valid"] = df["valid"].apply(ast.literal_eval)
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

        # if not os.path.exists("data\\valid"+str(random_seed)+".csv"):
        #     data_valid = data[data["No."].isin(df[df["name"] == conditon]["valid"].values[0])]
        #     data_valid.to_csv("data\\valid"+str(random_seed)+".csv", index=False, sep="\t", header=1,mode="a")
        # else:
        #     data_valid = data[data["No."].isin(df[df["name"] == conditon]["valid"].values[0])]
        #     data_valid.to_csv("data\\valid"+str(random_seed)+".csv", index=False, sep="\t", header=None,mode="a")

        if not os.path.exists("data\\test"+str(random_seed)+".csv"):
            data_test = data[data["No."].isin(df[df["name"] == conditon]["test"].values[0])]
            data_test.to_csv("data\\test"+str(random_seed)+".csv", index=False, sep="\t", header=1,mode="a")
        else:
            data_test = data[data["No."].isin(df[df["name"] == conditon]["test"].values[0])]
            data_test.to_csv("data\\test"+str(random_seed)+".csv",index=False,sep="\t",header=None,mode="a")

for seed in range(100):
    split_dataset(seed)
    read_data(seed)


