import numpy as np
import pandas as pd


def evalute_accuracy(result, true, label, rd, name):

    correct = 0
    for i in range(0, len(true)):
        if true[i] == label[i]:
            correct += 1
    with open(f"client_model/{rd}/logging.txt", 'a', encoding='utf-8') as f:
        print(f"  {name} Accuracy: {correct / len(true)}", file=f)

    dict = {}
    for i in sorted(list(np.unique(true))):
        class_indices = np.where(np.array(true) == i)[0]
        class_correct = 0
        dict[i] = []
        for j in class_indices:
            if true[j] == label[j]:
                class_correct += 1
            else:
                dict[i].append(label[j])
        with open(f"client_model/{rd}/logging.txt", 'a', encoding='utf-8') as f:
            print(f"Class {i} Accuracy: {class_correct / len(class_indices)}", file=f)

    true_labels_str = true
    pred_labels_str = label

    # Get all unique class names, ensuring consistent ordering
    unique_labels_str = sorted(list(np.unique(true_labels_str)))

    # 1. Calculate Precision, Recall, F1-Score
    #    classification_report directly generates a text report containing these metrics
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(true_labels_str, pred_labels_str, labels=unique_labels_str, digits=4)

    with open(f"client_model/{rd}/logging.txt", 'a', encoding='utf-8') as f:
        print(f"\nClassification Report:\n{report}", file=f)
    matrix = classification_report(true_labels_str,
                                   pred_labels_str,
                                   labels=unique_labels_str,
                                   digits=4, output_dict=True)
    dataframe = pd.DataFrame(matrix)
    dataframe.to_csv(f"10_27/{rd}/{name}_report.csv", index=True, header=True, sep="\t")

    # 2. Calculate confusion matrix
    #    The confusion matrix shows how many times the model predicted each class for each true class
    cm = confusion_matrix(true_labels_str, pred_labels_str, labels=unique_labels_str)

    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.figure(figsize=(8,6))  # You can adjust the size based on the number of classes
    sns.heatmap(cm,
                annot=True,  # Display values in cells
                fmt="d",  # Format values as integers ("d")
                cmap="Blues",  # Choose color map (e.g.: 'Blues', 'viridis', 'YlGnBu')
                xticklabels=unique_labels_str,  # Set x-axis tick labels
                yticklabels=unique_labels_str,  # Set y-axis tick labels
                linewidths=.5,  # Add thin lines between cells
                cbar=False  # Show color bar
                )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')  # Tilt x-axis labels 45 degrees
    plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    plt.tight_layout()
    plt.savefig(f"client_model/{rd}/confusion_matrix.png", dpi=800, bbox_inches='tight')
    plt.close()

    # (Optional) Use pandas to pretty-print the confusion matrix
    cm_df = pd.DataFrame(cm, index=unique_labels_str, columns=unique_labels_str)

    with open(f"client_model/{rd}/logging.txt", 'a', encoding='utf-8') as f:
        print(f"\nConfusion Matrix:\n{cm_df.to_string()}", file=f)  # Use to_string() to ensure complete printing

    with open(f"client_model/{rd}/logging.txt", 'a', encoding='utf-8') as f:
        print(f"\n\n", file=f)
import numpy as np
import pandas as pd


def evalute_accuracy_for_central(result, true, label, rd, name):

    correct = 0
    for i in range(0, len(true)):
        if true[i] == label[i]:
            correct += 1
    with open(f"centralized_model/{rd}_logging.txt", 'a', encoding='utf-8') as f:
        print(f"  {name} Accuracy: {correct / len(true)}", file=f)

    dict = {}
    for i in sorted(list(np.unique(true))):
        class_indices = np.where(np.array(true) == i)[0]
        class_correct = 0
        dict[i] = []
        for j in class_indices:
            if true[j] == label[j]:
                class_correct += 1
            else:
                dict[i].append(label[j])
        with open(f"centralized_model/{rd}_logging.txt", 'a', encoding='utf-8') as f:
            print(f"Class {i} Accuracy: {class_correct / len(class_indices)}", file=f)

    true_labels_str = true
    pred_labels_str = label

    # Get all unique class names, ensuring consistent ordering
    unique_labels_str = sorted(list(np.unique(true_labels_str)))

    # 1. Calculate Precision, Recall, F1-Score
    #    classification_report directly generates a text report containing these metrics
    from sklearn.metrics import classification_report, confusion_matrix
    report = classification_report(true_labels_str,
                                   pred_labels_str,
                                   labels=unique_labels_str,
                                   digits=4)

    with open(f"centralized_model/{rd}_logging.txt", 'a', encoding='utf-8') as f:
        print(f"\nClassification Report:\n{report}", file=f)

    matrix = classification_report(true_labels_str,
                                   pred_labels_str,
                                   labels=unique_labels_str,
                                   digits=4,output_dict=True)
    dataframe = pd.DataFrame(matrix)
    dataframe.to_csv(f"centralized_model/{rd}_result.csv", index=True, header=True,sep="\t")



    cm = confusion_matrix(true_labels_str, pred_labels_str, labels=unique_labels_str)

    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['font.serif'] = ['Times New Roman']
    import os
    plt.figure(figsize=(6, 4))  # You can adjust the size based on the number of classes
    sns.heatmap(cm,
                annot=True,  # Display values in cells
                fmt="d",  # Format values as integers ("d")
                cmap="Blues",  # Choose color map (e.g.: 'Blues', 'viridis', 'YlGnBu')
                xticklabels=unique_labels_str,  # Set x-axis tick labels
                yticklabels=unique_labels_str,  # Set y-axis tick labels
                linewidths=.5,  # Add thin lines between cells
                cbar=False  # Show color bar
                )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')  # Tilt x-axis labels 45 degrees
    plt.yticks(rotation=0)  # Keep y-axis labels horizontal
    plt.tight_layout()
    plt.savefig(f"centralized_model/{rd}_confusion_matrix.png", dpi=800, bbox_inches='tight')
    plt.close()

    # (Optional) Use pandas to pretty-print the confusion matrix
    cm_df = pd.DataFrame(cm, index=unique_labels_str, columns=unique_labels_str)

    with open(f"centralized_model/{rd}_logging.txt", 'a', encoding='utf-8') as f:
        print(f"\nConfusion Matrix:\n{cm_df.to_string()}", file=f)  # Use to_string() to ensure complete printing


