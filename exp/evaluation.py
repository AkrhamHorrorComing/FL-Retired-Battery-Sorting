import numpy as np
import pandas as pd


def evalute_accuracy(result, true, label, rd, name):

    correct = 0
    for i in range(0, len(true)):
        if true[i] == label[i]:
            correct += 1
    with open(f"client_model/{rd}/logging.txt", 'a', encoding='utf-8') as f:
        print(f"🚀  {name} Accuracy: {correct / len(true)}", file=f)

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

    # 获取所有唯一类别名称，确保顺序一致
    unique_labels_str = sorted(list(np.unique(true_labels_str)))

    # 1. 计算 Precision, Recall, F1-Score
    #    classification_report 直接生成包含这些指标的文本报告
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

    # 2. 计算混淆矩阵
    #    混淆矩阵显示了模型将每个真实类别预测为各个类别的次数
    cm = confusion_matrix(true_labels_str, pred_labels_str, labels=unique_labels_str)

    import matplotlib.pyplot as plt
    import seaborn as sns
    plt.rcParams['font.serif'] = ['Times New Roman']
    plt.figure(figsize=(8,6))  # 您可以根据类别数量调整大小
    sns.heatmap(cm,
                annot=True,  # 在单元格中显示数值
                fmt="d",  # 将数值格式化为整数 ("d")
                cmap="Blues",  # 选择颜色映射 (例如: 'Blues', 'viridis', 'YlGnBu')
                xticklabels=unique_labels_str,  # 设置 x 轴刻度标签
                yticklabels=unique_labels_str,  # 设置 y 轴刻度标签
                linewidths=.5,  # 在单元格之间添加细线
                cbar=False  # 显示颜色条
                )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')  # 让 x 轴标签倾斜 45 度
    plt.yticks(rotation=0)  # y 轴标签保持水平
    plt.tight_layout()
    plt.savefig(f"client_model/{rd}/confusion_matrix.png", dpi=800, bbox_inches='tight')
    plt.close()

    # (可选) 使用 pandas 美化混淆矩阵的打印
    cm_df = pd.DataFrame(cm, index=unique_labels_str, columns=unique_labels_str)

    with open(f"client_model/{rd}/logging.txt", 'a', encoding='utf-8') as f:
        print(f"\nConfusion Matrix:\n{cm_df.to_string()}", file=f)  # 使用 to_string() 保证完整打印

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
        print(f"🚀  {name} Accuracy: {correct / len(true)}", file=f)

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

    # 获取所有唯一类别名称，确保顺序一致
    unique_labels_str = sorted(list(np.unique(true_labels_str)))

    # 1. 计算 Precision, Recall, F1-Score
    #    classification_report 直接生成包含这些指标的文本报告
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
    plt.figure(figsize=(6, 4))  # 您可以根据类别数量调整大小
    sns.heatmap(cm,
                annot=True,  # 在单元格中显示数值
                fmt="d",  # 将数值格式化为整数 ("d")
                cmap="Blues",  # 选择颜色映射 (例如: 'Blues', 'viridis', 'YlGnBu')
                xticklabels=unique_labels_str,  # 设置 x 轴刻度标签
                yticklabels=unique_labels_str,  # 设置 y 轴刻度标签
                linewidths=.5,  # 在单元格之间添加细线
                cbar=False  # 显示颜色条
                )
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45, ha='right')  # 让 x 轴标签倾斜 45 度
    plt.yticks(rotation=0)  # y 轴标签保持水平
    plt.tight_layout()
    plt.savefig(f"centralized_model/{rd}_confusion_matrix.png", dpi=800, bbox_inches='tight')
    plt.close()

    # (可选) 使用 pandas 美化混淆矩阵的打印
    cm_df = pd.DataFrame(cm, index=unique_labels_str, columns=unique_labels_str)

    with open(f"centralized_model/{rd}_logging.txt", 'a', encoding='utf-8') as f:
        print(f"\nConfusion Matrix:\n{cm_df.to_string()}", file=f)  # 使用 to_string() 保证完整打印


