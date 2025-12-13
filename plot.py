import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
#绘制饼状图 每个client数据分区
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'Hiragino Sans', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'Times New Roman'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pickle
import itertools  # 导入itertools，虽然这里不直接用，但有时在client处理中会用到


def plot_stacked_bar_distribution(num_client=6, random_seed=42):
    # --- 1. 定义全局颜色映射 ---
    # 所有可能的电池类型（基于您client_model中的encode/decode）
    ALL_BATTERY_TYPES = [
        '10Ah_LMO', '15Ah_NMC', '21Ah_NMC', '24Ah_LMO',
        '25Ah_LMO', '26Ah_LMO', '35Ah_LFP', '68Ah_LFP'
    ]

    color_palette = plt.cm.get_cmap('tab10', len(ALL_BATTERY_TYPES))
    color_map = {battery_type: color_palette(i) for i, battery_type in enumerate(ALL_BATTERY_TYPES)}

    # --- 2. 收集所有客户端的电池分布数据 ---
    all_client_battery_data = []  # 存储每个客户端的电池分布 Series
    client_labels = []  # 存储客户端ID

    for i in range(num_client):
        client_model_path = f"client_model/{str(random_seed)}/client_{str(i)}.pkl"

        # 确保文件存在
        if not os.path.exists(client_model_path):
            print(f"Warning: Client model {client_model_path} not found. Skipping client {i}.")
            continue

        client_model = pickle.load(open(client_model_path, 'rb'))
        data = client_model.dataset

        battery_counts = data.groupby("condition")["No."].nunique().reindex(ALL_BATTERY_TYPES, fill_value=0)

        all_client_battery_data.append(battery_counts)
        client_labels.append(f'Client {i}')

    # 如果没有收集到数据，提前退出
    if not all_client_battery_data:
        print("No client data found to plot.")
        return

    stacked_df = pd.DataFrame(all_client_battery_data).T
    stacked_df.columns = [f'Client {i}' for i in range(len(all_client_battery_data))]

    # --- 3. 绘图 ---
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 确保支持中文
    matplotlib.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 8))  # 调整图大小

    stacked_df.loc[ALL_BATTERY_TYPES].T.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=[color_map[bt] for bt in ALL_BATTERY_TYPES],  # 使用全局颜色映射
        edgecolor='black'  # 给每个堆叠部分一个边框，增加可读性
    )

    # 完善图表
    ax.set_xlabel('Client ID')
    ax.set_ylabel('Number of Batteries')
    ax.tick_params(axis='x', rotation=45)  # 旋转X轴标签防止重叠
    ax.tick_params(axis='y')
    ax.set_ylim(0, stacked_df.sum(axis=0).max() * 1.2)  # 动态设置y轴上限

    # 添加图例
    # 使用 ALL_BATTERY_TYPES 来保证图例顺序和堆叠顺序一致
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Battery Types", bbox_to_anchor=(1.05, 1), loc='upper left')

    # 为每个条形添加总数标签 (可选，但可以增加信息量)
    for container in ax.containers:
        # 对每个堆叠条形的顶部添加标签
        for rect in container:
            height = rect.get_height()
            if height > 0:  # 只标记非零的高度
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_y() + height / 2,  # 文本位置在堆叠块中间
                        f'{int(height)}',
                        ha='center', va='center',
                        color='black',fontsize = 14)  # 文本颜色和大小

    # 添加每个条形的总数标签
    total_counts = stacked_df.sum(axis=0)
    for i, total in enumerate(total_counts):
        ax.text(i, total + 5, f'Total: {int(total)}', ha='center', va='bottom')

    plt.tight_layout()  # 自动调整布局，防止标签重叠

    fig_dir = f"client_model/{str(random_seed)}/fig/"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f"{fig_dir}client_battery_distribution_stacked_bar.png", dpi=600, bbox_inches='tight')
    plt.close()


def plot(num_client=6, random_seed=42):
    # --- 解决方案：创建全局颜色映射 ---
    # 1. 定义所有可能的电池类型（基于您client_model中的encode/decode）
    ALL_BATTERY_TYPES = [
        '10Ah_LMO', '15Ah_NMC', '21Ah_NMC', '24Ah_LMO',
        '25Ah_LMO', '26Ah_LMO', '35Ah_LFP', '68Ah_LFP'
    ]

    # 2. 创建一个全局的、固定的颜色映射字典
    # 使用一个有足够多不同颜色的色板 (Set3 有 12 种)
    color_palette = plt.cm.Set3(np.linspace(0, 1, len(ALL_BATTERY_TYPES)))
    color_map = {battery_type: color_palette[i] for i, battery_type in enumerate(ALL_BATTERY_TYPES)}

    # --- 解决方案结束 ---

    # 3. 将辅助函数定义移到循环之外
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            if total == 0:
                return f'{pct:.1f}%\n(0)'  # 避免除以零
            val = int(round(pct * total / 100.0))
            return f'{pct:.1f}%\n({val})'

        return my_autopct

    # --- 循环开始 ---
    for i in range(0, num_client):

        client_model_path = f"client_model/{str(random_seed)}/client_{str(i)}.pkl"


        client_model = pickle.load(open(client_model_path, 'rb'))
        data = client_model.dataset

        fig_dir = f"client_model/{str(random_seed)}/fig/"
        os.makedirs(fig_dir, exist_ok=True)

        # --- 修改点：排序并从全局映射中获取颜色 ---
        # 计算数据数量分布
        # .sort_index() 确保标签按字母顺序排列
        data_counts = data["condition"].value_counts().sort_index()
        data_labels = data_counts.index
        data_values = data_counts.values
        # 5. 从全局 color_map 中获取此客户端对应的颜色列表
        data_plot_colors = [color_map[label] for label in data_labels]

        # 计算电池数量分布
        battery_counts = data.groupby("condition")["No."].nunique().sort_index()
        battery_labels = battery_counts.index
        battery_values = battery_counts.values
        # 6. 再次从全局 color_map 中获取颜色
        battery_plot_colors = [color_map[label] for label in battery_labels]
        # --- 修改结束 ---

        # 创建更大的图形
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # 数据数量饼图 - 使用图例
        wedges1, texts1, autotexts1 = ax1.pie(
            data_values,
            labels=None,  # 移除直接标签
            autopct=make_autopct(data_values),
            startangle=90,
            colors=data_plot_colors,  # <-- 7. 使用固定的颜色列表
            textprops={'fontsize': 16},
            pctdistance=0.85
        )

        for autotext in autotexts1:
            autotext.set_color('black')

        # 添加图例
        ax1.legend(wedges1, [f'{label}: {value}' for label, value in zip(data_labels, data_values)],
                   title="Conditions", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        ax1.set_title(f'Client {i} - Data Distribution\nTotal: {sum(data_values)} records', fontsize=12)
        ax1.axis('equal')

        # 电池数量饼图 - 使用图例
        wedges2, texts2, autotexts2 = ax2.pie(
            battery_values,
            labels=None,  # 移除直接标签
            autopct=make_autopct(battery_values),
            startangle=90,
            colors=battery_plot_colors,  # <-- 8. 使用固定的颜色列表
            textprops={'fontsize': 16},
            pctdistance=0.85
        )

        for autotext in autotexts2:
            autotext.set_color('black')

        # 添加图例
        ax2.legend(wedges2, [f'{label}: {value}' for label, value in zip(battery_labels, battery_values)],
                   title="Conditions", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        ax2.set_title(f'Client {i} - Battery Distribution\nTotal: {sum(battery_values)} batteries', fontsize=12)
        ax2.axis('equal')

        plt.tight_layout()
        plt.savefig(f"{fig_dir}client_{i}.png", dpi=600, bbox_inches='tight')
        plt.close()

def similarity_heatmap(i_list,j_list,scores,path):
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib
    def create_similarity_matrix(i_list, j_list, scores):
        """
        根据模型对列表和分数列表，构建一个完整的、对称的相似度矩阵。
        """
        # 找到所有独特的模型ID，并进行排序
        all_ids = sorted(list(set(i_list + j_list)))

        # 创建一个空的DataFrame作为矩阵，用ID作为索引和列名
        similarity_df = pd.DataFrame(index=all_ids, columns=all_ids, dtype=float)

        # 遍历列表，填充矩阵
        for id_i, id_j, score in zip(i_list, j_list, scores):
            # 因为矩阵是对称的，所以两个位置都要填充
            similarity_df.loc[id_i, id_j] = score
            similarity_df.loc[id_j, id_i] = score

        # 用1填充对角线（一个模型与自身的相似度为1）
        np.fill_diagonal(similarity_df.values, 1.0)

        return similarity_df

    # 调用函数创建矩阵
    similarity_matrix = create_similarity_matrix(i_list, j_list, scores)
    with open (f"{path}\\similarity_matrix.txt", "w") as f:
        f.write(similarity_matrix.to_string())


    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # --- 开始绘图 ---
    fig, ax = plt.subplots(figsize=(8, 6))  # 创建一个图形和一个坐标轴

    # cmap 参数和 seaborn 中完全一样
    im = ax.imshow(similarity_matrix.values, cmap="YlGnBu")

    # 创建颜色条
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Similarity", rotation=-90, va="bottom")


    ax.set_xticks(np.arange(len(similarity_matrix.columns)))
    ax.set_yticks(np.arange(len(similarity_matrix.index)))
    ax.set_xticklabels(similarity_matrix.columns)
    ax.set_yticklabels(similarity_matrix.index)

    # 为了防止标签重叠，可以旋转X轴标签
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    # --- 添加数值注释 (这是最关键的一步) ---
    # 循环遍历所有数据点，并在相应位置添加文本
    for i in range(len(similarity_matrix.index)):
        for j in range(len(similarity_matrix.columns)):
            value = similarity_matrix.iloc[i, j]
            # 设置文本颜色：如果背景色太深，文本用白色；否则用黑色
            # 我们取颜色映射范围的一半作为阈值
            threshold = im.norm.vmax / 2.
            text_color = "white" if value > threshold else "black"

            ax.text(j, i, f"{value:.2f}",
                    ha="center", va="center", color=text_color)

    plt.xlabel('Client ID')
    plt.ylabel('Client ID')

    # 如果需要保存图像
    plt.savefig(path+'/similarity_heatmap.png', dpi=600, bbox_inches='tight')
    plt.close()

def plot_dataset():
    name = "mat_data.csv"
    df = pd.read_csv(name, sep="\t", header=0)
    condition = df["condition"].unique()
    import matplotlib.pyplot as plt

    # 准备数据
    conditions = []
    data_nums = []
    battery_nums = []

    for c in condition:
        df_c = df[df["condition"] == c]
        battery_num = len(df_c["No."].unique())
        data_num = len(df_c)

        conditions.append(c)
        battery_nums.append(battery_num)
        data_nums.append(data_num)

    # 绘制柱状图
    fig, ax = plt.subplots(2, 1, figsize=(8, 12),sharex=True)

    # 第一个子图：数据数量
    ax[0].bar(conditions, data_nums, color='skyblue', alpha=0.7)
    ax[0].set_ylim(0, max(data_nums) * 1.2)

    ax[0].set_ylabel('Data Number')
    # 在柱子上添加数值标签
    for i, v in enumerate(data_nums):
        ax[0].text(i, v, str(v), ha='center', va='bottom')

    # 第二个子图：电池数量
    ax[1].bar(conditions, battery_nums, color='lightcoral', alpha=0.7)
    ax[1].set_ylim(0, max(battery_nums) * 1.2)
    # ax[1].set_title('Battery Number by Condition')
    ax[1].set_xticklabels(conditions, rotation=45, ha='right')
    ax[1].set_ylabel('Battery Number')
    # 在柱子上添加数值标签
    for i, v in enumerate(battery_nums):
        ax[1].text(i, v, str(v), ha='center', va='bottom')
    plt.subplots_adjust(hspace=0)
    # plt.tight_layout()
    plt.savefig("dataset.png")

def plot_dataset_SOX():
    name = "mat_data.csv"
    df = pd.read_csv(name, sep="\t", header=0)
    SOH = df["SOH"]
    SOC = df["SOCR"]
    import matplotlib.pyplot as plt

    # 绘制柱状图
    fig, ax = plt.subplots(2, 1, figsize=(8, 10),sharex=False)

    # 第一个子图：数据数量
    bins = list(np.arange(0.3, 1, (1 - 0.3) / 14)) + [max(SOH)]
    ax[0].hist(SOH, bins=bins, color='skyblue', alpha=0.7)
    ax[0].set_ylim(0, 2500)
    ax[0].set_xlim(0.35, 1.0)
    ax[0].set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax[0].set_xticklabels(['0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])

    ax[0].set_ylabel('Data Number')
    ax[0].set_xlabel('SOH')



    # 第二个子图：电池数量
    ax[1].hist(SOC, bins=20, color='lightcoral', alpha=0.7)
    ax[1].set_ylim(0, max(ax[1].get_yticks()) * 1.2)

    ax[1].set_ylabel('Data Number')
    ax[1].set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    ax[1].set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
    ax[1].set_xlabel('SOC')
    # 在柱子上添加数值标签

    plt.tight_layout()
    plt.savefig("dataset_SOX.png",dpi=600)

# for rd in range(100):
#     plot_stacked_bar_distribution(num_client=6, random_seed=rd)
plot_dataset_SOX()
