#创建client model，每个client model有不定type电池，每类电池至少有minisize=5块。
import os.path

import pandas as pd
import numpy as np
import random
from client_model import client_model

def _get_random_partitions_with_min(total_sum, num_bins, min_size, rng):

    if total_sum < num_bins * min_size:
        raise ValueError(
            f"总数 {total_sum} 不足以满足 {num_bins} 份每份最少 {min_size} 的要求。"
            f"至少需要 {num_bins * min_size}。"
        )

    base_partitions = np.full(num_bins, min_size, dtype=int)

    remaining_sum = total_sum - (num_bins * min_size)

    if remaining_sum == 0:
        return base_partitions


    sum_to_distribute = remaining_sum + num_bins
    cut_points = rng.choice(np.arange(1, sum_to_distribute), size=num_bins - 1, replace=False)
    all_points = np.concatenate(([0], np.sort(cut_points), [sum_to_distribute]))
    random_additions = np.diff(all_points) - 1
    final_partitions = base_partitions + random_additions
    rng.shuffle(final_partitions)

    return final_partitions


# def partition_random_quantities(df_data, num_clients=5, mini_type=2,max_type=5, random_seed=42):
#     """
#     遵循“先选type再选数据”的逻辑，以随机数量将数据集完全分配。
#     """
#     print("--- 开始按类型进行“随机数量”的数据分配 ---")
#     rng = np.random.default_rng(random_seed)
#
#     all_types = df_data["type"].unique()
#
#     client_type_map = []
#     for _ in range(num_clients):
#         num_types = rng.integers(mini_type, max_type + 1)
#         types_for_client = rng.choice(all_types, size=num_types, replace=False)
#         client_type_map.append(types_for_client.tolist())
#
#     if not os.path.exists("client_model/"+str(random_seed)):
#         os.makedirs("client_model/"+str(random_seed))
#
#     with open("client_model/"+str(random_seed)+"/logging.txt", 'a', encoding='utf-8') as f:
#         print("--- 分配计划 (每个客户端将接收的类型) ---",file=f)
#         for i, types in enumerate(client_type_map):
#             print(f"客户端 {i}: 将接收 {len(types)} 种类型 -> {sorted(types)}",file=f)
#
#     # 步骤 2: 按计划分配数据
#     final_clients_data = [{} for _ in range(num_clients)]
#
#     final_nums = [[] for _ in range(num_clients)]
#
#     for battery_type in all_types:
#         eligible_clients_indices = [
#             i for i, types in enumerate(client_type_map) if battery_type in types
#         ]
#
#         if not eligible_clients_indices:
#             continue
#
#         data_for_type = df_data[df_data["type"] == battery_type]["nums"].values
#         rng.shuffle(data_for_type)
#
#         num_data_points = len(data_for_type)
#         num_eligible_clients = len(eligible_clients_indices)
#
#         random_quantities = _get_random_partitions_with_min(num_data_points,
#                                                             num_eligible_clients, 5, rng)
#
#         current_pos = 0
#         for i, client_idx in enumerate(eligible_clients_indices):
#             quantity = random_quantities[i]
#             data_slice = data_for_type[current_pos: current_pos + quantity]
#             final_clients_data[client_idx][battery_type] = data_slice.tolist()
#             final_nums[client_idx].extend(data_slice.tolist())
#             current_pos += quantity
#     sum_length = sum([len(l) for l in final_nums])
#     with open("client_model/" + str(random_seed) + "/logging.txt", 'a', encoding='utf-8') as f:
#         print(f"分配完成，共{sum_length}/{len(df_data)}个数据",file=f)
#     if sum_length != len(df_data):
#         raise ValueError("分配的数据数量与总数据数量不一致！")
#     return final_clients_data, final_nums
def partition_random_quantities(df_data, num_clients=5, mini_type=2, max_type=5, random_seed=42):
    """
    改进版本：确保所有数据都被分配，固定最小分配量为5
    """
    print("--- 开始按类型进行数据分配 ---")
    rng = np.random.default_rng(random_seed)

    all_types = df_data["type"].unique()

    # 步骤 1: 为每个客户端分配类型
    client_type_map = []
    for _ in range(num_clients):
        num_types = rng.integers(mini_type, max_type + 1)
        types_for_client = rng.choice(all_types, size=num_types, replace=False)
        client_type_map.append(types_for_client.tolist())

    if not os.path.exists("client_model/" + str(random_seed)):
        os.makedirs("client_model/" + str(random_seed))

    with open("client_model/" + str(random_seed) + "/logging.txt", 'a', encoding='utf-8') as f:
        print("--- 分配计划 (每个客户端将接收的类型) ---", file=f)
        for i, types in enumerate(client_type_map):
            print(f"客户端 {i}: 将接收 {len(types)} 种类型 -> {sorted(types)}", file=f)

    # 步骤 2: 按计划分配数据
    final_clients_data = [{} for _ in range(num_clients)]
    final_nums = [[] for _ in range(num_clients)]

    # 记录分配统计
    allocation_stats = {}
    total_allocated = 0

    for battery_type in all_types:
        eligible_clients_indices = [
            i for i, types in enumerate(client_type_map) if battery_type in types
        ]

        if not eligible_clients_indices:
            print(f"警告: 类型 {battery_type} 没有被任何客户端选择，将随机分配给一个客户端")
            # 随机选择一个客户端接收这个类型的数据
            eligible_clients_indices = [rng.integers(0, num_clients)]

        data_for_type = df_data[df_data["type"] == battery_type]["nums"].values
        #对应type的电池ID
        rng.shuffle(data_for_type)

        num_data_points = len(data_for_type)
        num_eligible_clients = len(eligible_clients_indices)

        # 记录统计信息
        allocation_stats[battery_type] = {
            'data_points': num_data_points,
            'eligible_clients': num_eligible_clients
        }

        # 固定最小分配量为5
        fixed_min_size = 5

        # 检查数据量是否足够
        if num_data_points < num_eligible_clients * fixed_min_size:
            print(f"警告: 类型 {battery_type} 数据量不足 ({num_data_points} < {num_eligible_clients * fixed_min_size})")
            # 减少有资格的客户端数量以满足最小分配要求
            max_possible_clients = num_data_points // fixed_min_size
            if max_possible_clients == 0:
                # 如果连一个客户端的最小要求都无法满足，选择数据量最多的客户端
                print(f"类型 {battery_type} 数据量过少，只分配给一个客户端")
                eligible_clients_indices = [rng.choice(eligible_clients_indices)]
            else:
                # 随机选择部分客户端
                eligible_clients_indices = rng.choice(
                    eligible_clients_indices,
                    size=max_possible_clients,
                    replace=False
                ).tolist()
            num_eligible_clients = len(eligible_clients_indices)


        random_quantities = _get_random_partitions_with_min(
            num_data_points, num_eligible_clients, fixed_min_size, rng
        )


        # 执行数据分配
        current_pos = 0
        for i, client_idx in enumerate(eligible_clients_indices):
            quantity = random_quantities[i]
            if current_pos + quantity > len(data_for_type):
                # 防止数组越界
                quantity = len(data_for_type) - current_pos
            if quantity > 0:
                data_slice = data_for_type[current_pos: current_pos + quantity]
                if battery_type in final_clients_data[client_idx]:
                    final_clients_data[client_idx][battery_type].extend(data_slice.tolist())
                else:
                    final_clients_data[client_idx][battery_type] = data_slice.tolist()
                final_nums[client_idx].extend(data_slice.tolist())
                current_pos += quantity

        total_allocated += current_pos

        # 检查是否有剩余数据未分配
        if current_pos < len(data_for_type):
            print(f"警告: 类型 {battery_type} 有 {len(data_for_type) - current_pos} 个数据未分配")
            # 将剩余数据随机分配给有资格的客户端
            remaining_data = data_for_type[current_pos:]
            for j, data_point in enumerate(remaining_data):
                client_idx = eligible_clients_indices[j % len(eligible_clients_indices)]
                if battery_type in final_clients_data[client_idx]:
                    final_clients_data[client_idx][battery_type].append(data_point)
                else:
                    final_clients_data[client_idx][battery_type] = [data_point]
                final_nums[client_idx].append(data_point)
                total_allocated += 1

    # 最终验证和调试信息
    sum_length = sum([len(l) for l in final_nums])
    total_data_points = len(df_data)

    with open("client_model/" + str(random_seed) + "/logging.txt", 'a', encoding='utf-8') as f:
        print(f"分配完成，共{sum_length}/{total_data_points}个数据", file=f)
        print("分配统计:", file=f)
        for type_name, stats in allocation_stats.items():
            print(f"类型 {type_name}: {stats['data_points']} 个数据 -> {stats['eligible_clients']} 个客户端", file=f)

        # 检查每个客户端的数据量
        for i, client_data in enumerate(final_nums):
            print(f"客户端 {i}: {len(client_data)} 个数据", file=f)

    if sum_length != total_data_points:
        print(f"错误: 分配的数据数量 ({sum_length}) 与总数据数量 ({total_data_points}) 不一致！")
        print("调试信息:")
        print(f"总数据点数: {total_data_points}")
        print(f"已分配数据点数: {sum_length}")
        print(f"差异: {total_data_points - sum_length}")

        # 检查哪些数据丢失了
        original_data = set(df_data["nums"].values)
        allocated_data = set()
        for client_data in final_nums:
            allocated_data.update(client_data)

        missing_data = original_data - allocated_data
        if missing_data:
            print(f"丢失的数据: {len(missing_data)} 个")

        raise ValueError("分配的数据数量与总数据数量不一致！")

    return final_clients_data, final_nums

def plot_train_dataset(random_seed=42):
    name = "data//train" + str(random_seed) + ".csv"
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
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))

    # 第一个子图：数据数量
    ax[0].bar(conditions, data_nums, color='skyblue', alpha=0.7)
    ax[0].set_ylim(0, max(data_nums) * 1.2)
    ax[0].set_title('Data Number by Condition')
    ax[0].set_xlabel('Condition')
    ax[0].set_xticklabels(conditions, rotation=45, ha='right')
    ax[0].set_ylabel('Data Number')
    # 在柱子上添加数值标签
    for i, v in enumerate(data_nums):
        ax[0].text(i, v, str(v), ha='center', va='bottom')

    # 第二个子图：电池数量
    ax[1].bar(conditions, battery_nums, color='lightcoral', alpha=0.7)
    ax[1].set_ylim(0, max(battery_nums) * 1.2)
    ax[1].set_title('Battery Number by Condition')
    ax[1].set_xlabel('Condition')
    ax[1].set_xticklabels(conditions, rotation=45, ha='right')
    ax[1].set_ylabel('Battery Number')
    # 在柱子上添加数值标签
    for i, v in enumerate(battery_nums):
        ax[1].text(i, v, str(v), ha='center', va='bottom')

    plt.tight_layout()
    plt.savefig("client_model//" + str(random_seed) + "//train_data.png")


def generate_client_model(random_seed = 42,num_client=6,mini_type=2,max_type=5,model_type="MLP",hidden_dim=10):
    random_seed = random_seed
    name = "data//train"+str(random_seed)+".csv"
    df = pd.read_csv(name, sep="\t", header=0)

    condition = df["condition"].unique()

    dict = {}

    for c in condition:
        df_c = df[df["condition"] == c]
        ID = df_c["No."].unique()
        dict[c] = sorted(list(ID))#dict keys是电池类型 values是对应电池ID


    data_list = []
    for battery_type, id_list in dict.items():
        for battery_id in id_list:
            data_list.append((battery_type, battery_id))

    # 创建 DataFrame
    df_data = pd.DataFrame(data_list, columns=['type', 'id'])
    df_data["nums"] = list(range(len(df_data)))

    num_client = num_client
    clients, nums_list = partition_random_quantities(df_data,
                                                     num_clients=num_client,
                                                     random_seed=random_seed,
                                                     mini_type=mini_type,
                                                     max_type=max_type
    )

    for i, client_data in enumerate(clients):
        client = client_model(i,random_seed=random_seed)

        index = df_data[df_data["nums"].isin(nums_list[i])][["type","id"]]
        data_dict = {j:index[index["type"] == j]["id"].values for j in index["type"].unique()}
        client.set_dataset(data_dict)


        client_df_filtered = pd.merge(df, index, left_on=['condition', 'No.'], right_on=['type', 'id'])
        client.set_dataframe(client_df_filtered)
        client.check_up()



        client.set_model(model=model_type)
        client.set_encoder(hidden_dim=hidden_dim)

        import os.path
        if not os.path.exists(f"client_model/{random_seed}/"):
            os.makedirs(f"client_model/{random_seed}/")
        client.write_to_csv(f"client_model/{random_seed}/")
        import pickle
        with open(f"client_model/{random_seed}/client_{i}.pkl", "wb") as f:
            pickle.dump(client, f)



