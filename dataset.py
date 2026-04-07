# Create client models, each client model has a variable number of battery types, each type has at least minisize=5 batteries.
import os.path

import pandas as pd
import numpy as np
import random
from client_model import client_model

def _get_random_partitions_with_min(total_sum, num_bins, min_size, rng):

    if total_sum < num_bins * min_size:
        raise ValueError(
            f"Total sum {total_sum} is insufficient for {num_bins} partitions with minimum {min_size} each. "
            f"At least {num_bins * min_size} required."
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
#     Follow the logic of 'select types first, then data', distributing the dataset completely with random quantities.
#     """
#     print("--- Start data distribution by type with 'random quantities' ---")
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
#         print("--- Distribution plan (types each client will receive) ---",file=f)
#         for i, types in enumerate(client_type_map):
#             print(f"Client {i}: will receive {len(types)} types -> {sorted(types)}",file=f)
#
#     # Step 2: Distribute data according to plan
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
#         print(f"Distribution complete, {sum_length}/{len(df_data)} data points",file=f)
#     if sum_length != len(df_data):
#         raise ValueError("Distributed data count does not match total data count!")
#     return final_clients_data, final_nums
def partition_random_quantities(df_data, num_clients=5, mini_type=2, max_type=5, random_seed=42):
    """
    Improved version: ensures all data is distributed, fixed minimum allocation is 5
    """
    print("--- Start data distribution by type ---")
    rng = np.random.default_rng(random_seed)

    all_types = df_data["type"].unique()

    # Step 1: Assign types to each client
    client_type_map = []
    for _ in range(num_clients):
        num_types = rng.integers(mini_type, max_type + 1)
        types_for_client = rng.choice(all_types, size=num_types, replace=False)
        client_type_map.append(types_for_client.tolist())

    if not os.path.exists("client_model/" + str(random_seed)):
        os.makedirs("client_model/" + str(random_seed))

    with open("client_model/" + str(random_seed) + "/logging.txt", 'a', encoding='utf-8') as f:
        print("--- Distribution plan (types each client will receive) ---", file=f)
        for i, types in enumerate(client_type_map):
            print(f"Client {i}: will receive {len(types)} types -> {sorted(types)}", file=f)

    # Step 2: Distribute data according to plan
    final_clients_data = [{} for _ in range(num_clients)]
    final_nums = [[] for _ in range(num_clients)]

    # Record distribution statistics
    allocation_stats = {}
    total_allocated = 0

    for battery_type in all_types:
        eligible_clients_indices = [
            i for i, types in enumerate(client_type_map) if battery_type in types
        ]

        if not eligible_clients_indices:
            print(f"Warning: Type {battery_type} was not selected by any client, will be randomly assigned to one client")
            # Randomly select a client to receive this type's data
            eligible_clients_indices = [rng.integers(0, num_clients)]

        data_for_type = df_data[df_data["type"] == battery_type]["nums"].values
        # Battery IDs for the corresponding type
        rng.shuffle(data_for_type)

        num_data_points = len(data_for_type)
        num_eligible_clients = len(eligible_clients_indices)

        # Record statistics
        allocation_stats[battery_type] = {
            'data_points': num_data_points,
            'eligible_clients': num_eligible_clients
        }

        # Fixed minimum allocation of 5
        fixed_min_size = 5

        # Check if data quantity is sufficient
        if num_data_points < num_eligible_clients * fixed_min_size:
            print(f"Warning: Type {battery_type} has insufficient data ({num_data_points} < {num_eligible_clients * fixed_min_size})")
            # Reduce eligible client count to meet minimum allocation requirements
            max_possible_clients = num_data_points // fixed_min_size
            if max_possible_clients == 0:
                # If even one client's minimum requirement cannot be met, select the client with the most data
                print(f"Type {battery_type} has too little data, only assigned to one client")
                eligible_clients_indices = [rng.choice(eligible_clients_indices)]
            else:
                # Randomly select some clients
                eligible_clients_indices = rng.choice(
                    eligible_clients_indices,
                    size=max_possible_clients,
                    replace=False
                ).tolist()
            num_eligible_clients = len(eligible_clients_indices)


        random_quantities = _get_random_partitions_with_min(
            num_data_points, num_eligible_clients, fixed_min_size, rng
        )


        # Execute data distribution
        current_pos = 0
        for i, client_idx in enumerate(eligible_clients_indices):
            quantity = random_quantities[i]
            if current_pos + quantity > len(data_for_type):
                # Prevent array out-of-bounds
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

        # Check if there is remaining unassigned data
        if current_pos < len(data_for_type):
            print(f"Warning: Type {battery_type} has {len(data_for_type) - current_pos} unassigned data points")
            # Distribute remaining data to eligible clients randomly
            remaining_data = data_for_type[current_pos:]
            for j, data_point in enumerate(remaining_data):
                client_idx = eligible_clients_indices[j % len(eligible_clients_indices)]
                if battery_type in final_clients_data[client_idx]:
                    final_clients_data[client_idx][battery_type].append(data_point)
                else:
                    final_clients_data[client_idx][battery_type] = [data_point]
                final_nums[client_idx].append(data_point)
                total_allocated += 1

    # Final validation and debug information
    sum_length = sum([len(l) for l in final_nums])
    total_data_points = len(df_data)

    with open("client_model/" + str(random_seed) + "/logging.txt", 'a', encoding='utf-8') as f:
        print(f"Distribution complete, {sum_length}/{total_data_points} data points", file=f)
        print("Distribution statistics:", file=f)
        for type_name, stats in allocation_stats.items():
            print(f"Type {type_name}: {stats['data_points']} data points -> {stats['eligible_clients']} clients", file=f)

        # Check each client's data quantity
        for i, client_data in enumerate(final_nums):
            print(f"Client {i}: {len(client_data)} data points", file=f)

    if sum_length != total_data_points:
        print(f"Error: Distributed data count ({sum_length}) does not match total ({total_data_points})!")
        print("Debug information:")
        print(f"Total data points: {total_data_points}")
        print(f"Allocated data points: {sum_length}")
        print(f"Difference: {total_data_points - sum_length}")

        # Check which data is missing
        original_data = set(df_data["nums"].values)
        allocated_data = set()
        for client_data in final_nums:
            allocated_data.update(client_data)

        missing_data = original_data - allocated_data
        if missing_data:
            print(f"Missing data: {len(missing_data)} points")

        raise ValueError("Distributed data count does not match total data count!")

    return final_clients_data, final_nums

def plot_train_dataset(random_seed=42):
    name = "data//train" + str(random_seed) + ".csv"
    df = pd.read_csv(name, sep="\t", header=0)
    condition = df["condition"].unique()
    import matplotlib.pyplot as plt

    # Prepare data
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

    # Draw bar chart
    fig, ax = plt.subplots(2, 1, figsize=(10, 12))

    # First subplot: data count
    ax[0].bar(conditions, data_nums, color='skyblue', alpha=0.7)
    ax[0].set_ylim(0, max(data_nums) * 1.2)
    ax[0].set_title('Data Number by Condition')
    ax[0].set_xlabel('Condition')
    ax[0].set_xticklabels(conditions, rotation=45, ha='right')
    ax[0].set_ylabel('Data Number')
    # Add value labels on bars
    for i, v in enumerate(data_nums):
        ax[0].text(i, v, str(v), ha='center', va='bottom')

    # Second subplot: battery count
    ax[1].bar(conditions, battery_nums, color='lightcoral', alpha=0.7)
    ax[1].set_ylim(0, max(battery_nums) * 1.2)
    ax[1].set_title('Battery Number by Condition')
    ax[1].set_xlabel('Condition')
    ax[1].set_xticklabels(conditions, rotation=45, ha='right')
    ax[1].set_ylabel('Battery Number')
    # Add value labels on bars
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
        dict[c] = sorted(list(ID))  # dict keys are battery types, values are corresponding battery IDs


    data_list = []
    for battery_type, id_list in dict.items():
        for battery_id in id_list:
            data_list.append((battery_type, battery_id))

    # Create DataFrame
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
