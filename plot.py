import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
# Draw pie charts for each client's data partition
plt.rcParams['font.sans-serif'] = ['Yu Gothic', 'Meiryo', 'Hiragino Sans', 'Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False  # Fix minus sign display issue
plt.rcParams['font.size'] = 20
plt.rcParams['font.family'] = 'Times New Roman'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pickle
import itertools  # Import itertools; not directly used here but sometimes needed in client processing


def plot_stacked_bar_distribution(num_client=6, random_seed=42):
    # --- 1. Define global color mapping ---
    # All possible battery types (based on encode/decode in client_model)
    ALL_BATTERY_TYPES = [
        '10Ah_LMO', '15Ah_NMC', '21Ah_NMC', '24Ah_LMO',
        '25Ah_LMO', '26Ah_LMO', '35Ah_LFP', '68Ah_LFP'
    ]

    color_palette = plt.cm.get_cmap('tab10', len(ALL_BATTERY_TYPES))
    color_map = {battery_type: color_palette(i) for i, battery_type in enumerate(ALL_BATTERY_TYPES)}

    # --- 2. Collect battery distribution data for all clients ---
    all_client_battery_data = []  # Store each client's battery distribution Series
    client_labels = []  # Store client IDs

    for i in range(num_client):
        client_model_path = f"client_model/{str(random_seed)}/client_{str(i)}.pkl"

        # Ensure file exists
        if not os.path.exists(client_model_path):
            print(f"Warning: Client model {client_model_path} not found. Skipping client {i}.")
            continue

        client_model = pickle.load(open(client_model_path, 'rb'))
        data = client_model.dataset

        battery_counts = data.groupby("condition")["No."].nunique().reindex(ALL_BATTERY_TYPES, fill_value=0)

        all_client_battery_data.append(battery_counts)
        client_labels.append(f'Client {i}')

    # If no data collected, exit early
    if not all_client_battery_data:
        print("No client data found to plot.")
        return

    stacked_df = pd.DataFrame(all_client_battery_data).T
    stacked_df.columns = [f'Client {i}' for i in range(len(all_client_battery_data))]

    # --- 3. Plotting ---
    matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # Ensure font support
    matplotlib.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(figsize=(12, 8))  # Adjust figure size

    stacked_df.loc[ALL_BATTERY_TYPES].T.plot(
        kind='bar',
        stacked=True,
        ax=ax,
        color=[color_map[bt] for bt in ALL_BATTERY_TYPES],  # Use global color mapping
        edgecolor='black'  # Add border to each stacked section for readability
    )

    # Improve chart
    ax.set_xlabel('Client ID')
    ax.set_ylabel('Number of Batteries')
    ax.tick_params(axis='x', rotation=45)  # Rotate X-axis labels to prevent overlap
    ax.tick_params(axis='y')
    ax.set_ylim(0, stacked_df.sum(axis=0).max() * 1.2)  # Dynamically set y-axis upper limit

    # Add legend
    # Use ALL_BATTERY_TYPES to ensure legend order matches stacking order
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="Battery Types", bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add total count label for each bar (optional, increases information)
    for container in ax.containers:
        # Add labels at the top of each stacked bar
        for rect in container:
            height = rect.get_height()
            if height > 0:  # Only label non-zero heights
                ax.text(rect.get_x() + rect.get_width() / 2,
                        rect.get_y() + height / 2,  # Text position in the middle of stacked block
                        f'{int(height)}',
                        ha='center', va='center',
                        color='black',fontsize = 14)  # Text color and size

    # Add total count label for each bar
    total_counts = stacked_df.sum(axis=0)
    for i, total in enumerate(total_counts):
        ax.text(i, total + 5, f'Total: {int(total)}', ha='center', va='bottom')

    plt.tight_layout()  # Automatically adjust layout to prevent label overlap

    fig_dir = f"client_model/{str(random_seed)}/fig/"
    os.makedirs(fig_dir, exist_ok=True)
    plt.savefig(f"{fig_dir}client_battery_distribution_stacked_bar.png", dpi=600, bbox_inches='tight')
    plt.close()


def plot(num_client=6, random_seed=42):
    # --- Solution: create global color mapping ---
    # 1. Define all possible battery types (based on encode/decode in client_model)
    ALL_BATTERY_TYPES = [
        '10Ah_LMO', '15Ah_NMC', '21Ah_NMC', '24Ah_LMO',
        '25Ah_LMO', '26Ah_LMO', '35Ah_LFP', '68Ah_LFP'
    ]

    # 2. Create a global, fixed color mapping dictionary
    # Use a color palette with enough distinct colors (Set3 has 12)
    color_palette = plt.cm.Set3(np.linspace(0, 1, len(ALL_BATTERY_TYPES)))
    color_map = {battery_type: color_palette[i] for i, battery_type in enumerate(ALL_BATTERY_TYPES)}

    # --- Solution complete ---

    # 3. Move helper function definition outside the loop
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            if total == 0:
                return f'{pct:.1f}%\n(0)'  # Avoid division by zero
            val = int(round(pct * total / 100.0))
            return f'{pct:.1f}%\n({val})'

        return my_autopct

    # --- Loop starts ---
    for i in range(0, num_client):

        client_model_path = f"client_model/{str(random_seed)}/client_{str(i)}.pkl"


        client_model = pickle.load(open(client_model_path, 'rb'))
        data = client_model.dataset

        fig_dir = f"client_model/{str(random_seed)}/fig/"
        os.makedirs(fig_dir, exist_ok=True)

        # --- Modification: Sort and get colors from global mapping ---
        # Calculate data quantity distribution
        # .sort_index() ensures labels are sorted alphabetically
        data_counts = data["condition"].value_counts().sort_index()
        data_labels = data_counts.index
        data_values = data_counts.values
        # 5. Get this client's color list from global color_map
        data_plot_colors = [color_map[label] for label in data_labels]

        # Calculate battery count distribution
        battery_counts = data.groupby("condition")["No."].nunique().sort_index()
        battery_labels = battery_counts.index
        battery_values = battery_counts.values
        # 6. Get colors from global color_map again
        battery_plot_colors = [color_map[label] for label in battery_labels]
        # --- Modification complete ---

        # Create larger figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

        # Data quantity pie chart - with legend
        wedges1, texts1, autotexts1 = ax1.pie(
            data_values,
            labels=None,  # Remove direct labels
            autopct=make_autopct(data_values),
            startangle=90,
            colors=data_plot_colors,  # <-- 7. Use the fixed color list
            textprops={'fontsize': 16},
            pctdistance=0.85
        )

        for autotext in autotexts1:
            autotext.set_color('black')

        # Add legend
        ax1.legend(wedges1, [f'{label}: {value}' for label, value in zip(data_labels, data_values)],
                   title="Conditions", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

        ax1.set_title(f'Client {i} - Data Distribution\nTotal: {sum(data_values)} records', fontsize=12)
        ax1.axis('equal')

        # Battery count pie chart - with legend
        wedges2, texts2, autotexts2 = ax2.pie(
            battery_values,
            labels=None,  # Remove direct labels
            autopct=make_autopct(battery_values),
            startangle=90,
            colors=battery_plot_colors,  # <-- 8. Use the fixed color list
            textprops={'fontsize': 16},
            pctdistance=0.85
        )

        for autotext in autotexts2:
            autotext.set_color('black')

        # Add legend
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
        Build a complete, symmetric similarity matrix from the model pair lists and score list.
        """
        # Find all unique model IDs and sort them
        all_ids = sorted(list(set(i_list + j_list)))

        # Create an empty DataFrame as the matrix, using IDs as index and column names
        similarity_df = pd.DataFrame(index=all_ids, columns=all_ids, dtype=float)

        # Iterate through the lists and fill the matrix
        for id_i, id_j, score in zip(i_list, j_list, scores):
            # Because the matrix is symmetric, both positions must be filled
            similarity_df.loc[id_i, id_j] = score
            similarity_df.loc[id_j, id_i] = score

        # Fill the diagonal with 1 (a model's similarity with itself is 1)
        np.fill_diagonal(similarity_df.values, 1.0)

        return similarity_df

    # Call the function to create the matrix
    similarity_matrix = create_similarity_matrix(i_list, j_list, scores)
    with open (f"{path}\\similarity_matrix.txt", "w") as f:
        f.write(similarity_matrix.to_string())


    matplotlib.rcParams['font.sans-serif'] = ['SimHei']
    matplotlib.rcParams['axes.unicode_minus'] = False

    # --- Start plotting ---
    fig, ax = plt.subplots(figsize=(8, 6))  # Create a figure and an axes

    # cmap parameter is exactly the same as in seaborn
    im = ax.imshow(similarity_matrix.values, cmap="YlGnBu")

    # Create color bar
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Similarity", rotation=-90, va="bottom")


    ax.set_xticks(np.arange(len(similarity_matrix.columns)))
    ax.set_yticks(np.arange(len(similarity_matrix.index)))
    ax.set_xticklabels(similarity_matrix.columns)
    ax.set_yticklabels(similarity_matrix.index)

    # To prevent label overlap, rotate X-axis labels
    plt.setp(ax.get_xticklabels(), ha="right", rotation_mode="anchor")

    # --- Add value annotations (this is the most critical step) ---
    # Loop through all data points and add text at the corresponding positions
    for i in range(len(similarity_matrix.index)):
        for j in range(len(similarity_matrix.columns)):
            value = similarity_matrix.iloc[i, j]
            # Set text color: if the background color is too dark, use white text; otherwise use black
            # We take half of the color map range as the threshold
            threshold = im.norm.vmax / 2.
            text_color = "white" if value > threshold else "black"

            ax.text(j, i, f"{value:.2f}",
                    ha="center", va="center", color=text_color)

    plt.xlabel('Client ID')
    plt.ylabel('Client ID')

    # Save the image if needed
    plt.savefig(path+'/similarity_heatmap.png', dpi=600, bbox_inches='tight')
    plt.close()

def plot_dataset():
    name = "mat_data.csv"
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
    fig, ax = plt.subplots(2, 1, figsize=(8, 12),sharex=True)

    # First subplot: data count
    ax[0].bar(conditions, data_nums, color='skyblue', alpha=0.7)
    ax[0].set_ylim(0, max(data_nums) * 1.2)

    ax[0].set_ylabel('Data Number')
    # Add value labels on bars
    for i, v in enumerate(data_nums):
        ax[0].text(i, v, str(v), ha='center', va='bottom')

    # Second subplot: battery count
    ax[1].bar(conditions, battery_nums, color='lightcoral', alpha=0.7)
    ax[1].set_ylim(0, max(battery_nums) * 1.2)
    # ax[1].set_title('Battery Number by Condition')
    ax[1].set_xticklabels(conditions, rotation=45, ha='right')
    ax[1].set_ylabel('Battery Number')
    # Add value labels on bars
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

    # Draw bar chart
    fig, ax = plt.subplots(2, 1, figsize=(8, 10),sharex=False)

    # First subplot: data count
    bins = list(np.arange(0.3, 1, (1 - 0.3) / 14)) + [max(SOH)]
    ax[0].hist(SOH, bins=bins, color='skyblue', alpha=0.7)
    ax[0].set_ylim(0, 2500)
    ax[0].set_xlim(0.35, 1.0)
    ax[0].set_xticks([0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    ax[0].set_xticklabels(['0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])

    ax[0].set_ylabel('Data Number')
    ax[0].set_xlabel('SOH')



    # Second subplot: battery count
    ax[1].hist(SOC, bins=20, color='lightcoral', alpha=0.7)
    ax[1].set_ylim(0, max(ax[1].get_yticks()) * 1.2)

    ax[1].set_ylabel('Data Number')
    ax[1].set_xticks([0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    ax[1].set_xticklabels(['0.1', '0.2', '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9'])
    ax[1].set_xlabel('SOC')
    # Add value labels on bars

    plt.tight_layout()
    plt.savefig("dataset_SOX.png",dpi=600)

# for rd in range(100):
#     plot_stacked_bar_distribution(num_client=6, random_seed=rd)
plot_dataset_SOX()
