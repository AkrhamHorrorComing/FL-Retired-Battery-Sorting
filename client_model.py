import os.path
import pandas as pd
import numpy as np
import random
from matplotlib import pyplot as plt

class client_model():
    def __init__(self, client_id, random_seed=42):
        self.client_id = client_id
        self.model = None
        self.random_seed = random_seed

        # Describes which batteries are included
        self.client_data = None

        # Training data
        self.dataset = None
        self.validation_data = None
        self.dataset_num_list = np.zeros([1,8])

        self.size = None
        self.type = None
        self.id = None

        self.PCA = None
        self.model = None

        self.encoder = None
        self.scaler = None
        self.encoder_PCA = None


    def encode(self, label):
        mapping_dict = {
            '10Ah_LMO': 0,
            '15Ah_NMC': 1,
            '21Ah_NMC': 2,
            '24Ah_LMO': 3,
            '25Ah_LMO': 4,
            '26Ah_LMO': 5,
            '35Ah_LFP': 6,
            '68Ah_LFP': 7
        }

        if isinstance(list(label), list):
            return [mapping_dict[l] for l in label]  # Multi-element list
        else:
            return mapping_dict[label]  # Single value (non-list)

    def decode(self, label):
        mapping_dict = {
            0: '10Ah_LMO',
            1: '15Ah_NMC',
            2: '21Ah_NMC',
            3: '24Ah_LMO',
            4: '25Ah_LMO',
            5: '26Ah_LMO',
            6: '35Ah_LFP',
            7: '68Ah_LFP'
        }

        if isinstance(label, list):
            if len(label) == 1:
                return mapping_dict[label[0]]  # Single-element list
            else:
                return [mapping_dict[l] for l in label]  # Multi-element list
        else:
            return mapping_dict[label]  # Single value (non-list)

    def set_dataset(self, dataset):
        self.client_data = dataset
        self.type = list(self.client_data.keys())
        self.size = sum([len(id_array) for id_array in self.client_data.values()])

        print(f"Client {self.client_id} dataset setup complete, containing {self.type}")

    def write_to_csv(self,path):

        data = self.dataset
        type_list = data["condition"].unique()

        # Collect data using a list
        data_list = []
        for type_ in type_list:
            data_type = data[data["condition"] == type_]
            battery_num = data_type["No."].unique()

            data_list.append({
                "type": type_,
                "battery_num": len(battery_num),
                "data_num": len(data_type),
                "battery": sorted(battery_num.tolist())  # Convert numpy array to list
            })

        # Create DataFrame at once
        dataframe = pd.DataFrame(data_list)
        dataframe.to_csv(path+f"client_{self.client_id}.csv", index=False,sep = "\t")

    def set_dataframe(self,dataframe):
        # This is the training data
        self.dataset = dataframe
        self.dataset_num_list = np.zeros([1,8])
        for t in self.type:
            self.dataset_num_list[0,self.encode([t])] = \
                self.dataset[self.dataset["condition"] == t]['No.'].nunique()
        self.dataset_num_list = self.dataset_num_list / np.linalg.norm(self.dataset_num_list)
        print(f"Client {self.client_id} dataframe setup complete")

    def check_up(self):
        type = self.dataset["condition"].unique()
        if set(type) != set(self.type):
            print("Category mismatch!")
        for t in type:
            if set(self.dataset[self.dataset["condition"] == t]["No."].values) != set(self.client_data[t]):
                print(f"Data mismatch for type {t}!")
        print("Check complete")

        import math

        # Get unique condition types
        types = self.dataset["condition"].unique()
        n_types = len(types)

        # Calculate appropriate subplot layout (including All subplot)
        total_plots = n_types + 1  # Total number of subplots
        n_cols = min(3, total_plots)
        n_rows = math.ceil(total_plots / n_cols)

        # Create subplots
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))

        # Ensure axes is in 2D array form
        if n_rows == 1 and n_cols == 1:
            axes = [[axes]]
        elif n_rows == 1:
            axes = [axes]
        elif n_cols == 1:
            axes = [[ax] for ax in axes]
        else:
            # Convert axes to list form for easier indexing
            axes = axes.tolist() if hasattr(axes, 'tolist') else axes

        data = self.dataset

        # Draw subplots for each type
        for idx, condition_type in enumerate(types):
            row = idx // n_cols
            col = idx % n_cols

            data_type = data[data["condition"] == condition_type]
            SOC = data_type["SOCR"]

            axes[row][col].hist(SOC, bins=20, alpha=0.7, color=f'C{idx}', edgecolor=None)
            axes[row][col].set_title(f'{condition_type}')
            axes[row][col].set_xlim([0, 1])
            axes[row][col].set_xlabel('SOC')
            axes[row][col].set_ylabel('Frequency')
            axes[row][col].grid(True, alpha=0.3)

        # Fix: explicitly calculate All subplot position
        all_idx = n_types  # Index of the All subplot
        row = all_idx // n_cols
        col = all_idx % n_cols

        SOC_all = data["SOCR"]
        axes[row][col].hist(SOC_all, bins=20, alpha=0.7, color='purple', edgecolor=None)  # Use a different color
        axes[row][col].set_title('All')
        axes[row][col].set_xlim([0, 1])
        axes[row][col].set_xlabel('SOC')
        axes[row][col].set_ylabel('Frequency')
        axes[row][col].grid(True, alpha=0.3)

        # Hide extra subplots
        for idx in range(total_plots, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row][col].set_visible(False)

        # Adjust layout
        plt.tight_layout()
        plt.savefig(f"client_model/{self.random_seed}/client_{self.client_id}_SOC.png", dpi=600, bbox_inches='tight')
        plt.close()

    def data_augmentation_1(self):
        data = self.dataset
        type_list = data["condition"].unique()

        # Collect data using a list
        data_list = []
        for type_ in type_list:
            data_type = data[data["condition"] == type_]
            battery_num = data_type["No."].unique()

            data_list.append({
                "type": type_,
                "battery_num": len(battery_num),
                "data_num": len(data_type),
                "battery": sorted(battery_num.tolist())  # Convert numpy array to list
            })
        # Create DataFrame at once
        dataframe = pd.DataFrame(data_list)

        data_series = dataframe.loc[:, ['type', 'data_num']]

        max_num = 1600

        for i in data_series.index:
            type_ = data_series.loc[i,"type"]
            data_num = data_series.loc[i,"data_num"]


            if data_num < max_num:
                print(f"Type {type_} has insufficient data ({data_num} < {max_num}), augmentation needed")
                # Calculate the number of samples to add
                num_to_add = int(max_num - data_num)

                origin_data =  data[data["condition"] == type_]
                # Randomly select batteries for duplication
                selected_index = random.choices(origin_data.index, k=num_to_add)
                # Duplicate data
                append_data = data.loc[selected_index]
                data = pd.concat([data, append_data], ignore_index=True)
        return data

    def data_augmentation_2(self):
        data = self.dataset
        type_list = data["condition"].unique()

        # Collect data using a list
        data_list = []
        for type_ in type_list:
            data_type = data[data["condition"] == type_]
            battery_num = data_type["No."].unique()

            data_list.append({
                "type": type_,
                "battery_num": len(battery_num),
                "data_num": len(data_type),
                "battery": sorted(battery_num.tolist())  # Convert numpy array to list
            })
        # Create DataFrame at once
        dataframe = pd.DataFrame(data_list)

        data_series = dataframe.loc[:, ['type', 'data_num']]

        max_num = data_series["data_num"].max()

        for i in data_series.index:
            type_ = data_series.loc[i,"type"]
            data_num = data_series.loc[i,"data_num"]


            if data_num < max_num:
                # Calculate the number of samples to add
                num_to_add = int(max_num - data_num)

                origin_data =  data[data["condition"] == type_]
                features = origin_data.loc[:, "U1":"U41"]
                mean_vec = features.mean().values
                cov_matrix = features.cov().values
                # Randomly select batteries for duplication
                selected_index = random.choices(origin_data.index, k=num_to_add)
                # Duplicate data
                append_data = data.loc[selected_index]

                correlated_noise = np.random.multivariate_normal(np.zeros(features.shape[1]), cov_matrix,
                                                                 size=num_to_add)
                # Add noise
                append_data.loc[:, "U1":"U41"] += correlated_noise


                data = pd.concat([data, append_data], ignore_index=True)
        return data



    def set_model(self, model=None):
        from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
        from sklearn.gaussian_process import GaussianProcessClassifier
        from sklearn.gaussian_process.kernels import RBF,RationalQuadratic
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.neural_network import MLPClassifier
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.decomposition import PCA, KernelPCA

        # pca = PCA(n_components=2)
        # kernel_pca = KernelPCA(
        #     n_components=64, kernel="rbf", gamma=10, fit_inverse_transform=True, alpha=0.1
        # )
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        lda = LinearDiscriminantAnalysis(n_components=len(self.type)-1)

        mapping_dict = {
            '10Ah_LMO': 0,
            '15Ah_NMC': 1,
            '21Ah_NMC': 2,
            '24Ah_LMO': 3,
            '25Ah_LMO': 4,
            '26Ah_LMO': 5,
            '35Ah_LFP': 6,
            '68Ah_LFP': 7
        }
        # train = self.dataset
        # train = self.data_augmentation(noise_percentage=1)
        train = self.data_augmentation_2()

        X_train = train.loc[:, "U1":"U41"]
        y_train = train["condition"].map(mapping_dict)

        lda.fit(X_train,y_train)
        self.PCA = lda
        X_train_kernel_pca = self.PCA.transform(X_train)

        if model == "MLP":
            clf = MLPClassifier(hidden_layer_sizes=[100,100,100,100],
                            max_iter=300,
                            random_state=self.random_seed,
                            learning_rate_init=0.01)
        elif model == 'MLP_2':
            clf = MLPClassifier(hidden_layer_sizes=[50, 50],
                        random_state=42,
                        learning_rate_init=0.01,
                        learning_rate='constant',
                        alpha = 0.0001,
                        activation= 'relu',
                        )
        elif model == "RF":
            clf = RandomForestClassifier(max_depth=None, random_state=42)
        elif model == "KNN":
            clf = KNeighborsClassifier(n_neighbors=10, weights='distance')
        elif model == "GP":
            clf = GaussianProcessClassifier(kernel=RationalQuadratic(1.0),
                                            random_state=42,
                                            n_restarts_optimizer=50)
        elif model == "DT":
            clf = DecisionTreeClassifier(max_depth=100, random_state=42)
        else:
            print("Model not specified")
            raise ValueError

        clf.fit(X_train_kernel_pca, y_train)
        self.model = clf

        with open(f"client_model/{self.random_seed}/logging.txt", 'a', encoding='utf-8') as f:
            print(f"Client {self.client_id} model training accuracy: {clf.score(X_train_kernel_pca, y_train)}",file=f)
        print(f"Client {self.client_id} model setup complete")


    def valid_score(self,x_valid,y_valid,write_to_file = 1):
        index = y_valid.isin(self.type)
        y_valid = self.encode(y_valid[index])
        x_valid = x_valid[index]
        model = self.model
        x_valid_kernel_pca = self.PCA.transform(x_valid)
        if write_to_file:
            with open(f"client_model/{self.random_seed}/logging.txt", 'a', encoding='utf-8') as f:
                print(f"Client {self.client_id} model validation accuracy: {model.score(x_valid_kernel_pca, y_valid)}",file=f)
        return model.score(x_valid_kernel_pca, y_valid)

    def test_score(self,x_test,y_test,write_to_file = 1):
        index = y_test.isin(self.type)
        y_test = self.encode(y_test[index])
        x_test = x_test[index]
        model = self.model
        x_test_kernel_pca = self.PCA.transform(x_test)
        if write_to_file:
            with open(f"client_model/{self.random_seed}/logging.txt", 'a', encoding='utf-8') as f:
                print(f"Client {self.client_id} model test accuracy: {model.score(x_test_kernel_pca, y_test)}",file=f)
        return model.score(x_test_kernel_pca, y_test)

    def type_score(self,x_test,y_test,write_to_file = 1):
        # First create a filter mask
        mask = y_test.isin(self.type)
        # Filter data
        x_test_filtered = x_test[mask]
        y_test_filtered = y_test[mask]
        # Encode labels
        y_test_encoded = self.encode(list(y_test_filtered))
        # Transform features
        x_test_kernel_pca = self.PCA.transform(x_test_filtered)
        # Predict
        label_encoded = self.model.predict(x_test_kernel_pca)

        class_accuracies = {}


        unique_classes = np.unique(y_test_encoded)
        for class_idx in unique_classes:
            # Find sample indices for this class
            class_indices = np.where(y_test_encoded == class_idx)[0]
            if len(class_indices) == 0:
                continue
            # Calculate accuracy for this class
            class_correct = 0
            for idx in class_indices:
                if y_test_encoded[idx] == label_encoded[idx]:
                    class_correct += 1

            class_name = self.decode([class_idx])
            accuracy = class_correct / len(class_indices)
            class_accuracies[class_name] = accuracy
            # print(f"Class {class_name} Accuracy: {accuracy:.4f} ({class_correct}/{len(class_indices)})")
            if write_to_file:
                with open(f"client_model/{self.random_seed}/logging.txt", 'a', encoding='utf-8') as f:
                    print(f"Class {class_name} Accuracy: {accuracy:.4f} ({class_correct}/{len(class_indices)})",file=f)
        return class_accuracies


    def plot_type_accuracy(self, client_accuracies):
        from matplotlib import pyplot as plt
        import numpy as np

        all_classes = set()
        for client_data in client_accuracies.values():
            all_classes.update(client_data.keys())
        all_classes = sorted(all_classes)

        # Create 3x3 subplot layout
        fig, axes = plt.subplots(3, 3, figsize=(15, 12),sharey=True)
        axes = axes.flatten()

        # Color settings
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

        # Create a subplot for each class (horizontal bar chart)
        for class_idx, class_name in enumerate(all_classes):
            if class_idx >= len(axes):
                break

            ax = axes[class_idx]

            # Prepare data
            client_names = []
            class_accuracies = []

            for client_name, acc_dict in client_accuracies.items():
                accuracy = acc_dict.get(class_name, 0)
                client_names.append(client_name)
                class_accuracies.append(accuracy)

            # Draw horizontal bar chart
            y_pos = np.arange(len(client_names))
            bars = ax.barh(y_pos, class_accuracies, color=colors[:len(client_names)], alpha=0.8)

            # Configure subplot
            ax.set_title(f'{class_name}', fontweight='bold')
            ax.set_xlabel('Accuracy')
            ax.set_xlim(0.6, 1.0)
            ticks = np.arange(0.6, 1.0 + 0.05, 0.1)
            ax.set_xticks(ticks)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(client_names)
            ax.grid(True, alpha=0.3)

            # Add value labels on bars
            for bar, acc in zip(bars, class_accuracies):
                width = bar.get_width()
                if width > 0:
                    ax.text(width-0.05, bar.get_y() + bar.get_height() / 2.,
                            f'{acc:.3f}', ha='left', va='center', fontsize=15,color='white')

        # Hide extra subplots
        for i in range(len(all_classes), len(axes)):
            axes[i].set_visible(False)

        plt.tight_layout()
        plt.savefig(f"client_model/{self.random_seed}/type_accuracy_horizontal.png", dpi=300, bbox_inches='tight')
        plt.close()
        # plt.show()


    def predict(self,x_test,if_decode=0):
        model = self.model
        x_test_kernel_pca = self.PCA.transform(x_test)

        predicted_labels = model.predict(x_test_kernel_pca)
        predicted_probs = model.predict_proba(x_test_kernel_pca)

        if if_decode:
            return self.decode(list(predicted_labels)),predicted_probs
        else:
            prob_matrix = np.zeros((x_test.shape[0],8),dtype=float)
            prob_matrix[:,model.classes_] = predicted_probs
            return predicted_labels,prob_matrix

    def set_encoder(self, hidden_dim=3):
        from exp.encoder import Autoencoder
        import torch
        from torch.utils.data import TensorDataset, DataLoader, random_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.decomposition import KernelPCA

        # Data augmentation
        augmented_data = self.data_augmentation_1()

        kpca = KernelPCA(
            n_components=20,
            kernel="rbf",
            gamma=None,  # gamma=None lets sklearn automatically choose a good value
            fit_inverse_transform=True,  # (optional, but recommended)
            random_state=self.random_seed
        )

        X_train_aug = augmented_data.loc[:, "U1":"U41"]
        self.encoder_PCA = kpca.fit(X_train_aug)

        type_list = augmented_data["condition"].unique()

        encoder_dict = {}

        scaler_dict = {}

        for type_ in type_list:
            type_data = augmented_data[augmented_data["condition"] == type_]
            data_values = type_data.loc[:, "U1":"U41"].values


            pca_data = self.encoder_PCA.transform(data_values)

            scaler = StandardScaler()
            scaler = scaler.fit(pca_data)
            pca_data = scaler.transform(pca_data)
            scaler_dict[type_] = scaler

            # data_values = (data_values-2.2)/4.4

            augmented_tensor = torch.tensor(
                pca_data,
                dtype=torch.float32
            )

            dataset = TensorDataset(augmented_tensor)

        # Split into training and validation sets
        #     train_size = int(0.8 * len(dataset))
        #     val_size = len(dataset) - train_size
        #     train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

            train_dataset = dataset

            train_loader = DataLoader(
                train_dataset,
                batch_size=64,
                shuffle=True,
                drop_last=False
            )
            # val_loader = DataLoader(
            #     val_dataset,
            #     batch_size=64,
            #     shuffle=False
            # )

            # Initialize model and optimizer
            encoder = Autoencoder(
                pca_data.shape[1],
                hidden_dim = hidden_dim
            )
            optimizer = torch.optim.AdamW(  # Using AdamW
                encoder.parameters(),
                lr=0.01,
                betas=(0.9, 0.999),
                weight_decay = 0
            )
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.8)
            criterion = torch.nn.L1Loss()

            # Training loop
            best_train_loss = float('inf')
            patience = 50
            patience_counter = 0
            train_losses = []

            for epoch in range(500):
                # Training phase
                encoder.train()
                train_loss = 0.0
                for batch in train_loader:
                    batch_data = batch[0]  # Extract data from TensorDataset
                    optimizer.zero_grad()
                    outputs = encoder(batch_data)
                    loss = criterion(outputs, batch_data)
                    loss.backward()
                    optimizer.step()
                    train_loss += loss.item()

                # Validation phase
                # encoder.eval()
                # val_loss = 0.0
                # with torch.no_grad():
                #     for batch in val_loader:
                #         batch_data = batch[0]
                #         outputs = encoder(batch_data)
                #         val_loss += criterion(outputs, batch_data).item()

                # Calculate average loss
                avg_train_loss = train_loss / len(train_loader)
                # avg_val_loss = val_loss / len(val_loader)
                train_losses.append(avg_train_loss)
                # val_losses.append(avg_val_loss)
                # Learning rate scheduling (called after each epoch)
                scheduler.step()

                # Early stopping mechanism
                if avg_train_loss < best_train_loss:
                    best_train_loss = avg_train_loss
                    torch.save(encoder.state_dict(), f"client_model/{self.random_seed}/best_encoder.pth")
                # else:
                #     patience_counter += 1

                # Print training information
                # current_lr = optimizer.param_groups[0]['lr']
                # print(f"Epoch {epoch + 1}, Train Loss: {avg_train_loss:.4f}, LR: {current_lr:.6f},  "
                #       # f"Val Loss: {avg_val_loss:.4f},"
                #       f"Patience: {patience_counter}/{patience}")


                # Early stopping check
                # if patience_counter >= patience:
                #     print(f"Early stopping at epoch {epoch + 1}")
                #     break

            with open(f"client_model/{self.random_seed}/logging.txt", "a") as f:
                f.write(f"\n{self.client_id} Autoencoder {type_} train loss {train_losses[-1]:.4f}")

            # Load best model
            encoder.load_state_dict(
                torch.load(f"client_model/{self.random_seed}/best_encoder.pth")
            )
            encoder_dict[type_] = encoder
        self.encoder = encoder_dict
        self.scaler = scaler_dict


    def calculate_pca_distance(self, X_test_values,
                               aggregation_method='min',
                               distance = 'manhattan'):

        features_pca = self.PCA.transform(X_test_values)

        origin_pca_ = self.PCA.transform(self.dataset.loc[:, "U1":"U41"])

        from scipy.spatial.distance import cdist

        if distance == 'euclidean':
            distance_matrix = cdist(features_pca, origin_pca_, 'euclidean')
        # elif distance == 'manhattan':
        #     mean = np.mean(origin_pca_, axis=0)
        #     cov = np.cov(origin_pca_, rowvar=False)
        #     inv_cov = np.linalg.inv(cov)
        #
        #     distance_matrix = cdist(features_pca, [mean],
        #                   metric='mahalanobis', VI=inv_cov)

        # Step D: Aggregate distance matrix to get final scores
        if aggregation_method == 'min':
            # Take the minimum value for each row (each test sample)
            scores = np.min(distance_matrix, axis=1)
        elif aggregation_method == 'mean':
            # Take the mean value for each row (each test sample)
            scores = np.mean(distance_matrix, axis=1)
        else:
            raise ValueError("aggregation_method must be 'min' or 'mean'")

        return scores
