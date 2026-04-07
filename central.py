import os.path
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

if not os.path.exists("centralized_model"):
    os.makedirs("centralized_model/")

train_accuracy = []
test_accuracy = []
rd_list = []

for random_seed in range(100):
    train_data = pd.read_csv("data/train" + str(random_seed) + ".csv", sep="\t")
    test_data = pd.read_csv("data/test" + str(random_seed) + ".csv", sep="\t")

    lda = LinearDiscriminantAnalysis\
        (n_components=len(train_data["condition"].unique()) - 1)

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

    inverse_mapping_dict = {v: k for k, v in mapping_dict.items()}

    X_train = train_data.loc[:, "U1":"U41"]
    y_train = train_data["condition"].map(mapping_dict)

    lda = lda.fit(X_train,y_train)
    X_train_kernel_pca = lda.transform(X_train)


    clf = MLPClassifier(hidden_layer_sizes=[100,100,100,100],
                            max_iter=300,
                            random_state = random_seed,
                            learning_rate_init=0.01)

    clf.fit(X_train_kernel_pca, y_train)

    print(f"Client {random_seed} model training accuracy: {clf.score(X_train_kernel_pca, y_train)}")
    print(f"Client {random_seed} model setup complete")

    X_test = test_data.loc[:, "U1":"U41"]
    y_test = test_data["condition"].map(mapping_dict)
    X_test_kernel_pca = lda.transform(X_test)

    result = clf.predict(X_test_kernel_pca)
    label = [inverse_mapping_dict[i] for i in result]
    true = test_data["condition"]

    from exp.evaluation import evalute_accuracy_for_central
    evalute_accuracy_for_central(result, true, label, random_seed, "Central Model")

    print(f"Client {random_seed} model test accuracy: {clf.score(X_test_kernel_pca, y_test)}")

    import pickle
    pickle.dump(clf, open(f"centralized_model/central_model_{random_seed}.pkl", "wb"))
    train_accuracy.append(clf.score(X_train_kernel_pca, y_train))
    test_accuracy.append(clf.score(X_test_kernel_pca, y_test))
    rd_list.append(random_seed)
dataframe = pd.DataFrame({"random_seed": rd_list,
                          "train_accuracy": train_accuracy,
                          "test_accuracy": test_accuracy})
dataframe.to_csv("centralized_model/centralized_model_accuracy.csv", index=False)


