import numpy as np
import pandas as pd
import pickle

def aggregate_test(num_client,rd,test_data):
    with open(f"client_model/{rd}/logging.txt", 'a', encoding='utf-8') as f:
        print(f"*** Voting method test results ***", file=f)
    sum_probability = np.zeros((test_data.shape[0],test_data["condition"].nunique()),dtype=float)
    client_accuracies = {}
    X_test = test_data.loc[:, "U1":"U41"]
    Y_test = test_data["condition"]

    for i in range(0,num_client):
        #Estimate accuracy for each client model
        client_model = pickle.load(open("client_model/"+str(rd)+"/client_"+str(i)+".pkl",
                                          'rb'))

        client_model.test_score(X_test,Y_test)


        class_accuracies = client_model.type_score(X_test, Y_test)
        client_accuracies[f'Client_{i}'] = class_accuracies

        #Aggregate results
        a,b=client_model.predict(X_test)
        filtered_arr = b
        alpha = len(client_model.type)/test_data["condition"].nunique()
        sum_probability += filtered_arr*alpha

    result = np.argmax(sum_probability, axis=1)
    label = client_model.decode(list(result))
    true = list(Y_test)

    from exp.evaluation import evalute_accuracy
    evalute_accuracy(result, true, label, rd, "Average Aggregation")
