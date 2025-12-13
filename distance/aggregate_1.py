import numpy as np
import pickle

def aggregate_test(num_client,rd,test_data):
    with open(f"client_model/{rd}/logging.txt", 'a', encoding='utf-8') as f:
        print(f"***下面是投票法的测试结果***", file=f)
    sum_probability = np.zeros((test_data.shape[0],test_data["condition"].nunique()),dtype=float)

    X_test = test_data.loc[:, "U1":"U41"]
    Y_test = test_data["condition"]

    for i in range(0,num_client):
        #估计每个client model的准确率
        client_model = pickle.load(open("client_model/"+str(rd)+"/client_"+str(i)+".pkl",
                                          'rb'))

        #聚合结果
        a,b=client_model.predict(X_test)
        filtered_arr = np.where(b > 0.6, b, 0)
        all_zero_rows = np.all(filtered_arr == 0, axis=1)
        all_zero_indices = np.where(all_zero_rows)[0]
        # print("困难样本索引:", all_zero_indices)
        alpha = len(client_model.type)/test_data["condition"].nunique()
        sum_probability += filtered_arr*alpha


    final_all_zero_rows = np.all(sum_probability == 0, axis=1)
    final_all_zero_indices = np.where(final_all_zero_rows)[0]


    result = np.argmax(sum_probability, axis=1)
    label = client_model.decode(list(result))
    true = list(Y_test)
    from exp.evaluation import evalute_accuracy
    evalute_accuracy(result, true, label, rd, "种类数量加权聚合法")


