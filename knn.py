import load_data
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from evaluationUtils import evaluate
import time


def cross_validation(train_data, train_labels, k_range=np.arange(1, 16)):
    kf10 = KFold(n_splits=10, shuffle=False)
    # k_best_list = []
    accuracy_list = np.zeros(len(k_range))
    for train_index, test_index in kf10.split(train_data):
        x_train, x_test = train_data[train_index], train_data[test_index]
        y_train, y_test = train_labels[train_index], train_labels[test_index]
        # best_score = 0
        # best_k = 0
        for i in k_range:
            k = i*2-1
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(x_train, y_train)
            score = knn.score(x_test, y_test)
            accuracy_list[i - 1] += score
        #     if score > best_score:
        #         best_score = score
        #         best_k = k
        # k_best_list.append(best_k)
    accuracy_list = accuracy_list / 10
    print("accuracy list")
    print(accuracy_list)
    accuracy_list = list(accuracy_list)
    index_k = accuracy_list.index(max(accuracy_list)) + 1
    optimal_k_avg_accuracy = accuracy_list[index_k - 1]
    optimal_k = index_k*2-1
    print("Optimal K:", optimal_k)
    plt.plot(k_range*2-1, accuracy_list)
    plt.xlabel("k")
    plt.ylabel("average accuracy over 10-fold cross validation")
    plt.show()
    return optimal_k, optimal_k_avg_accuracy


def main():
    train_data, train_labels, test_data, test_labels, vld_data, vld_labels = load_data.load_data(
        'corona_tested_individuals_ver_006.english.csv')
    # optimal_k, optimal_k_avg_accuracy = cross_validation(train_data, train_labels, k_range=np.arange(1, 16))
    optimal_k = 11
    optimal_k_avg_accuracy = 0.9601

    beforeTrainingTimeStamp = time.time()
    knn = KNeighborsClassifier(n_neighbors=optimal_k)
    knn.fit(train_data, train_labels)
    afterTrainingTimeStamp = time.time()

    # optimal_k_train_score = knn.score(train_data, train_labels)
    optimal_k_test_score = knn.score(test_data, test_labels)
    print("k = ", optimal_k,  ", the average accuracy across folds is: ",
          optimal_k_avg_accuracy, ", test accuracy is:", optimal_k_test_score)
    afterPredictingTimeStamp = time.time()
    # print("k = ", optimal_k, ", train accuracy is: ", optimal_k_train_score, ", the average accuracy across folds is: ",
    #       optimal_k_avg_accuracy, ", test accuracy is:", optimal_k_test_score)
    test_pred = knn.predict(test_data)
    evaluate(y_true=test_labels, y_pred=test_pred, model_name="KNN")
    trainTime = afterTrainingTimeStamp - beforeTrainingTimeStamp
    predictTime = afterPredictingTimeStamp - afterTrainingTimeStamp
    totalTime = afterPredictingTimeStamp - beforeTrainingTimeStamp
    print("Train Time: {}".format(trainTime))
    print("Predict Time: {}".format(predictTime))
    print("Total Time: {}".format(totalTime))


if __name__ == "__main__":
    main()
