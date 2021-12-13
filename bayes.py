import load_data
import numpy as np
from evaluationUtils import evaluate
from sklearn.naive_bayes import GaussianNB
import time


def main():
    train_data, train_labels, test_data, test_labels, vld_data, vld_labels = load_data.load_data(
        'corona_tested_individuals_ver_006.english.csv')
    beforeTrainTimeStamp = time.time()
    bys = GaussianNB()
    bys.fit(train_data, train_labels)
    afterTrainingTimeStamp = time.time()

    acc_bayes = bys.score(test_data, test_labels)
    afterPredictingTimeStamp = time.time()
    test_pred = bys.predict(test_data)
    evaluate(y_true=test_labels, y_pred=test_pred, model_name="Naive Bayes")
    trainTime = afterTrainingTimeStamp - beforeTrainTimeStamp
    predictTime = afterPredictingTimeStamp - afterTrainingTimeStamp
    totalTime = afterPredictingTimeStamp - beforeTrainTimeStamp
    print("Train Time: {}".format(trainTime))
    print("Predict Time: {}".format(predictTime))
    print("Total Time: {}".format(totalTime))


if __name__ == "__main__":
    main()
