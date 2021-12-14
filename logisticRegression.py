import load_data
import time
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression, LinearRegression
from evaluationUtils import evaluate


def logisticRegression(train_data, train_labels, test_data, test_labels, penalty):
    beforeTrainingTimeStamp = time.time()
    model = LogisticRegression(penalty=penalty, C=1.0)
    # Fit the model
    model.fit(train_data, train_labels)
    afterTrainingTimeStamp = time.time()
    # Score/Accuracy
    acc_logreg = model.score(test_data, test_labels)
    test_pred = model.predict(test_data)
    evaluate(y_true=test_labels, y_pred=test_pred,
             model_name="logistic regression")
    afterPredictingTimeStamp = time.time()
    trainTime = afterTrainingTimeStamp - beforeTrainingTimeStamp
    predictTime = afterPredictingTimeStamp - afterTrainingTimeStamp
    totalTime = afterPredictingTimeStamp - beforeTrainingTimeStamp
    print("Train Time: {}".format(trainTime))
    print("Predict Time: {}".format(predictTime))
    print("Total Time: {}".format(totalTime))
    print("Logistic Regression Accuracy: {}".format(acc_logreg))

    # Get weights of each feature.
    X_train_minmax = preprocessing.normalize(train_data)
    # X_train_minmax = min_max_scaler.fit_transform(train_data)
    reg = LogisticRegression(penalty=penalty)
    reg.fit(X_train_minmax, train_labels)
    print(reg.score(test_data, test_labels))
    w = reg.coef_
    print("w: {}".format(w))


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels, vld_data, vld_labels = load_data.load_data(
        'corona_tested_individuals_ver_006.english.csv')

    logisticRegression(train_data, train_labels, test_data, test_labels, 'l2')
