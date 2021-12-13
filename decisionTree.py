import numpy as np
import load_data
import time
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from evaluationUtils import evaluate, visualize_roc_curve


def single_decision_tree(train_data, train_labels, test_data, test_labels, max_depth, criterion):
    beforeTrainingTimeStamp = time.time()
    clf = DecisionTreeClassifier(
        max_depth=max_depth, criterion=criterion, random_state=1)
    clf.fit(train_data, train_labels)
    afterTrainingTimeStamp = time.time()
    print("Test Accuracy of single decision tree : ",
          clf.score(test_data, test_labels))
    afterPredictingTimeStamp = time.time()
    trainTime = afterTrainingTimeStamp - beforeTrainingTimeStamp
    predictTime = afterPredictingTimeStamp - afterTrainingTimeStamp
    totalTime = afterPredictingTimeStamp - beforeTrainingTimeStamp
    print("Train Time: {}".format(trainTime))
    print("Predict Time: {}".format(predictTime))
    print("Total Time: {}".format(totalTime))
    return clf


def adaboosted_decision_tree(train_data, train_labels, test_data, test_labels, n_estimators, max_depth):
    unit_decision_tree = DecisionTreeClassifier(max_depth=max_depth)
    clf = AdaBoostClassifier(unit_decision_tree, n_estimators=n_estimators)
    clf.fit(train_data, train_labels)
    pred = clf.predict(test_data)
    print("Test Accuracy of adaboosted decision tree: ",
          clf.score(test_data, test_labels))
    return pred


def bagging_decision_tree(train_data, train_labels, test_data, test_labels, n_estimators, max_depth):
    unit_decision_tree = DecisionTreeClassifier(max_depth=max_depth)
    clf = BaggingClassifier(unit_decision_tree, n_estimators=n_estimators)
    clf.fit(train_data, train_labels)
    pred = clf.predict(test_data)
    print("Test Accuracy of bagging decision tree: ",
          clf.score(test_data, test_labels))
    return pred


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels, vld_data, vld_labels = load_data.load_data(
        'corona_tested_individuals_ver_006.english.csv')

    # Decision Tree
    clf = single_decision_tree(train_data, train_labels,
                         test_data, test_labels, 11, criterion='gini')
    visualize_roc_curve(clf, test_data, test_labels)
    # Adaboost + Decision Tree
    test_pred = adaboosted_decision_tree(train_data, train_labels,
                                         test_data, test_labels, 50, 11)

    # Bagging + Decision Tree
    bagging_decision_tree(train_data, train_labels,
                          test_data, test_labels, 50, 11)

    evaluate(y_true=test_labels, y_pred=test_pred, model_name="Decision Tree")

    """
    Find optimal params for Decision Tree, Result: criterion: 'gini'; max_depth: 11
    """
    # for criterion in ('gini', 'entropy'):
    #     for max_depth in range(1, 10, 1):
    #         print(criterion, max_depth)
    #         single_decision_tree(train_data, train_labels,
    #                              test_data, test_labels, max_depth=max_depth, criterion=criterion)