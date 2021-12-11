import numpy as np
import load_data

from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import label_binarize
from evaluationUtils import evaluate


def single_decision_tree(train_data, train_labels, test_data, test_labels, max_depth):
    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(train_data, train_labels)
    print("Test Accuracy of single decision tree : ",
          clf.score(test_data, test_labels))
    return None


def adaboosted_decision_tree(train_data, train_labels, test_data, test_labels, n_estimators, max_depth):
    unit_decision_tree = DecisionTreeClassifier(max_depth=max_depth)
    clf = AdaBoostClassifier(unit_decision_tree, n_estimators=n_estimators)
    clf.fit(train_data, train_labels)
    pred = clf.predict(test_data)
    print("Test Accuracy of adaboosted decision tree: ",
          clf.score(test_data, test_labels))
    return pred


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels, vld_data, vld_labels = load_data.load_data(
        'corona_tested_individuals_ver_006.english.csv')

    # Decision Tree
    single_decision_tree(train_data, train_labels,
                         test_data, test_labels, 10)

    # Adaboost + Decision Tree
    test_pred = adaboosted_decision_tree(train_data, train_labels,
                                         test_data, test_labels, 50, 10)

    evaluate(y_true=test_labels, y_pred=test_pred, model_name="Decision Tree")
