import load_data
from sklearn.model_selection import KFold, train_test_split
from sklearn import svm
from sklearn.svm import SVC
from sklearn import metrics

def svm(train_data, train_labels, test_data, test_labels):

    non_linear_model = SVC(kernel='rbf',probability=True)
    # fit
    non_linear_model.fit(train_data, train_labels)

    # predict
    y_pred = non_linear_model.predict(test_data)
    print("\naccuracy:", metrics.accuracy_score(y_true=test_labels, y_pred=y_pred), "\n")


if __name__ == "__main__":
    train_data, train_labels, test_data, test_labels, vld_data, vld_labels = load_data.load_data(
        'corona_tested_individuals_ver_006.english.csv')
    svm(train_data, train_labels, test_data, test_labels)