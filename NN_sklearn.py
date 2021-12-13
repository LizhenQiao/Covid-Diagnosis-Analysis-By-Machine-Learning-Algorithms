import load_data
from evaluationUtils import evaluate
from sklearn.neural_network import MLPClassifier


def mlp(train_data, train_labels, test_data, test_labels):
    clf = MLPClassifier(random_state=1)
    clf.fit(train_data, train_labels)
    test_pred = clf.predict(test_data)
    acc = clf.score(test_data, test_labels)
    print(acc)
    return test_pred


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels, vld_data, vld_labels = load_data.load_data(
        'corona_tested_individuals_ver_006.english.csv')
    test_pred = mlp(train_data, train_labels, test_data, test_labels)
    evaluate(y_true=test_labels, y_pred=test_pred, model_name="NN")
