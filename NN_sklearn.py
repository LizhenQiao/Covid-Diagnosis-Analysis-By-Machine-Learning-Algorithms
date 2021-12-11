import load_data
from sklearn.neural_network import MLPClassifier


def mlp(train_data, train_labels, test_data, test_labels):
    clf = MLPClassifier(solver='lbfgs', alpha=1e-5,
                        hidden_layer_sizes=(5, 2), random_state=1)
    clf.fit(train_data, train_labels)
    acc = clf.score(test_data, test_labels)
    print(acc)


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels, vld_data, vld_labels = load_data.load_data(
        'corona_tested_individuals_ver_006.english.csv')
    mlp(train_data, train_labels, test_data, test_labels)
