import load_data
from sklearn.linear_model import LogisticRegression


if __name__ == '__main__':
    train_data, train_labels, test_data, test_labels, vld_data, vld_labels = load_data.load_data(
        'corona_tested_individuals_ver_006.english.csv')

model = LogisticRegression()
# Fit the model
model.fit(train_data, train_labels)
# Score/Accuracy
acc_logreg = model.score(test_data, test_labels)

print("Logistic Regression Accuracy: {}".format(acc_logreg))
