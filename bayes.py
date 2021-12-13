import load_data
import numpy as np
from evaluationUtils import evaluate
from sklearn.naive_bayes import GaussianNB
import timeit

def main():
    train_data, train_labels, test_data, test_labels, vld_data, vld_labels = load_data.load_data(
        'corona_tested_individuals_ver_006.english.csv')
    start = timeit.default_timer()
    bys = GaussianNB()
    bys.fit(train_data, train_labels)
    stop = timeit.default_timer()

    acc_bayes = bys.score(test_data, test_labels)
    test_pred = bys.predict(test_data)
    evaluate(y_true=test_labels, y_pred=test_pred, model_name="Naive Bayes")
    print('Time: ', stop - start)

if __name__ == "__main__":
    main()
