import matplotlib.pyplot as plt
import numpy as np


def plot():
    labels = ['bayes', 'decision tree', 'knn', 'LR', 'NN', 'svm']

    trainTime = [0.0176, 0.036, 7.611, 0.181, 14.67, 26.21]
    testTime = [0.005, 0.003, 16.685, 0.002, 1.55, 9.58]
    totalTime = [0.0226, 0.039, 24.297, 0.183, 16.22, 35.79]

    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars

    fig, ax = plt.subplots()
    rects1 = ax.bar(x - width, trainTime, width, label='Train')
    rects2 = ax.bar(x, testTime, width, label='Test')
    rects3 = ax.bar(x + width, totalTime, width, label='Total')

    # Add some text for labels, title and custom x-axis tick labels, etc.
    ax.set_ylabel('Time')
    ax.set_title('Run time of each model')
    ax.set_xticks(x, labels)
    ax.legend()

    ax.bar_label(rects1, padding=3)
    ax.bar_label(rects2, padding=3)
    ax.bar_label(rects3, padding=3)

    fig.tight_layout()

    plt.show()


if __name__ == "__main__":

    """
    Time Statistic
    list index: 0, 1, 2 => Training Time, Test Time, Total Time
    """
    bayesTime = [0.0176, 0.005, 0.0226]
    decisionTreeTime = [0.036, 0.003, 0.039]
    knnTime = [7.611, 16.685, 24.297]
    logisticRegressionTime = [0.181, 0.002, 0.183]
    neuralNetworkTime = [14.67, 1.55, 16.22]
    svmTime = [26.21, 9.58, 35.79]

    """
    Accuracy Statistic
    Accuracy, Precision, Recall
    """
    bayes = [0.9379, 0.9474, 0.9379]
    decisionTree = [0.9614, 0.9587, 0.9614]
    knn = [0.9573, 0.954, 0.9573]
    logisticRegression = [0.9556, 0.9521, 0.9556]
    neuralNetwork = [0.9603, 0.9576, 0.9603]
    svm = [0.9627, 0.9602, 0.9627]

    plot()
