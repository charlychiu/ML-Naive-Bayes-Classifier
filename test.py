import numpy as np
import math


def divide_total_review_in_dict(my_dict, total_review):
    for key, value in my_dict.items():
        my_dict[key] = value / total_review
    return my_dict


def topK_testing(K, pos_probability, neg_probability, x_test, y_test):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    combined_test = zip(x_test, y_test)
    for each_data in combined_test:
        if each_data[1] == 1:  # pos
            pos_probability_val = 0.0
            neg_probability_val = 0.0
            for word in list(set(each_data[0])):
                if word <= K:
                    pos_probability_val += math.log(
                        pos_probability.get(word, 1.0 / (len(pos_probability) + 1)))
                    neg_probability_val += math.log(
                        neg_probability.get(word, 1.0 / (len(neg_probability) + 1)))
            if pos_probability_val > neg_probability_val:
                true_pos += 1
            else:
                false_pos += 1
        if each_data[1] == 0:  # neg
            pos_probability_val = 0.0
            neg_probability_val = 0.0
            for word in list(set(each_data[0])):
                if word <= K:
                    pos_probability_val += math.log(
                        pos_probability.get(word, 1.0 / (len(pos_probability) + 1)))
                    neg_probability_val += math.log(
                        neg_probability.get(word, 1.0 / (len(neg_probability) + 1)))
            if pos_probability_val > neg_probability_val:
                false_neg += 1
            else:
                true_neg += 1

    return [true_pos, true_neg, false_pos, false_neg]


def calculate_accuracy(result_list):
    TPs, TNs, FPs, FNs = result_list
    return (TPs + TNs) / (TPs + FPs + FNs + TNs)


def calculate_precision(result_list):
    TPs, TNs, FPs, FNs = result_list
    return (TPs) / (TPs + FPs)


def calculate_recall(result_list):
    TPs, TNs, FPs, FNs = result_list
    return (TPs) / (TPs + FNs)


path = 'imdb/'
x_train = np.load(path + 'x_train.npy')
y_train = np.load(path + 'y_train.npy')
x_test = np.load(path + 'x_test.npy')
y_test = np.load(path + 'y_test.npy')
pos_dict = {}
neg_dict = {}
combined_train = zip(x_train, y_train)

count_pos = 0
count_neg = 0

for each_data in combined_train:
    if each_data[1] == 1:  # pos
        for word in list(set(each_data[0])):
            pos_dict[word] = pos_dict.get(word, 0) + 1
        count_pos += 1

    if each_data[1] == 0:  # neg
        for word in list(set(each_data[0])):
            neg_dict[word] = neg_dict.get(word, 0) + 1
        count_neg += 1

pos_probability = divide_total_review_in_dict(pos_dict, count_pos)
neg_probability = divide_total_review_in_dict(neg_dict, count_neg)

print("Top 100: ")
top_100 = topK_testing(100, pos_probability, neg_probability, x_test, y_test)
print("Accuracy: {}".format(calculate_accuracy(top_100)))
print("Precision: {}".format(calculate_precision(top_100)))
print("Recall: {}".format(calculate_recall(top_100)))
print("Top 1000: ")
top_1000 = topK_testing(1000, pos_probability, neg_probability, x_test, y_test)
print("Accuracy: {}".format(calculate_accuracy(top_1000)))
print("Precision: {}".format(calculate_precision(top_1000)))
print("Recall: {}".format(calculate_recall(top_1000)))
print("Top 10000: ")
top_10000 = topK_testing(10000, pos_probability, neg_probability, x_test, y_test)
print("Accuracy: {}".format(calculate_accuracy(top_10000)))
print("Precision: {}".format(calculate_precision(top_10000)))
print("Recall: {}".format(calculate_recall(top_10000)))
