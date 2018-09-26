import numpy as np
import math


class Bayes_Classifier():

    def __init__(self):
        path = 'imdb/'
        x_train = np.load(path + 'x_train.npy')
        y_train = np.load(path + 'y_train.npy')
        x_test = np.load(path + 'x_test.npy')
        y_test = np.load(path + 'y_test.npy')
        pos_dict = {}
        neg_dict = {}

        combined_train = zip(x_train, y_train)
        for each_data in combined_train:
            if each_data[1] == 1:  # pos
                for word in each_data[0]:
                    pos_dict[word] = pos_dict.get(word, 0) + 1
            if each_data[1] == 0:  # neg
                for word in each_data[0]:
                    neg_dict[word] = neg_dict.get(word, 0) + 1

        pos_probability = self.divide_sum_in_dict(pos_dict)
        neg_probability = self.divide_sum_in_dict(neg_dict)

    def divide_sum_in_dict(my_dict):
        sum_count = sum(my_dict.values())
        for key, value in my_dict.items():
            my_dict[key] = value / sum_count
        return my_dict

    def topK_testing(self, K):
        true_pos = 0
        true_neg = 0
        false_pos = 0
        false_neg = 0
        combined_test = zip(self.x_test, self.y_test)
        for each_data in combined_test:
            if each_data[1] == 1:  # pos
                pos_probability_val = 0.0
                neg_probability_val = 0.0
                for word in each_data[0]:
                    if word <= K:
                        pos_probability_val += math.log(self.pos_probability.get(word, 1.0 / (len(self.pos_probability) + 1)))
                        neg_probability_val += math.log(self.neg_probability.get(word, 1.0 / (len(self.neg_probability) + 1)))
                if pos_probability_val > neg_probability_val:
                    true_pos += 1
                else:
                    false_pos += 1
            if each_data[1] == 0:  # neg
                pos_probability_val = 0.0
                neg_probability_val = 0.0
                for word in each_data[0]:
                    if word <= K:
                        pos_probability_val += math.log(self.pos_probability.get(word, 1.0 / (len(self.pos_probability) + 1)))
                        neg_probability_val += math.log(self.neg_probability.get(word, 1.0 / (len(self.neg_probability) + 1)))
                if pos_probability_val > neg_probability_val:
                    false_neg += 1
                else:
                    true_neg += 1

        return [true_pos, true_neg, false_pos, false_neg]

    def calculate_accuracy(result_list):
        TPs, FPs, FNs, TNs = result_list
        return (TPs + TNs) / (TPs + FPs + FNs + TNs)

    def calculate_precision(result_list):
        TPs, FPs, FNs, TNs = result_list
        return (TPs) / (TPs + FPs)

    def calculate_recall(result_list):
        TPs, FPs, FNs, TNs = result_list
        return (TPs) / (TPs + FNs)


print("Top 100: ")
top_100 = Bayes_Classifier()
print(top_100.topK_testing(100))
# top_100 = topK_testing(100, pos_probability, neg_probability)
# print(calculate_accuracy(top_100))
# print(calculate_precision(top_100))
# print(calculate_recall(top_100))
# print("Top 1000: ")
# top_1000 = topK_testing(1000, pos_probability, neg_probability)
# print(calculate_accuracy(top_1000))
# print(calculate_precision(top_1000))
# print(calculate_recall(top_1000))
# print("Top 10000: ")
# top_10000 = topK_testing(10000, pos_probability, neg_probability)
# print(calculate_accuracy(top_10000))
# print(calculate_precision(top_10000))
# print(calculate_recall(top_10000))
