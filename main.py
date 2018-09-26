import numpy as np
import math

path = 'imdb/'
x_train = np.load(path + 'x_train.npy')
y_train = np.load(path + 'y_train.npy')
x_test = np.load(path + 'x_test.npy')
y_test = np.load(path + 'y_test.npy')

pos_dict = {}
neg_dict = {}

combined_train = zip(x_train, y_train)


def divide_sum_in_dict(my_dict):
    sum_count = sum(my_dict.values())
    for key, value in my_dict.items():
        my_dict[key] = value / sum_count
    return my_dict


pos_probability = divide_sum_in_dict(pos_dict)
neg_probability = divide_sum_in_dict(neg_dict)


def store_to_dict(my_dict, word):
    my_dict[word] = my_dict.get(word, 0) + 1
    return my_dict


pos_smooth_data = 1 / (sum(pos_probability.values()) + 1)
neg_smooth_data = 1 / (sum(neg_probability.values()) + 1)

for each_data in combined_train:
    if each_data[1] == 1:  # pos
        for word in each_data[0]:
            pos_dict = store_to_dict(pos_dict, word)
    if each_data[1] == 0:  # neg
        for word in each_data[0]:
            neg_dict = store_to_dict(neg_dict, word)

def topK_testing(K):
    true_pos = 0
    true_neg = 0
    false_pos = 0
    false_neg = 0
    combined_test = zip(x_test, y_test)
    for each_data in combined_test:
        if each_data[1] == 1:  # pos
            pos_probability_val = 0.0
            neg_probability_val = 0.0
            for word in each_data[0]:
                if word <= K:
                    pos_probability_val += math.log(pos_probability.get(word, pos_smooth_data))
                    neg_probability_val += math.log(neg_probability.get(word, neg_smooth_data))
            if pos_probability_val > neg_probability_val:
                true_pos += 1
            else:
                false_pos += 1
        if each_data[1] == 0:  # neg
            pos_probability_val = 0.0
            neg_probability_val = 0.0
            for word in each_data[0]:
                if word <= K:
                    pos_probability_val += math.log(pos_probability.get(word, pos_smooth_data))
                    neg_probability_val += math.log(neg_probability.get(word, neg_smooth_data))
            if pos_probability_val > neg_probability_val:
                false_neg += 1
            else:
                true_neg += 1

    return true_pos, true_neg, false_pos, false_neg



print(topK_testing(100))
print(topK_testing(1000))
print(topK_testing(10000))
