import numpy as np

path = 'imdb/'
x_train = np.load(path + 'x_train.npy')
y_train = np.load(path + 'y_train.npy')
x_test = np.load(path + 'x_test.npy')
y_test = np.load(path + 'y_test.npy')

pos_dict = {}
neg_dict = {}

combined_train = zip(x_train, y_train)


def store_to_dict(my_dict, word):
    my_dict[word] = my_dict.get(word, 0) + 1
    return my_dict

# def dict_add(a_dict, b_dict):
#     # z = {**a_dict, **b_dict}
#     result = {key: a_dict.get(key, 0) + b_dict.get(key, 0)
#               for key in set(a_dict) | set(b_dict)}
#     return result

def divide_sum_in_dict(my_dict):
    sum_count = sum(my_dict.values())
    for key, value in my_dict.items():
        my_dict[key] = value / sum_count
    return my_dict


for each_data in combined_train:
    if each_data[1] == 1:  # pos
        for word in each_data[0]:
            pos_dict = store_to_dict(pos_dict, word)
    if each_data[1] == 0:  # neg
        for word in each_data[0]:
            neg_dict = store_to_dict(neg_dict, word)

# print(pos_dict)
# print(neg_dict)
# word_total_appear = dict_add(pos_dict, neg_dict)
# sorted_by_value = sorted(word_total_appear.items(), key=lambda kv: kv[1], reverse=True)
# pos_total_count = sum(pos_dict.values())
# neg_total_count = sum(neg_dict.values())
# print(pos_total_count)
# print(neg_total_count)

pos_probability = divide_sum_in_dict(pos_dict)
net_probability = divide_sum_in_dict(neg_dict)
# print(pos_probability)

for each_data in

