import math
from collections import defaultdict
import sys


def remove_head_tags(context):
    s = context.find('<head>')
    e = context.find('</head>')

    if s and e:
        return context[:s] + context[e + 7:]
    return context


def read_data(filename):
    instances = []
    with open(filename, 'r') as f:
        lines = f.readlines()

    instance_id, sense_id, context = None, None, []
    inside_context = False
    temp_context = ""

    for line in lines:
        line = line.strip()

        if line.startswith('<answer instance='):
            instance_id = line.split('"')[1]
            sense_id = line.split('"')[3]

        elif line.startswith('<context>'):
            inside_context = True
            temp_context = ""

        elif line.startswith('</context>'):
            inside_context = False
            normalized_context = remove_head_tags(temp_context).lower().split()
            instances.append((instance_id, sense_id, normalized_context))

        elif inside_context:
            temp_context += line + " "

    return instances


def naive_bayes(train_data, test_data):
    sense_counts = defaultdict(int)
    word_given_sense_counts = defaultdict(lambda: defaultdict(int))
    vocab = set()
    predictions = []

    for _, sense, context in train_data:
        sense_counts[sense] += 1
        for word in context:
            word_given_sense_counts[sense][word] += 1
            vocab.add(word)

    V = len(vocab)

    for instance_id, _, context in test_data:
        max_prob = float('-inf')
        best_sense = None
        for sense in sense_counts:
            log_prob = math.log2(sense_counts[sense])
            for word in context:
                count = word_given_sense_counts[sense][word] + 1
                log_prob += math.log2(count / (sense_counts[sense] + V))
            if log_prob > max_prob:
                max_prob = log_prob
                best_sense = sense
        predictions.append((instance_id, best_sense))

    return predictions


def main():
    filename = sys.argv[1]
    data = read_data(filename)
    num_folds = 5
    f_size = len(data) // num_folds
    remain = len(data) % num_folds
    f_sizes = [f_size for _ in range(num_folds)]

    for i in range(remain):
        f_sizes[i] += 1


    fold_s = 0
    accuracies = []

    target_word = filename.split('.')[0]
    print(f'Target Word: {target_word}')

    with open(f'{target_word}.wsd.out', 'w') as outfile:

        for fold in range(1, num_folds + 1):
            fold_e = fold_s + f_sizes[fold - 1]

            test_data = data[fold_s:fold_e]
            train_data = data[:fold_s] + data[fold_e:]

            predictions = naive_bayes(train_data, test_data)

            correct = 0

            for true_instance, predicted_instance in zip(test_data, predictions):
                (_, true_sense, _) = true_instance
                (_, predicted_sense) = predicted_instance

                if true_sense == predicted_sense:
                    correct += 1

            accuracy = correct / len(test_data) * 100
            accuracies.append(accuracy)

            print(f'Accuracy for {target_word} - Fold {fold}: {accuracy:.2f}%')

            outfile.write(f'Fold {fold}\n')
            for instance_id, predicted_sense in predictions:
                outfile.write(f'{instance_id} {predicted_sense}\n')

            fold_s = fold_e

        average_accuracy = sum(accuracies) / len(accuracies)
        print(f'Average Accuracy for {target_word}: {average_accuracy:.2f}%')


if __name__ == '__main__':
    main()
