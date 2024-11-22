import numpy as np
from CNN import predict

def get_cutoff_score(state, images, onehot_labels, coverage, m):
    softmax_all = predict(state, images)
    softmax_true = []
    for i in range (0, len(onehot_labels)):
        label = onehot_labels[i]
        true_index = np.where(label == 1)[0][0]
        # Get softmax scores of true classes
        true_score = softmax_all[i][true_index]
        softmax_true.append(true_score)
    # Sort true softmax scores in descending order
    softmax_true = -np.sort(-np.array(softmax_true))
    # Find cutoff softmax score
    lowest_index = int(np.floor(coverage * m * ((len(images) / m) + 1)) - m)
    cutoff_score = softmax_true[lowest_index]
    return cutoff_score

def conformal_prediction(state, images, labels, cutoff_score):
    softmax_all = predict(state, images)
    predictions = np.zeros((len(images), len(labels[0])))
    # Include all softmax scores greater than cutoff score in prediction set
    for i in range (0, len(softmax_all)):
        scores = np.array(softmax_all[i])
        new_scores = np.where(scores >= cutoff_score, scores, scores * 0)
        predictions[i] = new_scores
    return predictions

def test_coverage(onehot_labels, prediction_set):
    contains_correct_class = []
    for i in range (0, len(onehot_labels)):
        correct_class = np.where(onehot_labels[i] == 1)[0][0]
        # Check if correct class has a softmax score greater than 0 in the prediction set
        if prediction_set[i][correct_class] != 0:
            contains_correct_class.append(True)
        else:
                contains_correct_class.append(False)
    coverage = np.mean(np.asarray(contains_correct_class))
    return coverage