import collections
from collections import namedtuple

METRICS =  ["top1_acc", "top1_recall", "top1_f1",
            "maj_acc", "maj_recall", "maj_f1"]
MetricsTuple = namedtuple("MetricsTuple", METRICS)

def compute_smart_maj_preds(ranks, distances, train_labels, maj_closeness_threshold: float = 0.25, maj_zero_branch: bool = False) -> list:
    num_samples = len(ranks)
    maj_vote = []
    zero_mask = distances == 0
    close_mask = distances <= maj_closeness_threshold
    for i in range(num_samples):
        if maj_zero_branch and zero_mask[i].any():
            pred = collections.Counter(train_labels[ranks[i][zero_mask[i]]]).most_common(1)[0][0]
        elif close_mask[i].any():
            pred = collections.Counter(train_labels[ranks[i][close_mask[i]]]).most_common(1)[0][0]
        else:
            # Fallback to the closest 1 prediction
            pred = train_labels[ranks[i][0]]
        maj_vote.append(pred)
    return maj_vote
