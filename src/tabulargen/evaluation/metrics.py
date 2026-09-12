import numpy as np
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score, precision_recall_curve

def evaluate(y_true, y_pred, task_type, threshold=0.5):
    if task_type == 'binclass':
        y_prob = y_pred
        y_label = (y_prob > threshold).astype(int)

        f1 = f1_score(y_true, y_label, average='weighted')
        acc = accuracy_score(y_true, y_label)
        auc = roc_auc_score(y_true, y_prob)

        # balanced_acc = balanced_accuracy_score(y_true, y_label)

    elif task_type == 'multiclass':
        y_prob = y_pred
        y_label = np.argmax(y_prob, axis=1)

        f1 = f1_score(y_true, y_label, average='macro')
        acc = accuracy_score(y_true, y_label)
        auc = roc_auc_score(y_true, y_prob, multi_class='ovr')
    else:
        raise 'Task type is error!'

    return {
        'f1': f1,
        'accuracy': acc,
        'roc_auc': auc
    }


def get_optimal_threshold_from_pr(y_true, y_prob):
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1_scores = 2 * precision[1:] * recall[1:] / (precision[1:] + recall[1:] + 1e-8)
    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]


def print_metrics(results):
    res = {
        "val": {k: np.around(results["val"][k], 4) for k in results["val"]},
        "test": {k: np.around(results["test"][k], 4) for k in results["test"]}
    }

    print("*"*100)
    print("[val]")
    print(res["val"])
    print("[test]")
    print(res["test"])

    return res




def average_metrics(per_model):
    """Give each fitted model one equal vote, separately for every split/metric."""
    if not per_model:
        raise ValueError('At least one model is required for averaging')
    return {
        split: {
            metric: sum(result[split][metric] for result in per_model.values()) / len(per_model)
            for metric in ['f1', 'accuracy', 'roc_auc']
        }
        for split in ['train', 'val', 'test']
    }
