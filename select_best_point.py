import os
import numpy as np
from sklearn.metrics import average_precision_score
from collections import Counter
import pandas as pd
import argparse
import sys

def get_labels_and_contexts(dataset="nico_plus_plus_super_95", split="source-train"):
    data = np.load(f"data/metadata/{dataset}_{split}.npy", allow_pickle=True).item()
    return data["labels"], data["contexts"], None

def compute_class_wise_retrieval(scores, labs, ctxs, k=10, dataset=None):
    assert dataset is not None
    n_classes = len(np.unique(labs))
    per_slice_precs = []
    per_class_scores = []
    for cls_id in range(n_classes):
        if dataset == 'celeba' and cls_id == 0:
            continue
        scores_cls = scores[labs == cls_id]
        ctxs_cls = ctxs[labs == cls_id]
        if dataset == 'celeba':
            minority_gt = np.logical_and(ctxs_cls == 1, cls_id == 1)
        else:
            minority_gt = ctxs_cls != cls_id
        score = average_precision_score(minority_gt, scores_cls)
        top_k = np.argsort(scores_cls)[-1 : -k - 1 : -1]
        for ctx in range(n_classes):
            if dataset == 'celeba' and cls_id == 1 and ctx == 1:
                per_slice_precs.append(np.sum(ctx == ctxs_cls[top_k]) / k)
                per_class_scores.append(score)
            elif dataset != 'celeba' and ctx != cls_id:
                per_slice_precs.append(np.sum(ctx == ctxs_cls[top_k]) / k)
                per_class_scores.append(score)
    return np.mean(per_class_scores), np.mean(per_slice_precs), per_class_scores


def compute_overall_retrieval(scores, labs, ctxs, dataset=None):
    assert dataset is not None
    if dataset == 'celeba':
        minority_gt = np.logical_and(labs == 1, ctxs == 1)
    else:
        minority_gt = ctxs != labs
    return average_precision_score(minority_gt, scores)


def get_mean_majority_minority_gap(labs, ctxs,  scores, preds, dataset='celeba'):
    conf_diffs = []
    acc_diffs = []
    for c in sorted(np.unique(labs)):
        if dataset == 'celeba' and c == 0:
            continue
        labs_c =   labs[labs == c]
        ctxs_c = ctxs[labs == c]
        scores_c = scores[labs == c]
        preds_c = preds[labs == c]
        corrects_c = preds_c == labs_c
        if dataset == 'celeba':
            minority_indices = ctxs_c == 1
            majority_indices = ctxs_c == 0
        else:
            minority_indices = ctxs_c != labs_c
            majority_indices = ctxs_c == labs_c
        acc_diffs.append(corrects_c[majority_indices].mean() - corrects_c[minority_indices].mean())
        conf_diffs.append(scores_c[majority_indices].mean() - scores_c[minority_indices].mean())
    return np.mean(acc_diffs), np.mean(conf_diffs)


def collect_retrieval_at_all_points(all_outputs, dataset=None):
    assert dataset is not None
    dfs = {}
    for split in all_outputs.keys():
        dfs[split] = pd.DataFrame()
        outputs = all_outputs[split]
        for idx, eval_step in enumerate(outputs["eval_steps"]):
            labs = outputs["labs"][idx]
            classes = len(np.unique(labs))
            ctxs = outputs["ctxs"][idx]
            scores = outputs["conf_gts"][idx]
            acc = 100 * (outputs['outputs'][idx].argmax(-1) == labs).sum() / len(labs)
            classwise_accs = [100 * (outputs['outputs'][idx][labs == i].argmax(-1) == i).sum() / len(labs[labs == i]) for i in range(classes)]
            mean_acc = np.mean(classwise_accs)
            classwise_conf_gt_means = [scores[labs == i].mean() for i in range(classes)]
            classwise_conf_gt_max = np.array([scores[labs == i].max() for i in range(classes)])
            classwise_conf_gt_min = np.array([scores[labs == i].min() for i in range(classes)])
            classwise_conf_gt_vars = np.array([scores[labs == i].var() for i in range(classes)])
            conf_gap, acc_gap = get_mean_majority_minority_gap(labs, ctxs, scores, outputs['outputs'][idx].argmax(-1), dataset=dataset)
            overall = compute_overall_retrieval(1 - scores, labs, ctxs, dataset=dataset)
            cl_wise_score, cl_wise_prec, per_class_scores = compute_class_wise_retrieval(
                1 - scores, labs, ctxs, k=10, dataset=dataset
            )
            class_diff = np.abs(np.array(classwise_accs)[:, np.newaxis] - np.array(classwise_accs)[np.newaxis, :]).sum() /  2
            new_row = {'eval_idx': idx, 'eval_step':eval_step, 'acc':acc, 'mean_acc': mean_acc, 'overall_ap': overall,
                        'average_classwise_ap': cl_wise_score, 'prec_at_10': cl_wise_prec,  'conf_gt_vars': classwise_conf_gt_vars.mean(),
                        'maj_min_conf_gap': conf_gap, 'maj_min_acc_gap': acc_gap}
            dfs[split] = pd.concat([dfs[split], pd.DataFrame(new_row, index=[idx])])
        dfs[split] = dfs[split].set_index('eval_step')
    return dfs


def get_best_retrieval_point(stats, selection='oracle'):
    if selection == 'best_cls_wise_train_retrieval':
        return stats['source-train-eval'].nlargest(1, 'average_classwise_ap')[['eval_idx']].reset_index()
    elif selection == 'max_training_acc':
        return stats['source-train-eval'].nlargest(1, 'acc')[['eval_idx']].reset_index()
    else:
        raise NotImplementedError


def get_numbers_best_steps(stats, dataset='nico_plus_plus_super_95'):
    for selection in ['max_training_acc']:
        best_point = get_best_retrieval_point(stats,  selection=selection)
        for eval_split in ['source-train-eval']: 
            row = stats[eval_split].loc[best_point['eval_step'].item()]
            print(f'Selected eval step ({selection}): ', best_point['eval_step'].item())
            print(f'Selected eval idx ({selection}): ', best_point['eval_idx'].item())
            print(f'Avg AP @ selected point ({selection}): ', row['average_classwise_ap'])
            print('sigma_{AmCo}: ', row['conf_gt_vars'])


def get_results(all_outputs, dataset=None):
    dfs = collect_retrieval_at_all_points(all_outputs, dataset=dataset)
    return get_numbers_best_steps(dfs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_file", type=str, default="outputs/nico_plus_plus/amco_95/seed_0/all_outputs.npy")
    parser.add_argument(
        "--dataset_name", type=str, default='nico_plus_plus_super_95'
    )
    args = parser.parse_args()
    all_outputs = np.load(args.outputs_file, allow_pickle=True).item()
    get_results(all_outputs, dataset=args.dataset_name)