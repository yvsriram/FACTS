#!/usr/bin/env python
# coding: utf-8



import meerkat as mk
import pandas as pd
import sys
from CoSi import CoSiSlicer, DominoSlicer, CaptionModel, embed
import os

import numpy as np
from  collections import defaultdict, Counter
from itertools import product
import sys
from typing import  List
import matplotlib.pyplot as plt
from CoSi.plotting_utils import plot_panel

from dataclasses import dataclass
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--outputs_file", type=str, default="outputs/nico_plus_plus/amco_95/seed_0/all_outputs.npy")
parser.add_argument(
    "--stopping_time", type=int, default=90
)
parser.add_argument(
    "--dataset_name", type=str, default='nico_plus_plus_super_95'
)

args = parser.parse_args()

oracle = False
gen_captions = True


def label_to_class_mapping(dataset):
    if dataset == 'waterbirds':
        classes = {0: 'Landbird', 1:'Waterbird'}
    elif 'nico_plus_plus_super' in dataset:
        classes = {0: "mammals", 1:"birds",2:"plants",3:"airways", 4:"waterways", 5:"landways"}
    elif 'nico_plus_plus' in dataset:
        classes = {0: "giraffe", 1: "cow", 2: "sheep", 3:"dog", 4:"horse", 5:"bird"}
    else:
        classes = {0:'blonde', 1:'black hair'}
    return classes

def get_data_panel(all_outputs, split_info, split='source-train', step=-1, dataset='waterbirds', class_id=-1):
    outputs = all_outputs[split]
    ids = outputs['ids'][step]
    classes = label_to_class_mapping(dataset)
    contexts = np.take_along_axis(outputs['ctxs'][step], ids, axis=0)
    labels = np.take_along_axis(outputs['labs'][step], ids, axis=0)
    paths = np.load(os.path.join(split_info, f'{dataset}_{split.replace("-eval","")}.npy'), allow_pickle=True).item()['paths']
    outs = np.take_along_axis(outputs['outputs'][step], np.expand_dims(ids, 1), axis=0)
    classes_list = [classes[l] for l in labels]
    dp = mk.DataPanel({'contexts': contexts, 'label':classes_list, 'img_path':paths, 'img':  mk.ImageColumn.from_filepaths(paths), 'label_idx': labels, 'input': mk.ImageColumn.from_filepaths(paths), 'probs': outs, 'pred': outs.argmax(-1), 'label_idx': labels })
    if class_id != -1:
        dp = dp[dp['label_idx'] == class_id]
    dp['target'] = dp['label_idx']
    return dp


exps = [[args.dataset_name, args.outputs_file, args.stopping_time]]
params =  [[25, 1e-3]]
full_exps = list(product(exps, params))


@dataclass
class Slice:
    images: List[str]
    captions: List[str]
    keywords: Counter


precs = defaultdict(lambda: defaultdict(list))

if gen_captions:
    captioner = CaptionModel()

for exp, params  in full_exps:
    weight2, reg_covar2 = params
    dataset, scores_file, step = exp
    classes = len(label_to_class_mapping(dataset))
    for class_id in range(classes):
        all_outputs = np.load(scores_file, allow_pickle=True).item()
        split_info = 'data/metadata'
        dp_train = get_data_panel(all_outputs,  split_info,  split='source-train-eval', dataset=dataset,class_id=class_id,  step=step)
        dp_val = get_data_panel(all_outputs, split_info, split='val_env-test', dataset=dataset, class_id=class_id, step=step)
        dp_test = get_data_panel(all_outputs, split_info, split='target-test', dataset=dataset, class_id=class_id, step=step)

        dp_val = embed(
            dp_val, 
            input_col="img",
            encoder="clip", 
            modality='image',
            device='cuda'
        )

        dp_train = embed(
            dp_train, 
            input_col="img",
            encoder="clip", 
            modality='image',
            device='cuda'
        )

        dp_test = embed(
            dp_test, 
            input_col="img",
            encoder="clip", 
            modality='image',
            device='cuda'
        )

        cosi = CoSiSlicer(
            y_log_likelihood_weight=0,
            y_hat_log_likelihood_weight=0,
            y_hat_prob_log_likelihood_weight=weight2,
            n_mixture_components=36,
            n_slices=10,
            tol=1e-7,
            init_params='confusion',
            confusion_noise=0,
            num_classes=classes,
            random_state=0,
            covariance_type_y_hat='full',
            reg_covar_y_hat=reg_covar2
        )

        dp_fit = dp_val

        identity = np.eye(6)
        if oracle:
            dp_train['probs'] = [identity[c] for c in dp_train['contexts']]
            dp_val['probs'] = [identity[c] for c in dp_val['contexts']]
            dp_test['probs'] = [identity[c] for c in dp_test['contexts']]


        # Fit the mixture model on the validation set
        cosi.fit(data=dp_fit, embeddings="clip(img)", targets="target", pred_probs="probs")


        def get_max(batch: mk.DataPanel):
            max_prob = np.max(batch['domino_slices'], axis=-1)
            return {
                "max_prob": max_prob,
                "slice_idx": np.argmax(batch['domino_slices'], axis=-1)
            }

        dp_evals = [('test', dp_test)]
        for name, dp_eval in dp_evals:
            # Predict slice probabilities
            dp_eval["domino_slices"] = cosi.predict_proba(
                data=dp_eval, embeddings="clip(img)", targets="target", pred_probs="probs"
            )
            # Assign each sample to a slice
            dp_eval['domino_slice_idx'] = cosi.predict(data=dp_eval, embeddings="clip(img)", targets="target", pred_probs="probs")


            dp_eval = dp_eval.update(
                function=get_max,
                is_batched_fn=True,
                batch_size=32,
                input_columns=["domino_slices"], 
                pbar=True
            )

            n_classes = dp_eval['label_idx'].max() + 1
            dp_eval['group'] = dp_eval['contexts'] * n_classes + dp_eval['label_idx']
            dp_subset = dp_eval

            counts = {i:0 for i in range(36)}

            class_wise_precs = []
            for cls in [class_id]:
                for ctx in range(classes):
                    if dataset != 'celeba' and cls == ctx:
                        continue
                    if dataset == 'celeba' and (cls != 1 or ctx != 1):
                        continue
                    # For each ground truth slice, find the best matching predicted slice
                    max_prec = 0
                    for i in range(36):
                        sl = dp_subset[dp_subset['slice_idx'] == i].sort('max_prob', ascending=False).head(10)
                        max_prec = max(np.sum((sl['contexts'] == ctx) * (sl['label_idx'] == cls)) / 10, max_prec)
                    class_wise_precs.append(max_prec)

            precs[name][dataset + '_' + str(weight2) + '_' + str(reg_covar2)] += class_wise_precs


            # Get validation accuracies per slice
            accs = []
            for i in range(36):
                dp_group = dp_subset[dp_subset['slice_idx'] == i]
                if len(dp_group) > 0:
                    accs.append((i, (dp_group['label_idx'] == dp_group['pred']).mean()))
            
            # Sort slices based on accuracy
            accs.sort(key=lambda x: x[1])

            # plot
            slices = plot_panel(dp_subset, accs[:6], dataset, split=name, class_idx=class_id)

            # Generate captions
            if gen_captions:
                captions = [[captioner.get_caption(i) for i in s] for s in slices]
                keywords = [captioner.extract_keywords(c) for c in captions]
                slices = [Slice(images, captions[i], keywords[i]) for i, images in enumerate(slices)]
                os.makedirs(f'qual_figs/{dataset}/', exist_ok=True)
                np.save(f'qual_figs/{dataset}/{name}_{class_id}.npy', slices)

    precs_mean = {k: {k1: np.mean(v1) for k1, v1 in v.items()} for k, v in precs.items()}
precs_mean = {k: {k1: np.mean(v1) for k1, v1 in v.items()} for k, v in precs.items()}
print('Precision-at-10: ', pd.DataFrame(precs_mean))
