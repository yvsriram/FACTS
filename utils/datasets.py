import numpy as np
from tqdm import tqdm
from data.jtt.jtt import get_jtt_datasets
import time
from argparse import Namespace
from torch.utils.data import DataLoader, ConcatDataset, BatchSampler, RandomSampler, WeightedRandomSampler
import torch
from data.nico_plus_plus import get_datasets as get_nico_plus_plus_datasets
import os
from scipy import signal

class DatasetWrapper:
    def __init__(self, base_dataset, dataset, train=False, split=None, encode_strings=False, balance_classes_in_dataset=False):
        self.base_dataset = base_dataset
        self.order = None 
        self.dataset = dataset
        self.encode_strings = encode_strings
        if balance_classes_in_dataset:
            # balance classes by random (static) upsampling
            np.random.seed(42)
            assert self.order is None
            classes, class_freq = np.unique(self.labels, return_counts=True)
            max_count = np.max(class_freq)
            self.order = []
            for i in classes:
                indices = np.where(self.labels == i)[0]
                self.order += indices.tolist()
                rem = max_count - len(indices)
                if rem > 0:
                    self.order += np.random.choice(indices, rem).tolist()
            self.order = np.array(self.order)
        if self.order is None:
            self.order = np.arange(len(self.base_dataset))

        filename = f'data/metadata/{dataset}_{split}.npy'
        if os.path.exists(filename):
            data = np.load(filename, allow_pickle=True).item()
            self.contexts, self.labels, self.paths = data['contexts'], data['labels'], data['paths']
        else:
            self.labels = []
            self.contexts = []
            self.paths = []
            for idx in tqdm(range(self.__len__())):
                data = self.__getitem__(idx)
                self.labels.append(data[1])
                self.contexts.append(data[3])
                self.paths.append(data[4])
            self.labels = np.array(self.labels)
            self.contexts = np.array(self.contexts)
            data = {'contexts':self.contexts, 'labels':self.labels, 'paths':self.paths}
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            np.save(filename, data)

    def __getitem__(self, idx):
        if self.order is not None:
            idx = self.order[idx]

        data = self.base_dataset[idx]
        if len(data) == 2:
            return data

        return data[0], data[1], idx, data[2], np.frombuffer(data[3].encode('utf-8'), dtype='byte') if self.encode_strings else data[3]

    def __len__(self):
        return len(self.order)

def get_wrapped_datasets(args, train_dataset, target_dataset, val_dataset, source_eval_dataset, encode_strings=False):
    for k in ['train_order', 'target_order', 'train_eval_order', 'val_order']:
        setattr(args, k, None)
    datasets = {}
    datasets['source-train'] = DatasetWrapper(train_dataset, 
        dataset=args.dataset_name, train=True, split='source-train', encode_strings=encode_strings, balance_classes_in_dataset=hasattr(args, 'balance_classes_in_dataset') and args.balance_classes_in_dataset,)
    
    if args.train_eval_order is not None:
        datasets['source-train-eval'] = DatasetWrapper(train_dataset, 
        dataset=args.dataset_name, split='source-train-eval', encode_strings=encode_strings)
    
    if source_eval_dataset is not None:
        datasets['source-test'] = DatasetWrapper(source_eval_dataset, 
            dataset=args.dataset_name, split='source-test', encode_strings=encode_strings)
    if val_dataset is not None:
        datasets['val_env-test'] = DatasetWrapper(val_dataset, 
            dataset=args.dataset_name, split='val_env-test', encode_strings=encode_strings)

    datasets['target-test'] = DatasetWrapper(target_dataset, 
            dataset=args.dataset_name, split='target-test', encode_strings=encode_strings)
    
    return datasets

def get_loaders(args, datasets):
    N_WORKERS = 6 * torch.cuda.device_count()
    loaders = {}
    if not args.balance_classes:
        # Shuffled loader
        loaders['source-train'] = DataLoader(
            dataset=datasets['source-train'],
            shuffle=True,
            batch_size=args.batch_size,
            drop_last=False,
            num_workers=N_WORKERS)
    else:
        # Unshuffled loader
        if args.balance_classes:
            labels = datasets['source-train'].labels
            num_classes = np.unique(labels).shape[0]
            n_samples = labels.shape[0]
            sample_weight = np.zeros(n_samples)
            class_counts = np.array([len(np.where(labels == j)[0]) for j in range(num_classes)])
            weights = 1 / class_counts[labels]
            sampler = WeightedRandomSampler(weights, len(weights), replacement = True)
        else:
            sampler = None

        loaders['source-train'] = DataLoader(
            dataset=datasets['source-train'],
            sampler = sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=N_WORKERS)

    for name, dataset in datasets.items():
        if 'source-train'!= name and dataset is not None:
            loaders[name] = DataLoader(
                dataset=dataset,
                batch_size=args.batch_size,
                shuffle=False,
                sampler=None,
                num_workers=N_WORKERS)

    loaders['source-train-eval'] = DataLoader(
                dataset=datasets['source-train'],
                batch_size=args.batch_size,
                shuffle=False,
                sampler=None,
                num_workers=N_WORKERS)

    loaders_len = {name: len(loader) for name, loader in loaders.items()}
    
    return loaders, loaders_len

def get_dataset(args):
    if args.dataset_name in ['waterbirds', 'celeba']:
        train_dataset, val_dataset, target_dataset = get_jtt_datasets(args)
        return [train_dataset,  target_dataset, val_dataset, None]
    elif 'nico_plus' in args.dataset_name:
        train_dataset, val_dataset, target_dataset = get_nico_plus_plus_datasets(args)
        return [train_dataset,  target_dataset, val_dataset, None]

def get_all_loaders(args):
    datasets = get_dataset(args)
    wrapped_datasets = get_wrapped_datasets(args, *datasets)
    # save_context_label_lists(args, wrapped_datasets)
    loaders, loaders_len = get_loaders(args, wrapped_datasets)
    return loaders, loaders_len

def get_loaders_for_training(args):
    start_time = time.time()
    args.shuffle = not args.balance_classes
    loaders, loaders_len = get_all_loaders(args)
    return loaders, loaders_len, start_time, None