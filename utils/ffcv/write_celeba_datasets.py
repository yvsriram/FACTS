from argparse import ArgumentParser, Namespace
from typing import List
import time
import numpy as np
from tqdm import tqdm

import torch as ch
import torchvision

from fastargs import get_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.writer import DatasetWriter
from ffcv.fields import IntField, RGBImageField, BytesField
import sys
sys.path.extend("../")
from utils.datasets import get_dataset, get_wrapped_datasets

import os

Section("training").params(
    arch=Param(str, default="resnet50"),
)
Section("data", "data related stuff").params(
    dataset_name=Param(
        And(
            str,
            OneOf(
                [
                    "NICOMixed",
                    "cifar10",
                    "NICOPlusPlusMixed",
                    "CorruptedCIFAR10",
                    "waterbirds",
                    "celeba",
                    "PACS",
                ]
            ),
        ),
        "Name of dataset",
        default="celeba",
    ),
    write_path=Param(str, "Folder where the datasets are to be written", default='ffcv_datasets/celeba'),
    train_dataset=Param(str, default="ffcv_celeba"),
    target_order=Param(
        And(str, OneOf(["easy_first", "hard_first"])),
        "the order in \
        which the data is to be generated",
        default="easy_first",
    ),
    rand_label_fraction=Param(
        float, "Fraction of labels to randomize", default=0.0
    ),
    num_classes=Param(int, "Number of classes", default=2),
    train_order=Param(
        str,
        ".npy file containing train sample indices sorted in difficulty \
    order",
        default=None,
    ),
    train_eval_order=Param(str, default=None),
    val_order=Param(
        str,
        ".npy file containing test sample indices sorted in difficulty \
    order",
        default=None,
    ),
    balance_contexts=Param(bool, "whether to balance contexts", default=False),
    balance_classes=Param(bool, "whether to balance classes", default=False),
    use_sub_classes=Param(bool, default=False),
    global_sampling=Param(
        bool, "whether to disregard contexts and sample globally", default=False
    ),
    train_loader_key=Param(
        str, "split to use for training", default="source-train"
    ),
    metadata_file=Param(str, "dataset metadata file", default="metadata.csv"),
)


def main(args):
    args.no_transform=True
    if args.balance_classes:
        args.balance_classes_in_dataset = True
    datasets = get_dataset(args)
    wrapped_datasets = get_wrapped_datasets(args,  *datasets, encode_strings=True)
    import pdb; pdb.set_trace()
    path = args.write_path
    for (name, ds) in wrapped_datasets.items():
        fields = {
            'image': RGBImageField(),
            'label': IntField(),
            'index': IntField(),
            'context': IntField(),
            'path': BytesField()    
        }
        os.makedirs(path, exist_ok=True)
        writer = DatasetWriter(os.path.join(path, f'{name}.beton'), fields)
        writer.from_indexed_dataset(ds)

if __name__ == "__main__":
    config = get_current_config()
    parser = ArgumentParser(description='Fast CIFAR-10 training')
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    arguments = config.get()
    combined_args = Namespace(
        **{
            k: v
            for subspace in vars(arguments).values()
            for k, v in vars(subspace).items()
        }
    )
    main(combined_args)