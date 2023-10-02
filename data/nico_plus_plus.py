import random
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import defaultdict
import torchvision.transforms as transforms
from argparse import Namespace


def prepare_data(args):
    dataset_name = args.dataset_name

    train_labels, train_contexts, train_paths = [], [], []
    val_labels, val_contexts, val_paths = [], [], []
    test_labels, test_contexts, test_paths = [], [], []

    samples_in_test = 50
    corr_strength = float(dataset_name.split("_")[-1]) / 100
    held_out_percent = 20
    train_dataset = args.train_dataset
    np.random.seed(41)
    random.seed(41)

    super_classes_plus = {
        "mammals": [
            "sheep",
            "wolf",
            "lion",
            "fox",
            "elephant",
            "kangaroo",
            "cat",
            "rabbit",
            "dog",
            "monkey",
            "squirrel",
            "tiger",
            "giraffe",
            "horse",
            "bear",
            "cow",
        ],
        "birds": ["bird", "owl", "goose", "ostrich"],
        "plants": ["flower", "sunflower", "cactus"],
        "airways": ["hot air balloon", "airplane", "helicopter"],
        "waterways": ["sailboat", "ship", "lifeboat"],
        "landways": [
            "bicycle",
            "motorcycle",
            "train",
            "bus",
            "scooter",
            "truck",
            "car",
        ],
    }
    label_context_plus = {
        "mammals": "rock",
        "birds": "grass",
        "plants": "dim",
        "airways": "outdoor",
        "waterways": "water",
        "landways": "autumn",
    }


    label_map_plus = {
        class_name: class_idx for class_idx, class_name in enumerate(super_classes_plus)
    }
    context_map_plus = {
        label_context_plus[class_name]: class_idx
        for class_idx, class_name in enumerate(super_classes_plus)
    }

    paths = defaultdict(lambda: defaultdict(str))
    majority_group_counts = defaultdict(int)
    train_counts = defaultdict(lambda: defaultdict(int))
    val_counts = defaultdict(lambda: defaultdict(int))

    base_dir = os.path.join(train_dataset, "track_1/public_dg_0416/train")
    for c_idx, context in enumerate(os.listdir(base_dir)):
        if context not in label_context_plus.values():
            print(f"skipping {context}")
            continue
        for s_idx, super_class in enumerate(super_classes_plus):
            imgs_in_group = []
            for c_idx, animal in enumerate(super_classes_plus[super_class]):
                img_paths = os.listdir(f"{base_dir}/{context}/{animal}")
                img_paths = [
                    os.path.join(f"{base_dir}/{context}/{animal}", p)
                    for p in img_paths
                ]
                imgs_in_group += img_paths
            imgs_in_group = sorted(imgs_in_group)
            test_split = np.random.choice(imgs_in_group, samples_in_test)
            for path in test_split:
                test_labels.append(super_class)
                test_contexts.append(context)
                test_paths.append(path)

            remaining = list(set(imgs_in_group).difference(test_split))
            paths[context][super_class] = sorted(remaining)
            if context == label_context_plus[super_class]:
                majority_group_counts[context] = len(remaining)

    for c_idx, context in enumerate(label_context_plus.values()):
        for s_idx, super_class in enumerate(label_context_plus):
            base_dir = os.path.join(
                train_dataset, "track_1/public_dg_0416/train"
            )
            paths_in_group = paths[context][super_class]
            if context != label_context_plus[super_class]:
                num_samples = int(
                    (majority_group_counts[context] * (1 - corr_strength))
                    / (len(label_context_plus) - 1)
                )
                paths_in_group = np.random.choice(paths_in_group, num_samples)
            assert len(paths_in_group) > 0
            train_split, val_split = train_test_split(
                paths_in_group, test_size=0.01 * held_out_percent, shuffle=False
            )
            assert len(val_split) > 0
            if not isinstance(train_split, list):
                train_split = train_split.tolist()
            train_paths += train_split

            train_labels += [super_class] * len(train_split)
            train_contexts += [context] * len(train_split)
            train_counts[super_class][context] = len(train_split)
            if not isinstance(val_split, list):
                val_split = val_split.tolist()
            val_paths += val_split
            val_labels += [super_class] * len(val_split)
            val_contexts += [context] * len(val_split)
            val_counts[super_class][context] = len(val_split)
    return {
        "train": (train_paths, train_labels, train_contexts),
        "val": (val_paths, val_labels, val_contexts),
        "test": (test_paths, test_labels, test_contexts),
        "label_map": label_map_plus,
        "context_map": context_map_plus,
    }


class CustomNICO(Dataset):
    def __init__(self, paths, targets, contexts, label_map, context_map, args):
        self.paths = paths
        self.targets = targets
        self.contexts = contexts
        self.transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        self.label_map = label_map
        self.context_map = context_map

    def read_image(self, path):
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        else:
            img = img.convert("RGB")
            img = np.asarray(img)
            img = np.resize(img, (224, 224, 3))
            img = torch.tensor(img).float() / 255.0
            img = img.moveaxis(2, 0)
        return img

    def __getitem__(self, index):
        image = self.read_image(self.paths[index])
        target = torch.tensor(self.label_map[self.targets[index]])
        context = torch.tensor(self.context_map[self.contexts[index]])
        return image, target, context, self.paths[index]

    def __len__(self):
        return len(self.paths)


def get_datasets(args):
    datasets = prepare_data(args)
    label_map = datasets["label_map"]
    context_map = datasets["context_map"]
    return (
        CustomNICO(*datasets["train"], label_map, context_map, args),
        CustomNICO(*datasets["val"], label_map, context_map, args),
        CustomNICO(*datasets["test"], label_map, context_map, args),
    )


if __name__ == "__main__":
    args = Namespace(dataset_name="", transform=1)
    abcd = get_datasets(args)
