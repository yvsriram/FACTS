from argparse import Namespace
from data.jtt.confounder_utils import prepare_confounder_data
from data.jtt.dro_dataset import get_loader
import os

def get_jtt_datasets(args):
    if args.dataset_name == 'waterbirds':
        args_new = Namespace(
            root_dir=args.train_dataset,
            target_name=None,
            model=args.arch,
            confounder_names=None,
            augment_data=False,
            metadata_csv_name=os.path.join(args.train_dataset, 'metadata.csv'),
            fraction=1.0,
            dataset='CUB',
            no_transform=args.no_transform if hasattr(args, 'no_transform') else False
        )
    elif args.dataset_name == 'celeba':
        args_new = Namespace(
            root_dir=args.train_dataset,
            target_name="Blond_Hair",
            model=args.arch,
            confounder_names=["Male"],
            augment_data=False,
            metadata_csv_name=os.path.join(args.train_dataset, 'metadata.csv'),
            fraction=1.0,
            dataset='CelebA',
            no_transform=args.no_transform if hasattr(args, 'no_transform') else False
        )
    train_data, val_data, test_data = prepare_confounder_data(
        args_new,
        train=True,
    )
    return train_data, val_data, test_data


if __name__ == '__main__':
    get_jtt_datasets()