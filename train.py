from argparse import ArgumentParser, Namespace
from typing import List, Optional
import time
import numpy as np
from tqdm import tqdm

import torch as ch

ch.backends.cudnn.benchmark = False
ch.backends.cudnn.deterministic = True
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn import CrossEntropyLoss
from torch.optim import SGD, lr_scheduler, Adam
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Variable

from fastargs import get_current_config, Param, Section
from fastargs.decorators import param
from fastargs.validation import And, OneOf

import sys
import wandb

from PIL import Image
from models import construct_model

from utils.datasets import get_loaders_for_training

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import os
import random
import json

from dataclasses import dataclass

nico_plus_plus_datasets = [
    "nico_plus_plus_super_" + corr
    for corr in ["50", "75", "90", "95"]
]

DATASETS_WITH_CTXS = [
    "waterbirds",
    "celeba",
    "ffcv_celeba",
    *nico_plus_plus_datasets,
]


def init_configs():
    Section("training", "Hyperparameters").params(
        lr=Param(float, "The learning rate to use", required=True),
        epochs=Param(int, "Number of epochs to run for", default=None),
        steps=Param(int, "Number of steps to run for", default=None),
        batch_size=Param(int, "Batch size", default=512),
        momentum=Param(float, "Momentum for SGD", default=0.9),
        weight_decay=Param(float, "l2 weight decay", default=5e-4),
        num_workers=Param(int, "The number of workers", default=8),
        arch=Param(str, "The architecture to use", default="resnet50"),
        eval_freq=Param(int, "Model evaluation frequency during training", default=100),
        seed=Param(int, "Fix seed for reproducing results", default=0),
        save_dir=Param(
            str,
            "the directory to save logs/checkpoints/visualizations to",
            default="outputs/",
        ),
        exp_name=Param(str, "Experiment identifier", required=True),
        ckpt_save_freq=Param(
            int, "Model checkpoint save frequency (number of steps)", default=2500
        ),
        pretrained=Param(
            bool, "whether to load the pretrained IN1k supervised weights", default=False
        ),
        load_ckpt_path=Param(
            str,
            "Path to load the model checkpoint from; overrides `pretrained` flag",
            default="",
        ),
        n_trials=Param(int, "Number of trials", default=1),
        optimizer=Param(And(str, OneOf(["adam", "sgd"])), default="sgd"),
        eval=Param(bool, "whether to just evaluate a loaded model", default=False),
    )

    Section("data", "data related stuff").params(
        dataset_name=Param(
            And(
                str,
                OneOf(
                    [
                        "waterbirds",
                        "celeba",
                        "ffcv_celeba",
                        *nico_plus_plus_datasets,
                    ]
                ),
            ),
            "Name of dataset",
            default="cifar10",
        ),
        train_dataset=Param(
            str,
            ".dat file to use for training for ffcv datasets, base dataset dir otherwise",
            required=True,
        ),
        train_eval_dataset=Param(
            str,
            ".dat file (ordered) of training samples used for evaluation",
            required=False,
        ),
        val_dataset=Param(
            str, ".dat file (ordered) used for validation", required=False
        ),
        target_dataset=Param(str, ".dat file to use for ood evaluation", default=None),
        num_classes=Param(int, "Number of classes", default=10),
        balance_classes=Param(bool, "whether to balance classes", default=False),
    )


@param("data.dataset_name")
@param("training.batch_size")
@param("data.train_dataset")
@param("training.arch")
def get_dataloaders(
    args,
    dataset_name,
    seed=None,
    balance_contexts=None,
    batch_size=None,
    train_dataset=None,
    arch=None,
):
    if "ffcv" in dataset_name:
        raise NotImplementedError
    elif dataset_name in ["waterbirds", "celeba", *nico_plus_plus_datasets]:
        return get_loaders_for_training(args)


def update(
    model,
    loss_fn,
    ims,
    labs,
    opt,
    scaler,
    scheduler,
):
    opt.zero_grad(set_to_none=True)
    ims = Variable(ims, requires_grad=True)
    with autocast():
        out = model(ims)
        ce_loss = loss_fn(out, labs)
        loss = ce_loss

    scaler.scale(loss).backward()

    scaler.step(opt)
    scaler.update()
    scheduler.step()
    return loss, out


def get_model_stats(model, weight_decay, ce_loss):
    stats_dict = {}
    model_norm_sq = 0.0
    for param in model.parameters():
        model_norm_sq += ch.norm(param) ** 2
    stats_dict["loss/model_norm_sq"] = model_norm_sq.item()
    stats_dict["loss/reg_loss"] = weight_decay / 2 * model_norm_sq.item()
    stats_dict["loss/reg_by_ce"] = stats_dict["loss/reg_loss"] / ce_loss.item()
    stats_dict["loss/ce_loss"] = ce_loss.item()
    return stats_dict


@param("training.lr")
@param("training.epochs")
@param("training.steps")
@param("training.optimizer")
@param("training.momentum")
@param("training.weight_decay")
@param("training.eval_freq")
@param("training.save_dir")
@param("training.exp_name")
@param("training.ckpt_save_freq")
@param("data.dataset_name")
@param("training.batch_size")
def train(
    model,
    loaders,
    lr=None,
    epochs=None,
    steps=None,
    optimizer=None,
    momentum=None,
    weight_decay=None,
    lr_schedule=None,
    eval_freq=None,
    dataset_name=None,
    save_dir=None,
    exp_name=None,
    ckpt_save_freq=None,
    seed=None,
    loaders_len=None,
    start_epoch=0,
    batch_size=64,
):

    to_np_cpu = lambda x: np.copy(x.detach().cpu().numpy())

    assert steps is not None or epochs is not None

    all_outputs = {}

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of parameters (M): %.2f" % (n_parameters / 1.0e6))


    loader_len = loaders_len['source-train']
    print(f"Train loader length: {loader_len}")

    # The total number of steps to train the model for
    if steps is not None:
        total_steps = steps
    else:
        total_steps = loader_len * epochs

    if steps is not None:
        epochs = int(np.ceil(total_steps / loader_len))

    lr_schedule = np.array([1] * (total_steps + 1))

    # file for storing training and evaluation logs
    f = open(
        os.path.join(save_dir, exp_name, f"seed_{seed}", "log.txt"),
        mode="a",
        encoding="utf-8",
    )
    source_loader = loaders['source-train']

    step = start_epoch * len(source_loader)

    if optimizer == "sgd":
        opt = SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
        )
    elif optimizer == "adam":
        opt = Adam(
            model.parameters(), lr=lr, betas=(0.9, 0.999), weight_decay=weight_decay
        )
    else:
        raise NotImplementedError
    scheduler = lr_scheduler.LambdaLR(opt, lr_schedule.__getitem__)

    scaler = GradScaler()

    loss_fn = CrossEntropyLoss(reduce=True)

    log_stats = {}
    if dataset_name in DATASETS_WITH_CTXS:
        best_val_acc = 0

    # Training loop
    for epoch in range(start_epoch, epochs):
        print("Epoch: ", epoch)
        idx = 0
        training_labels = []
        model.train()

        # Evaluate every epoch if eval_freq = -1
        if eval_freq == -1:
            eval_freq = len(source_loader)

        for it, data_iter in tqdm(enumerate(source_loader)):
            data = (
                data_iter[0].to(device="cuda").half(),
                data_iter[1].to(device="cuda"),
            )
            model.train()

            ims, labs = data[0], data[1]

            loss, out = update(
                model,
                loss_fn,
                ims,
                labs,
                opt,
                scaler,
                scheduler,
            )

            model_stats = get_model_stats(model, weight_decay, loss)
            log_stats = {
                "step": step,
                "train_loss": loss.item(),
            }
            log_stats.update(model_stats)

            log_stats["lr/lr"] = scheduler.optimizer.param_groups[0]["lr"]

            # Evaluate at start, end and at intermediate points with given frequency
            if step == 0 or (step + 1) % eval_freq == 0 or step + 1 == total_steps:
                only_train = (step + 1) % eval_freq == 0
                eval_stats = evaluate(
                    model,
                    loaders,
                    step=step,
                    all_outputs=all_outputs,
                )
                if dataset_name in DATASETS_WITH_CTXS and not only_train:
                    ckpt_select_key = "perf/val_env-test_acc"
                    if eval_stats[ckpt_select_key] > best_val_acc:
                        best_val_acc = eval_stats[ckpt_select_key]
                        ch.save(
                            model.state_dict(),
                            os.path.join(
                                save_dir,
                                exp_name,
                                f"seed_{seed}",
                                "checkpoints",
                                "checkpoint-best.pth",
                            ),
                        )
                log_stats.update(eval_stats)

            # Save stats to file and wandb
            if len(log_stats) > 0:
                f.write(json.dumps(log_stats) + "\n")
            wandb.log(log_stats, step=step)

            # Checkpoint model
            if (step + 1) % ckpt_save_freq == 0 or step + 1 == total_steps:
                ch.save(
                    model.state_dict(),
                    os.path.join(
                        save_dir,
                        exp_name,
                        f"seed_{seed}",
                        "checkpoints",
                        "checkpoint-%04d.pth" % (step),
                    ),
                )

            step += 1

            # If max number of steps are reached, then stop
            if step >= total_steps:
                break

        # save once in a while
        if epoch % 10 == 0:
            np.save(
                os.path.join(save_dir, exp_name, f"seed_{seed}", "all_outputs.npy"),
                all_outputs,
            )
        # If max number of steps are reached, then stop
        if step >= total_steps:
            break

    # save model predictions at all steps after training finishes
    np.save(
        os.path.join(save_dir, exp_name, f"seed_{seed}", "all_outputs.npy"),
        all_outputs,
    )
    return log_stats


@param("data.dataset_name")
def evaluate(
    model,
    loaders,
    step=None,
    all_outputs=None,
    dataset_name=None,
):
    stats = {}
    model.eval()

    to_np_cpu = lambda x: np.copy(x.detach().cpu().numpy())
    keys = list(loaders.keys())
    if "source-train-eval" in keys:
        keys.remove("source-train")
    for name in keys:
        # save per-sample model predictions
        correct, outputs, gt, ctxs, ids, top1_confs, conf_gts = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )
        idx = 0
        loader = loaders[name]
        for data_idx, data in tqdm(enumerate(loader)):
            ids.append(to_np_cpu(data[2]))
            if dataset_name in DATASETS_WITH_CTXS:
                ctxs.append(to_np_cpu(data[3]))
            data = (
                data[0].to(device="cuda"),
                data[1].to(device="cuda"),
            )  

            ims, labs = data[0].half(), data[1]
            ims = Variable(ims, requires_grad=True)
            with autocast():
                with ch.no_grad():
                    out = model(ims)
                    preds = F.softmax(out, dim=-1)
                    correct.append(to_np_cpu(out.argmax(1).eq(labs)))
                    top1_confs.append(to_np_cpu(preds.max(1)[0]))
                    conf_gts.append(
                        to_np_cpu(preds.gather(1, labs.view(-1, 1)).view(-1))
                    )
                    outputs.append(to_np_cpu(preds))
                    gt.append(to_np_cpu(labs))

        outputs = np.concatenate(outputs)
        labs = np.concatenate(gt)
        ids = np.concatenate(ids)
        top1_confs = np.concatenate(top1_confs)
        conf_gts = np.concatenate(conf_gts)
        if dataset_name in DATASETS_WITH_CTXS:
            ctxs = np.concatenate(ctxs)
        else:
            ctxs = None
        correct = np.concatenate(correct)
        total_correct = correct.sum().item()
        total_num = len(correct)
        accuracy = total_correct / total_num * 100
        stats.update({"perf/" + name + "_acc": accuracy, "step": step})
        print(f"{name} accuracy at step {step}: {accuracy:.1f}%")


        # Compute group-wise accuracies if dataset has group information
        if dataset_name in DATASETS_WITH_CTXS:
            # compute worst-test group accuracy
            if "nico_plus_plus" in dataset_name:
                n_ctxs = 6
            else:
                n_ctxs = 2
            if "nico_plus_plus" in dataset_name:
                n_cls = 6
            else:
                n_cls = 2

            contexts = np.reshape(np.arange(n_ctxs), (1, -1, 1))
            classes = np.reshape(np.arange(n_cls), (1, 1, -1))
            ctxs_reshape = np.reshape(ctxs, (-1, 1, 1))
            labels = np.reshape(labs, (-1, 1, 1))
            correct_preds = np.reshape(correct, (-1, 1, 1))
            grouped = (ctxs_reshape == contexts) & (classes == labels)
            grouped_total = np.sum(grouped, axis=0)
            grouped_correct = np.sum(grouped & correct_preds, axis=0)
            grouped_acc = 100 * grouped_correct / grouped_total
            worst_group_acc = np.min(grouped_acc)
            grouped_accs = {"worst_group": worst_group_acc}
            if "nico_plus_plus" in dataset_name:
                majority_acc = np.mean(np.diagonal(grouped_acc))
                off_diagonal_mask = 1 - np.eye(n_cls)
                minority_acc = np.sum(off_diagonal_mask * grouped_acc) / np.sum(
                    off_diagonal_mask
                )
                grouped_accs["avg_minority_acc"] = minority_acc
                grouped_accs["avg_majority_acc"] = majority_acc
            for (k, val) in grouped_accs.items():
                stats.update({f"{k}/{name}_{k}_acc": val, "step": step})

            for i in range(n_ctxs):
                for j in range(n_cls):
                    stats.update(
                        {
                            f"group_wise_acc/{name}_ctx_{i}_cls_{j}_acc": grouped_acc[
                                i
                            ][j],
                            "step": step,
                        }
                    )

        eval_info = EvalInfo(outputs, labs, ids, top1_confs, conf_gts, ctxs)
        keys = EvalInfo.__dict__["__dataclass_fields__"].keys()

        if all_outputs is not None:
            update_outputs(all_outputs, name, step, eval_info)

    return stats


def update_outputs(all_outputs, split, step, eval_info):
    keys = EvalInfo.__dict__["__dataclass_fields__"].keys()
    if split not in all_outputs:
        all_outputs[split] = {"eval_steps": [step]}
        for key in keys:
            value = getattr(eval_info, key)
            if value is not None:
                all_outputs[split].update({key: np.expand_dims(value, 0)})
    else:
        all_outputs[split]["eval_steps"].append(step)
        for key in ["outputs", "labs", "ids", "top1_confs", "conf_gts", "ctxs"]:
            value = getattr(eval_info, key)
            if value is not None:
                all_outputs[split][key] = np.append(
                    all_outputs[split][key], [value], axis=0
                )


@dataclass
class EvalInfo:
    outputs: np.array
    labs: np.array
    ids: np.array
    top1_confs: np.array
    conf_gts: np.array
    ctxs: Optional[np.array] = None


def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    ch.manual_seed(seed)


if __name__ == "__main__":
    init_configs()
    config = get_current_config()
    parser = ArgumentParser(description="Fast CIFAR-10 training")
    config.augment_argparse(parser)
    # Also loads from args.config_path if provided
    config.collect_argparse_args(parser)
    config.validate(mode="stderr")
    config.summary()
    arguments = config.get()

    metrics = {}
    combined_args = Namespace(
        **{
            k: v
            for subspace in vars(arguments).values()
            for k, v in vars(subspace).items()
        }
    )

    # repeat experiment with different seeds
    for trial in range(arguments.training.n_trials):
        base_dir = os.path.join(
            arguments.training.save_dir,
            arguments.training.exp_name,
            f"seed_{arguments.training.seed + trial}",
        )
        os.makedirs(os.path.join(base_dir, "checkpoints"), exist_ok=True)

        config.summary(open(os.path.join(base_dir, "config.txt"), "w"))

        # create an empty file for storing accuracy logs
        _ = open(os.path.join(base_dir, "log.txt"), "w")
        wandb.init(
            project="facts",
            group=arguments.data.dataset_name,
            name=f"{os.path.basename(arguments.training.save_dir)}_{arguments.training.exp_name}_{arguments.training.seed + trial}",
        )
        wandb.configs = vars(combined_args)
        fix_seeds(arguments.training.seed + trial)

        loaders, loaders_len, start_time, corrupted_labels = get_dataloaders(
            combined_args, seed=arguments.training.seed + trial
        )
        model = construct_model(
            arguments.training.arch,
            arguments.data.num_classes,
            load_ckpt_path=arguments.training.load_ckpt_path,
            pretrained=arguments.training.pretrained,
        )

        start_epoch = 0

        # Evaluate loaded model
        if arguments.training.eval:
            all_outputs = {}
            eval_stats = evaluate(model, loaders, step=-1, all_outputs=all_outputs)
            np.save(
                os.path.join(
                    arguments.training.save_dir,
                    arguments.training.exp_name,
                    f"seed_{0}",
                    "all_outputs.npy",
                ),
                all_outputs,
            )
            sys.exit(1)

        # If starting from a pretrained checkpoint, set start_epoch correctly
        if arguments.training.load_ckpt_path != "":
            path = arguments.training.load_ckpt_path
            try:
                trained_steps = int(path.split(".")[0].split("-")[-1])
                source_loader = loaders[arguments.data.train_loader_key]
                if isinstance(source_loader, list):
                    source_loader = zip(*source_loader)
                start_epoch = trained_steps // len(source_loader)
            except:
                pass

        # train
        stats = train(
            model,
            loaders,
            seed=arguments.training.seed + trial,
            loaders_len=loaders_len,
            start_epoch=start_epoch,
        )

        # record metrics per seed
        for k, v in stats.items():
            if k in metrics:
                metrics[k].append(stats[k])
            else:
                metrics[k] = [stats[k]]

        print(f"Total time: {time.time() - start_time:.5f}")
        wandb.finish()

    # save metrics averaged across seeds
    average_stats = {k: (np.mean(v), np.std(v)) for k, v in metrics.items()}
    if len(average_stats) > 0:
        with open(
            os.path.join(
                arguments.training.save_dir, arguments.training.exp_name, "log.txt"
            ),
            mode="a",
            encoding="utf-8",
        ) as f:
            f.write(json.dumps(average_stats) + "\n")
