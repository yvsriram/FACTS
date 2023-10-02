## PyTorch implementation of FACTS: First Amplify Correlations and Then Slice to Discover Bias  (ICCV 2023)
### Sriram Yenamandra, Pratik Ramesh, Viraj Prabhu, Judy Hoffman

Computer vision datasets frequently contain spurious correlations between task-relevant labels and (easy to learn) latent task-irrelevant attributes (e.g. context). Models trained on such datasets learn “shortcuts” and underperform on bias-conflicting slices of data where the correlation does not hold. In this work, we study the problem of identifying such slices to inform downstream bias mitigation strategies. We propose First Amplify Correlations and Then Slice to Discover Bias (FACTS), wherein we first amplify correlations to fit a simple bias-aligned hypothesis via strongly regularized empirical risk minimization. Next, we perform correlation-aware slicing via mixture modeling in bias-aligned feature space to discover underperforming data slices that capture distinct correlations. Despite its simplicity, our method considerably improves over prior work (by as much as 35\% precision@10) in correlation bias identification across a range of diverse evaluation settings.

```
@inproceedings{yenamandra2023facts, 
  author = {Yenamandra, Sriram and Ramesh, Pratik and Prabhu, Viraj and Hoffman, Judy},
  title = {FACTS: First Amplify Correlations and Then Slice to Discover Bias},
  year = 2023,
  booktitle = {IEEE/CVF International Conference in Computer Vision (ICCV)}
}
```

### Installation instructions
Create a new conda environment and install requirements using the environment.yml file:
```
conda env create -f environment.yml -n facts
conda activate facts
```

### Download datasets
Download `track_1` of the NICO++ dataset from [here](https://www.dropbox.com/sh/u2bq2xo8sbax4pr/AADbhZJAy0AAbap76cg_XkAfa?dl=0) and place it inside `data/NICO` directory.

### Step-1: Amplify Correlations (AmCo)
To train on $NICO++95$ with high regularization, run:
```
python train.py --config-file configs/nico_plus_plus.yaml --data.dataset_name nico_plus_plus_super_95
```
This will create `all_outputs.npy` inside `outputs/nico_plus_plus/amco_95/seed_0/`. This file contains model predictions at different points in training.

Now, we select the point at which maximum training accuracy peaks.
```
python select_best_point.py --outputs_file outputs/nico_plus_plus/amco_95/seed_0/all_outputs.npy --dataset_name nico_plus_plus_super_95
```

### Step-2: Correlation-aware Slicing (CoSi)
```
python slice.py --outputs_file outputs/nico_plus_plus/amco_95/seed_0/all_outputs.npy --stopping_time 90 --dataset_name nico_plus_plus_super_95
```


This repository uses code from Domino (https://github.com/HazyResearch/domino.git) and JTT (https://github.com/anniesch/jtt).


### License

This project is under the CC-BY-NC 4.0 license. See [LICENSE](LICENSE) for details.