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

Code coming soon - please stay tuned!
