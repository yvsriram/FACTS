import matplotlib.pyplot as plt
from PIL import Image
import os

def plot_panel(dp_subset, sorted_accs, dataset_name, split='test', class_idx=-1, show_labels=False):
    rows = 10 
    all_slices = []
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    fig, axes = plt.subplots(nrows=10, ncols=len(sorted_accs), gridspec_kw = {'wspace':0, 'hspace':0}, figsize=(8/6 * len(sorted_accs),8))
    for i, (s_idx, acc) in enumerate(sorted_accs):
        dp_group = dp_subset[dp_subset['slice_idx'] == s_idx].sort('max_prob', ascending=False).head(10)
        images = list(dp_group['img_path'])
        classes = list(dp_group['label'])
        imgs_in_slice = []
        axes[0][i].set_title("Acc: %0.0f%%" % (100 * acc,))
        for j in range(10):
            if dataset_name == 'celeba':
                resize_dim = (540, 480)
            else:
                resize_dim = (480, 320)
            if j < len(images):
                axes[j][i].imshow(Image.open(images[j]).resize(resize_dim))
                imgs_in_slice.append(images[j])
            axes[j][i].set_xticks([])
            axes[j][i].set_yticks([])
            axes[j][i].set_xticklabels([])        
            axes[j][i].set_yticklabels([])
            if show_labels:
                axes[j][i].set_ylabel(dp_group['label'])
        axes[len(images) - 1][i].set_xlabel(f'Slice #{i + 1}')
        all_slices.append(imgs_in_slice)
    os.makedirs(f'qual_figs/{dataset_name}/', exist_ok=True)
    plt.savefig(f'qual_figs/{dataset_name}/{split}_{class_idx}.png', bbox_inches='tight')
    return all_slices
