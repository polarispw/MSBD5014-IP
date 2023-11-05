import os

import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt, colors


class ProfileVisualizer:
    def __init__(self, **kwargs):
        self.save_dir = kwargs.pop("save_dir", "../.profiles/pics")
        if os.path.exists(self.save_dir):
            pass
        else:
            os.makedirs(self.save_dir)

    def save_images(self, images, names, type_prefix):
        if not isinstance(images, list):
            names = f"{type_prefix}_" + names.replace(".", "_")
            images.savefig(f"{self.save_dir}/{names}.png")
        else:
            for image, name in zip(images, names):
                name = f"{type_prefix}_" + name.replace(".", "_")
                image.savefig(f"{self.save_dir}/{name}.png")

    def draw_heatmap(self, profiles, show: bool = True, save: bool = False):
        for name, profile in tqdm(profiles.items()):
            name_idx = ["act_mean", "act_std", "act_max", "act_min", "act_percentile"]
            nr = len(name_idx)
            nc = 3
            fig, axs = plt.subplots(nr, nc, figsize=(25, 16), dpi=300)
            fig.suptitle(name, fontsize=28)
            images = []
            for i in range(nr):
                data = profile[name_idx[i]]
                region_len = data.shape[0] * 3
                selec_list = [(0, region_len),
                              ((data.shape[1] - region_len) // 2, (data.shape[1] + region_len) // 2),
                              (-region_len, -1)]
                for j in range(nc):
                    # select some parts of hidden state since it is too long
                    start, end = selec_list[j]
                    images.append(axs[i, j].imshow(data[:, start:end], aspect='auto'))
                    axs[i, j].set_title(name_idx[i] + f"_{start}_{end}")
                    axs[i, j].set_ylabel("seq_len")
                    axs[i, j].set_xlabel("hidden_state_index")
                    axs[i, j].label_outer()

            # Find the min and max of all colors for use in setting the color scale.
            vmin = min(image.get_array().min() for image in images)
            vmax = max(image.get_array().max() for image in images)

            norm = colors.TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)

            for im in images:
                im.set_norm(norm)

            fig.colorbar(images[0], ax=axs, orientation='horizontal', fraction=.1)

            # Make images respond to changes in the norm of other images (e.g. via the
            # "edit axis, curves and images parameters" GUI on Qt), but be careful not to
            # recurse infinitely!
            def update(changed_image):
                for im in images:
                    if (changed_image.get_cmap() != im.get_cmap()
                            or changed_image.get_clim() != im.get_clim()):
                        im.set_cmap(changed_image.get_cmap())
                        im.set_clim(changed_image.get_clim())

            for im in images:
                im.callbacks.connect('changed', update)

            if show:
                plt.show()

            if save:
                self.save_images(fig, name, "heatmap")

    def violinplot(self, profiles, show: bool = True, save: bool = False):
        for name, profile in tqdm(profiles.items()):
            name_idx = ["act_mean", "act_std", "act_max", "act_min", "act_percentile"]
            nr = len(name_idx)
            nc = 4
            fig, axs = plt.subplots(nr, nc, figsize=(25, 16), dpi=300)
            fig.suptitle(name, fontsize=28)
            images = []
            for i, feature in enumerate(name_idx):
                data = profile[feature]
                data_row_mean = np.mean(data, axis=1)
                data_col_mean = np.mean(data, axis=0)
                data_flatten = data.flatten()
                top1 = np.percentile(data_flatten, 99)
                data_99 = np.where(data_flatten > top1, top1, data_flatten)
                for j, d in enumerate([data_row_mean, data_col_mean, data_flatten, data_99]):
                    images.append(axs[i, j].violinplot(d,
                                                       quantiles=[[0.95, 0.99]],
                                                       showmeans=True))
                    axs[i, j].yaxis.grid(True)
                    # axs[i, j].label_outer()

            pad = 5
            cols = ["hidden_state", "seq_len", "all", "exclude_top1%"]
            rows = name_idx
            for ax, col in zip(axs[0], cols):
                ax.annotate(col, xy=(0.5, 1), xytext=(0, pad),
                            xycoords='axes fraction', textcoords='offset points',
                            size='large', ha='center', va='baseline')

            for ax, row in zip(axs[:, 0], rows):
                ax.annotate(row, xy=(0, 0.5), xytext=(-ax.yaxis.labelpad - pad, 0),
                            xycoords=ax.yaxis.label, textcoords='offset points',
                            size='large', ha='right', va='center')

            if show:
                plt.show()

            if save:
                self.save_images(fig, name, "violinplot")

    def grad_plot(self, profiles, show: bool = True, save: bool = False):
        # plot the gradient matrices
        for name, profile in profiles.items():
            data = profile["grad_val"]
            aspect_ratio = data.shape[0] / data.shape[1]
            fig, axs = plt.subplots(2, 1, figsize=(10, 16), dpi=300)
            fig.suptitle(name, fontsize=28)

            heat_map = axs[0].imshow(data[:, :data.shape[0]], aspect=aspect_ratio)
            axs[0].set_title("grad_val_matrix")

            violinplot = axs[1].violinplot(data.flatten(), quantiles=[[0.95, 0.99]], showmeans=True)
            axs[1].set_title("grad_val_distribution")

            fig.colorbar(heat_map, ax=axs[0], fraction=.1)
            if show:
                plt.show()

            if save:
                self.save_images(fig, name, "grad_plot")

    def animation_plot(self, data):
        ...
