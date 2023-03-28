import matplotlib.pyplot as plt
import random
from matplotlib import cm
from matplotlib import axes
# from matplotlib.font_manager import FrontProperties
import numpy as np
import pandas as pd
from pandas import Series,DataFrame
import seaborn as sns
import palettable#python颜色库
from sklearn import datasets 


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Arguments:
        data       : A 2D numpy array of shape (N,M)
        row_labels : A list or array of length N with the labels
                     for the rows
        col_labels : A list or array of length M with the labels
                     for the columns
    Optional arguments:
        ax         : A matplotlib.axes.Axes instance to which the heatmap
                     is plotted. If not provided, use current axes or
                     create a new one.
        cbar_kw    : A dictionary with arguments to
                     :meth:`matplotlib.Figure.colorbar`.
        cbarlabel  : The label for the colorbar
    All other arguments are directly passed on to the imshow call.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Arguments:
        im         : The AxesImage to be labeled.
    Optional arguments:
        data       : Data used to annotate. If None, the image's data is used.
        valfmt     : The format of the annotations inside the heatmap.
                     This should either use the string format method, e.g.
                     "$ {x:.2f}", or be a :class:`matplotlib.ticker.Formatter`.
        textcolors : A list or array of two color specifications. The first is
                     used for values below a threshold, the second for those
                     above.
        threshold  : Value in data units according to which the colors from
                     textcolors are applied. If None (the default) uses the
                     middle of the colormap as separation.

    Further arguments are passed on to the created text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = plt.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[im.norm(data[i, j]) > threshold])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


xLabel = ['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3', 'GPU 4', 'GPU 5', 'GPU 6', 'GPU 7', 'CPU']
yLabel = ['GPU 0', 'GPU 1', 'GPU 2', 'GPU 3', 'GPU 4', 'GPU 5', 'GPU 6', 'GPU 7']

bw = np.array([ [750, 24, 24, 48, 10, 10, 48, 10, 10],
                [24, 750, 48, 24, 10, 10, 10, 48, 10], 
                [24, 48, 750, 48, 24, 10, 10, 10, 10], 
                [48, 24, 48, 750, 10, 24, 10, 10, 10], 
                [10, 10, 24, 10, 750, 48, 24, 48, 10], 
                [10, 10, 10, 24, 48, 750, 48, 24, 10], 
                [48, 10, 10, 10, 24, 48, 750, 24, 10], 
                [10, 48, 10, 10, 48, 24, 24, 750, 10]])

cost_dgl = np.array([  [0, 0, 0, 0, 0, 0, 0, 0, 100],
                            [0, 0, 0, 0, 0, 0, 0, 0, 100], 
                            [0, 0, 0, 0, 0, 0, 0, 0, 100], 
                            [0, 0, 0, 0, 0, 0, 0, 0, 100], 
                            [0, 0, 0, 0, 0, 0, 0, 0, 100], 
                            [0, 0, 0, 0, 0, 0, 0, 0, 100], 
                            [0, 0, 0, 0, 0, 0, 0, 0, 100], 
                            [0, 0, 0, 0, 0, 0, 0, 0, 100]])


cost_gnnlab = np.array([[10, 0, 0, 0, 0, 0, 0, 0, 60],
                        [0, 10, 0, 0, 0, 0, 0, 0, 60], 
                        [0, 0, 10, 0, 0, 0, 0, 0, 60], 
                        [0, 0, 0, 10, 0, 0, 0, 0, 60], 
                        [0, 0, 0, 0, 10, 0, 0, 0, 60], 
                        [0, 0, 0, 0, 0, 10, 0, 0, 60], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0], 
                        [0, 0, 0, 0, 0, 0, 0, 0, 0]])

cost_legion = np.array([[10, 15, 15, 15, 0, 0, 5, 0, 40],
                        [15, 10, 15, 15, 0, 0, 0, 5, 40], 
                        [15, 15, 10, 15, 5, 0, 0, 0, 40], 
                        [15, 15, 15, 10, 0, 5, 0, 0, 40], 
                        [0, 0, 5, 0, 10, 15, 15, 15, 40], 
                        [0, 0, 0, 5, 15, 10, 15, 15, 40], 
                        [5, 0, 0, 0, 15, 15, 10, 15, 40], 
                        [0, 5, 0, 0, 15, 15, 15, 10, 40]])

bw = np.log2(bw)

fig, ((ax, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(8, 6))

# im, _ = sns.heatmap(bw, yLabel, xLabel, ax=ax,
#                    cmap="YlGn", cbarlabel="Bandwidth [log2(GB/s)]")
im, _ = sns.heatmap(bw, yLabel, xLabel, ax=ax,
                   cmap="YlGn", cbar=True,
                    cbar_kws={'label': 'Bandwidth [log2(GB/s)]', #color bar的名称
                                'orientation': 'vertical',#color bar的方向设置，默认为'vertical'，可水平显示'horizontal'
                                "format":"%.3f",#格式化输出color bar中刻度值
                                "pad":0.15,#color bar与热图之间距离，距离变大热图会被压缩
                            },
                    )   
ax.set_title("(a) Hardware Bandwidth Matrix", y = -0.2)

im, _ = sns.heatmap(cost_dgl, yLabel, xLabel, ax=ax2,
                   cmap="YlGnBu", cbar=True,
                    cbar_kws={'label': 'Cost [s]', #color bar的名称
                                'orientation': 'vertical',#color bar的方向设置，默认为'vertical'，可水平显示'horizontal'
                                "ticks":np.arange(0,100,1),#color bar中刻度值范围和间隔
                                "format":"%.3f",#格式化输出color bar中刻度值
                                "pad":0.15,#color bar与热图之间距离，距离变大热图会被压缩
                            },
                    )   
ax2.set_title("(b) DGL", y = -0.2)

im, _ = sns.heatmap(cost_gnnlab, yLabel, xLabel, ax=ax3,
                   cmap="YlGnBu", cbar=True,
                    cbar_kws={'label': 'Cost [s]', #color bar的名称
                                'orientation': 'vertical',#color bar的方向设置，默认为'vertical'，可水平显示'horizontal'
                                "ticks":np.arange(0,60,1),#color bar中刻度值范围和间隔
                                "format":"%.3f",#格式化输出color bar中刻度值
                                "pad":0.15,#color bar与热图之间距离，距离变大热图会被压缩
                            },
                    )   
ax3.set_title("(c) GNNLab", y = -0.2)

im, _ = sns.heatmap(cost_legion, yLabel, xLabel, ax=ax4,
                   cmap="YlGnBu", cbar=True,
                    cbar_kws={'label': 'Cost [s]', #color bar的名称
                                'orientation': 'vertical',#color bar的方向设置，默认为'vertical'，可水平显示'horizontal'
                                "ticks":np.arange(0,40,1),#color bar中刻度值范围和间隔
                                "format":"%.3f",#格式化输出color bar中刻度值
                                "pad":0.15,#color bar与热图之间距离，距离变大热图会被压缩
                            },
                    )                              
ax4.set_title("(d) Legion", y = -0.2)

fig.tight_layout()
plt.show()
