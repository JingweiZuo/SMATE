import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


class DataPlot:

    def __init__(self):
        self.init_plotting()
        pass

    def init_plotting(self):
        plt.rcParams['figure.figsize'] = (6.5, 5.5)
        plt.rcParams['font.size'] = 15
        #plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
        plt.rcParams['axes.titlesize'] = 20
        plt.rcParams['legend.fontsize'] = 13
        plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
        plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']
        plt.rcParams['savefig.dpi'] = plt.rcParams['savefig.dpi']
        plt.rcParams['xtick.major.size'] = 3
        plt.rcParams['xtick.minor.size'] = 3
        plt.rcParams['xtick.major.width'] = 1
        plt.rcParams['xtick.minor.width'] = 1
        plt.rcParams['ytick.major.size'] = 3
        plt.rcParams['ytick.minor.size'] = 3
        plt.rcParams['ytick.major.width'] = 1
        plt.rcParams['ytick.minor.width'] = 1
        plt.rcParams['axes.linewidth'] = 2


    def tnse_plot(self, path, ds):

        dataset, nclass, n_train, D = ds
        # X = np.empty((0,200), float)
        # X_embedded = TSNE(n_components=2).fit_transform(X)
        # print(X_embedded.shape)
        # color_plate = ['black',
        #                'gold', 'chartreuse', 'deepskyblue',
        #                'purple', 'tomato', 'gainsboro']

        color_plate = ['black', 'red', 'rosybrown', 'tan', 'grey', 
                       'gold', 'olivedrab', 'chartreuse', 'darkgreen', 'deepskyblue',
                       'royalblue', 'navy', 'darkorchid', 'm', 'skyblue',
                       'slateblue', 'y', 'purple', 'tomato', 'gainsboro',
                       'royalblue', 'navy', 'darkorchid', 'm', 'skyblue']

        # load embedding data
        repr_file = path + dataset + '_embeddings.txt'

        X = np.empty((0, D), float)
        colors = []
        with open(repr_file, "r") as f:
            for line in f:
                #print(line)
                results = line.split(',')
                target = int(results[0])
                repr = [float(i) for i in results[1:]]

                colors.append(target)
                X = np.append(X, np.array([repr]), axis=0)

        X_embedded = TSNE(n_components=2).fit_transform(X)

        fig, ax = plt.subplots()
        for i, repr in enumerate(X_embedded):
            x = repr[0]
            y = repr[1]
            if i < nclass:
                scale = 150.0
                color = color_plate[colors[i]]
                edgecolors = 'black'
                zorder = 10
                marker = '*'
            elif i < nclass + n_train:
                scale = 20.0
                color = color_plate[colors[i]]
                edgecolors = 'none'
                zorder = 1
                marker = 'o'
            else:
                scale = 20.0
                color = color_plate[colors[i]]
                edgecolors = 'black'
                zorder = 1
                marker = 'x'
            ax.scatter(x, y, c=color, s=scale, marker=marker,
                       alpha=0.8, edgecolors=edgecolors, zorder=zorder)

        #ax.legend()
        ax.grid(True)
        print("drawing ...")
        plt.savefig(path + dataset + "_embed.pdf")
        #plt.show()

