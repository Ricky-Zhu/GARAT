from openTSNE import TSNE
import numpy as np
from examples import utils
import matplotlib.pyplot as plt


def tsne_visualize(X,
                   Y,
                   perplexity=30):
    tsne = TSNE(
        perplexity=perplexity,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=True,
    )
    embedding_x = tsne.fit(X)
    utils.plot(embedding_x, Y)
    plt.show()


if __name__ == "__main__":
    hopper_path = '/home/ruiqi/data/optim_state_visit_Hopper-v2_2500000.npy'
    hopper_friction_path = '/home/ruiqi/data/optim_state_visit_HopperFrictionModified-v2_2500000.npy'
    hopper_x = np.load(hopper_path).squeeze(1)
    hopper_friction = np.load(hopper_friction_path).squeeze(1)

    hopper_y = ['hopper'] * len(hopper_x)
    hopper_friction_y = ['hopper_friction'] * len(hopper_friction)

    X = np.concatenate([hopper_x, hopper_friction], axis=0)
    Y = np.asarray(hopper_y + hopper_friction_y)

    tsne_visualize(X, Y)
