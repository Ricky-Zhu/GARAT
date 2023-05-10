from openTSNE import TSNE
import numpy as np
from examples import utils
import matplotlib.pyplot as plt


def tsne_visualize(paths,
                   labels,
                   perplexity=500):
    assert len(paths) == len(labels)
    X = []
    Y = []
    for i in range(len(paths)):
        x = np.load(paths[i])
        if x.ndim == 3:
            x = x.squeeze(1)
        X += list(x)
        Y += [labels[i]] * len(x)
    X = np.asarray(X)
    x_mean = X.mean()
    x_std = X.std()
    X_norm = (X - x_mean) / (x_std + 1e-7)
    tsne = TSNE(
        perplexity=perplexity,
        metric="euclidean",
        n_jobs=8,
        random_state=42,
        verbose=True,
    )
    embedding_x = tsne.fit(X_norm)
    utils.plot(embedding_x, Y)
    plt.show()


if __name__ == "__main__":
    paths = ['/home/ruiqi/projects/GARAT/data/optim_state_visit_HalfCheetah-v2_2500000.npy',
             '/home/ruiqi/projects/GARAT/data/optim_state_visit_HalfCheetahModified-v2_2500000.npy',
             '/home/ruiqi/projects/GARAT/data/models/garat/state_visitation_evaluate__HalfCheetahModified-v2_HalfCheetah-v2-2023-05-03-15-31-44/state_visit_grounding_step_1.npy',
             '/home/ruiqi/projects/GARAT/data/models/garat/state_visitation_evaluate__HalfCheetahModified-v2_HalfCheetah-v2-2023-05-03-15-31-44/state_visit_grounding_step_2.npy',
             '/home/ruiqi/projects/GARAT/data/models/garat/state_visitation_evaluate__HalfCheetahModified-v2_HalfCheetah-v2-2023-05-03-15-31-44/state_visit_grounding_step_3.npy',
             '/home/ruiqi/projects/GARAT/data/models/garat/state_visitation_evaluate__HalfCheetahModified-v2_HalfCheetah-v2-2023-05-03-15-31-44/state_visit_grounding_step_4.npy',
             '/home/ruiqi/projects/GARAT/data/models/garat/state_visitation_evaluate__HalfCheetahModified-v2_HalfCheetah-v2-2023-05-03-15-31-44/state_visit_grounding_step_5.npy']
    labels = ['source', 'target', 'g_1', 'g_2', 'g_3', 'g_4', 'g_5']
    tsne_visualize(paths, labels)
