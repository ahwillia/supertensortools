from scipy.io import loadmat 
import numpy as np
import matplotlib.pyplot as plt
import torch
import supertensortools as stt
from tqdm import tqdm

# Y vector
# 1 - hit
# 2 - miss
# 3 - correct rejects
# 4 - false alarm

# List of file names
n_sessions = 26
filenames = [f"/home/alex/data/jerry/pr023/rank_dim_pr023-{i + 1}.mat" for i in range(n_sessions)]

def load_tensor(filename):
    # load activity (neurons x timebins x trials)
    X = np.ascontiguousarray(loadmat(filename)["backupdata"].astype("float32"))
    # normalize each neuron
    X -= np.min(X, axis=(1, 2), keepdims=True)
    X /= np.percentile(X, 99, axis=(1, 2), keepdims=True)
    return stt.AnnotatedTensor(
        name=filename.strip(".mat"),
        data=torch.tensor(X),
        axes=["neurons", "timebins", "trials"],
        nonneg=["soft", "soft", "soft"],
        # nonneg=[False, False, False],
        noise="gaussian"
    )

def load_y(filename):
    _y = loadmat(filename)["Y"].ravel().astype("int64") - 1
    return stt.CategoricalVariable(
        tensor=filename.strip(".mat"),
        axis="trials",
        data=torch.tensor(_y),
        num_classes=4
    )

alpha = 0.5
model = stt.TensorModel(
    Xs=[load_tensor(f) for f in tqdm(filenames)],
    ys=[load_y(f) for f in tqdm(filenames)],
    rank=8,
    wx=[alpha / n_sessions for _ in range(n_sessions)],
    wy=[(1 - alpha) / n_sessions for _ in range(n_sessions)],
    shared_axes=["neurons"],
    shared_readout_axes=["trials"]
)

print("INITIAL LOSSES...")
print("Reconstruction Loss:", model.reconstruction_loss())
_, confusions = model.decoding_loss(return_confusion_matrices=True)
n_correct, n_total = 0, 0
for c in confusions:
    n_correct += torch.diag(c).sum()
    n_total += torch.sum(c)
print("Decoding Acc: ", n_correct / n_total)
print("Total Loss:", model.total_loss())

# trace = stt.fit_tuned_apg(
#     model, patience=100, rtol=1e-4, atol=1e-5, max_iter=5000,
# )
trace = stt.fit_alt_apg(
    model, patience=100, rtol=1e-4, atol=1e-5, max_iter=2500,
)

print("\nFINAL LOSSES...")
print("Reconstruction Loss:", model.reconstruction_loss())
print("Decoding Loss:", model.decoding_loss())
_, confusions = model.decoding_loss(return_confusion_matrices=True)
n_correct, n_total = 0, 0
for c in confusions:
    n_correct += torch.diag(c).sum()
    n_total += torch.sum(c)
print("Decoding Acc: ", n_correct / n_total)
print("Total Loss:", model.total_loss())

fig, axes = plt.subplots(n_sessions, 4, sharex=True)
for i in range(n_sessions):
    for j in range(4):
        axes[i, j].bar(
            np.arange(4), confusions[i][j],
            color=["b" if k == j else "k" for k in range(4)]
        )
        axes[i, j].axhline(0, color="k", lw=1)
        # axes[i, j].axhline(confusions[i][j].sum(), dashes=[2, 2])
        axes[i, j].axis("off")
for j in range(4):
    ymax = max([ax.get_ylim()[1] for ax in axes[:, j]])
    for ax in axes[:, j]:
        ax.set_ylim([-1, ymax])
axes[0, 0].set_title("H")
axes[0, 1].set_title("M")
axes[0, 2].set_title("CR")
axes[0, 3].set_title("FA")
fig.subplots_adjust(hspace=0)


tfctrs = [model.get_factors(f.strip(".mat"))[1].detach() for f in filenames]
fig, axes = plt.subplots(n_sessions, model.rank, sharex=True)
for r in range(model.rank):
    for i, ax in enumerate(axes[:, r]):
        ax.plot(tfctrs[i][r], color=(i/40, 0., 0., .8))
    ymax = max([ax.get_ylim()[1] for ax in axes[:, r]])
    for ax in axes[:, r]:
        ax.set_ylim([0, ymax])
        ax.set_yticks([])
        ax.set_xticks([])
fig.subplots_adjust(hspace=0, wspace=0, bottom=0, top=1, left=0, right=1)
plt.show()
