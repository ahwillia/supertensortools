import torch
import numpy as np
import matplotlib.pyplot as plt
import supertensortools as stt

# Load data.
_X = torch.from_numpy(np.load("X.npy").astype("float32"))
_y = torch.from_numpy(np.load("y.npy").astype("int64"))
num_classes = torch.max(_y).item() + 1

# Specify tensor and dependent variable objects.
X = stt.AnnotatedTensor(
    axes=["trials", "freqs", "timebins"],
    nonneg=[False, False, False],
    name="face_tensor", data=_X, noise="gaussian"
)
y = stt.CategoricalVariable(
    tensor="face_tensor", axis="trials", data=_y, num_classes=num_classes
)

# Create model
alpha = 0.9   # tradeoff between reconstructing tensor and decoding trial labels.
model = stt.TensorModel(Xs=[X], ys=[y], rank=6, wx=[alpha], wy=[1 - alpha])

print("INITIAL LOSSES...")
print("Reconstruction Loss:", model.reconstruction_loss())
print("Decoding Loss:", model.decoding_loss())
print("Total Loss:", model.total_loss())

# Fit model
trace = stt.fit_apg(
    model, patience=100, rtol=1e-3, atol=1e-4, max_iter=1000,
    trace_decoding_loss=True, trace_reconstruction_loss=True
)

print("\nFINAL LOSSES...")
print("Reconstruction Loss:", model.reconstruction_loss())
print("Decoding Loss:", model.decoding_loss())
print("Total Loss:", model.total_loss())

# Generate plots.
fig, ax = plt.subplots(1, 1)
ax.plot(trace["loss"], "-k", label="total loss")
ax.plot(trace["decoding_loss"], "-r", label="decode loss")
ax.plot(trace["reconstruction_loss"], "-b", label="recon loss")
ax.set_xlabel("iterations")
ax.set_ylabel("loss")
ax.legend()

fig, ax = plt.subplots(1, 1)
ax.plot(trace["learning_rate"])
ax.set_xlabel("iterations")
ax.set_ylabel("learning rate")

fig, ax = plt.subplots(1, 1)
ax.plot(trace["momentum"])
ax.set_xlabel("iterations")
ax.set_ylabel("momentum")

plt.show()
