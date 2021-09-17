import matplotlib.pyplot as plt
import torch
import supertensortools as stt

# Create data with 50 neurons, 100 trials, 75 timebins.
#   >> Note that trial_factors are very sparse, so the
#      overall variance explained by these factors is small.
torch.manual_seed(1234)
n_neurons = 50
n_timebins = 75
n_trials = 100
neuron_factor = torch.rand(50)
temporal_factor = torch.exp(
    -torch.linspace(-3, 3, 75) ** 2
)
trial_factor = torch.zeros(100)
# trial_factor[45:55] = 1.0
trial_factor[50:] = 1.0

fig, axes = plt.subplots(1, 3)
axes[0].stem(neuron_factor, use_line_collection=True)
axes[0].set_title("neuron factor")
axes[1].plot(temporal_factor)
axes[1].set_title("temporal factor")
axes[2].plot(trial_factor)
axes[2].set_title("trial factor")

# Create tensor data.
NOISE = 0.5
_X = torch.einsum("i,j,k->ijk", neuron_factor, temporal_factor, trial_factor)
_X += NOISE * torch.randn(*_X.shape)
_X.relu_()

# Create trial labels.
_y = trial_factor.type(torch.int64)

# Specify tensor and dependent variable objects.
X = stt.AnnotatedTensor(
    axes=["neurons", "timebins", "trials"],
    nonneg=["soft", "soft", "soft"],
    name="neural_data", data=_X, noise="poisson"
)
y = stt.CategoricalVariable(
    tensor="neural_data", axis="trials", data=_y, num_classes=2
)

# Fit tensor decomposition model.
alpha = 0.5
model = stt.TensorModel(Xs=[X], ys=[y], rank=1, wx=[alpha], wy=[1 - alpha])

print("INITIAL LOSSES...")
print("Reconstruction Loss:", model.reconstruction_loss())
print("Decoding Loss:", model.decoding_loss())
print("Total Loss:", model.total_loss())

# Fit model
trace = stt.fit_alt_apg(
    model, patience=100, rtol=1e-4, atol=1e-5, max_iter=100,
    trace_decoding_loss=True, trace_reconstruction_loss=True
)

print("\nFINAL LOSSES...")
print("Reconstruction Loss:", model.reconstruction_loss())
print("Decoding Loss:", model.decoding_loss())
print("Total Loss:", model.total_loss())

# Generate plots.
fig, ax = plt.subplots(1, 1)
ax.plot(torch.stack(trace["decoding_loss"]) / trace["decoding_loss"][0], "-r", label="decode loss")
ax.plot(torch.stack(trace["reconstruction_loss"]) / trace["reconstruction_loss"][0], "-b", label="recon loss")
ax.set_xlabel("iterations")
ax.set_ylabel("loss")
ax.legend()

fig, axes = plt.subplots(1, 3)
est_factors = model.get_factors("neural_data")
axes[0].plot(est_factors["neurons"][0].detach())
# axes[0].plot(model.factors["neural_data"]["neurons"][1].detach())
axes[0].set_title("neuron factor")
axes[1].plot(est_factors["timebins"][0].detach())
# axes[1].plot(model.factors["neural_data"]["timebins"][1].detach())
axes[1].set_title("temporal factor")
axes[2].plot(est_factors["trials"][0].detach())
# axes[2].plot(model.factors["neural_data"]["trials"][1].detach())
axes[2].set_title("trial factor")

plt.show()
