import torch
from torch.optim import SGD
from tqdm import tqdm


def fit_open_loop_apgd(
        model, *, patience, lr_lambda, mom_lambda, rtol=1e-1, atol=1e-4, max_iter=1000,
        trace_decoding_loss=False, trace_reconstruction_loss=False,
        verbose=True,
    ):

    # Define optimizer
    optimizer = SGD([
        {
            "params": list(model._factor_params.parameters()),
            "lr": 2.0,
            "momentum": 0.5
        },
        {
            "params": list(model.biases.parameters()) + list(model.readouts.parameters()),
            "lr": 0.001,
            "momentum": 0.0
        },
    ])

    # Keep history of optimization progress.
    trace = {
        "converged": False,
        "loss": [],
        "learning_rate": [],
        "momentum": [],
    }
    if trace_decoding_loss:
        trace["decoding_loss"] = []
    if trace_reconstruction_loss:
        trace["reconstruction_loss"] = []

    if verbose:
        pbar = tqdm(total=max_iter)

    itercount = 0
    while (not trace["converged"]) and (itercount < max_iter):
        
        # Compute loss and take backward step.
        optimizer.zero_grad()
        loss = model.total_loss()
        loss.backward()

        # Take gradient descent step.
        optimizer.step()

        # Apply proximal operator
        model.apply_prox()

        # Store loss.
        trace["loss"].append(loss.item())        

        # Trace additional losses.
        with torch.no_grad():
            if "reconstruction_loss" in trace:
                trace["reconstruction_loss"].append(model.reconstruction_loss())
            if "decoding_loss" in trace:
                trace["decoding_loss"].append(model.decoding_loss())

        # Count iterations.
        itercount += 1
        if verbose:
            pbar.update(1)

        # # Update learning rate and momentum
        # for group in optimizer.param_groups:
        #     group["lr"] = lr_lambda(itercount)
        #     group["momentum"] = mom_lambda(itercount)

        # Check convergence.
        if itercount > patience:
            min_loss = torch.min(trace["loss"][-patience:])
            abs_imp = trace["loss"][-patience] - min_loss
            rel_imp = abs_imp / min_loss
            if abs_imp < atol:
                print("Converged: reached absolute tolerance. aImp: ", abs_imp)
                trace["converged"] = True
            if rel_imp < rtol:
                print("Converged: reached relative tolerance. rImp: ", rel_imp)
                trace["converged"] = True

    if verbose:
        pbar.close()

    return trace
