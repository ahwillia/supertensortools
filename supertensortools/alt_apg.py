import torch
from tqdm import tqdm
from supertensortools.apg import AcceleratedProxGrad


def fit_alt_apg(
        model, *, patience, rtol=1e-1, atol=1e-4, max_iter=1000,
        trace_decoding_loss=False, trace_reconstruction_loss=False,
        verbose=True, return_learning_rates=False,
        lr_backtrack_factor=0.5, momentum_backtrack_factor=0.9,
        init_lr_readouts=0.001, init_lr_factors=10.0
    ):
    """
    Fits model using acceleration proximal gradient method.
    """

    # Define proximal operators.
    optimizers = [
        AcceleratedProxGrad(
            model._factor_params,
            lr=init_lr_factors, init_momentum=1.0,
            momentum_backtrack_factor=momentum_backtrack_factor,
            lr_backtrack_factor=lr_backtrack_factor
        )
    ] + [
        AcceleratedProxGrad(
            [r for r in model._readout_params] + [b for b in model._bias_params],
            lr=init_lr_readouts, init_momentum=1.0,
            momentum_backtrack_factor=momentum_backtrack_factor,
            lr_backtrack_factor=lr_backtrack_factor
        )
    ]

    # Keep history of optimization progress.
    trace = {
        "converged": False,
        "loss": [],
    }
    if trace_decoding_loss:
        trace["decoding_loss"] = []
    if trace_reconstruction_loss:
        trace["reconstruction_loss"] = []

    if verbose:
        pbar = tqdm(total=max_iter)

    itercount = 0
    while (not trace["converged"]) and (itercount < max_iter):
        
        # Update parameters.
        for opt in optimizers:
            loss = opt.step(
                model.total_loss, # function that evaluates objective.
                model.apply_prox  # function that implement prox operator.
            )
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

        # Check convergence.
        if itercount > patience:
            abs_imp = trace["loss"][-patience] - trace["loss"][-1]
            rel_imp = abs_imp / trace["loss"][-1]
            if abs_imp < atol:
                if verbose:
                    pbar.close()
                    print("Converged: reached absolute tolerance. aImp: ", abs_imp)
                trace["converged"] = True
            if rel_imp < rtol:
                if verbose:
                    pbar.close()
                    print("Converged: reached relative tolerance. rImp: ", rel_imp)
                trace["converged"] = True

    if verbose:
        pbar.close()

    if return_learning_rates:
        return (
            optimizers[0].prxgrad.param_groups[0]["lr"],
            optimizers[1].prxgrad.param_groups[0]["lr"]
        )
    else:
        return trace

