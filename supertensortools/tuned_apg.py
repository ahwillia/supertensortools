import torch
from tqdm import tqdm
from supertensortools.apg import AcceleratedProxGrad
from supertensortools.alt_apg import fit_alt_apg

def fit_tuned_apg(
        model, *, patience, rtol=1e-1, atol=1e-4, max_iter=1000,
        warm_up_iterations=10, verbose=True,
    ):
    """
    Fits model using acceleration proximal gradient method.
    """

    print("Warming up...")
    lr_factors, lr_readouts = fit_alt_apg(
        model,
        patience=warm_up_iterations,
        max_iter=warm_up_iterations,
        verbose=verbose,
        return_learning_rates=True,
        lr_backtrack_factor=0.1,
        init_lr_readouts=0.001,
        init_lr_factors=100.0
    )

    print(lr_factors)
    print(lr_readouts)

    optimizer = AcceleratedProxGrad(
        [
            {
                "params": model._factor_params,
                "lr": lr_factors,
                "lr_backtrack_factor": 0.5,
                "lr_tuning_patience": 10,
            },
            {
                "params": [r for r in model._readout_params] + [b for b in model._bias_params],
                "lr": lr_readouts,
                "lr_backtrack_factor": 0.5,
                "lr_tuning_patience": 10,
            }
        ],
        init_momentum=1.0,
        momentum_backtrack_factor=0.9,
    )

    # Keep history of optimization progress.
    trace = {
        "converged": False,
        "loss": [],
        "learning_rate": [],
        "momentum": [],
    }

    print("POST-WARMUP LOSSES...")
    with torch.no_grad():
        print("Reconstruction Loss:", model.reconstruction_loss())
        print("Decoding Loss:", model.decoding_loss())
        print("Total Loss:", model.total_loss())

    if verbose:
        pbar = tqdm(total=max_iter)

    itercount = 0
    while (not trace["converged"]) and (itercount < max_iter):
        
        # Update parameters.
        loss = optimizer.step(
            model.total_loss, # function that evaluates objective.
            model.apply_prox  # function that implement prox operator.
        )
        trace["loss"].append(loss.item())
        trace["learning_rate"].append(optimizer.prxgrad.param_groups[0]["lr"])
        trace["momentum"].append(optimizer.param_groups[0]["beta"])

        # Count iterations.
        itercount += 1
        if verbose:
            pbar.update(1)

        # Check convergence.
        if itercount > patience:
            abs_imp = trace["loss"][-patience] - trace["loss"][-1]
            rel_imp = abs_imp / trace["loss"][-1]
            if abs_imp < atol:
                print("Converged: reached absolute tolerance. aImp: ", abs_imp)
                trace["converged"] = True
            if rel_imp < rtol:
                print("Converged: reached relative tolerance. rImp: ", rel_imp)
                trace["converged"] = True

    if verbose:
        pbar.close()

    return trace

