import torch
from torch.optim import Optimizer
from torch.optim.optimizer import required
import warnings


def fit_apg(
        model, *, patience, rtol=1e-1, atol=1e-4, max_iter=1000,
        trace_decoding_loss=False, trace_reconstruction_loss=False,
    ):
    """
    Fits model using acceleration proximal gradient method.
    """

    # Define proximal operators.
    optimizer = AcceleratedProxGrad(
        list(model.parameters()),
        lr=1.0, init_momentum=1.0,
        momentum_backtrack_factor=0.9,
        lr_backtrack_factor=0.5,
    )

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

    itercount = 0
    while (not trace["converged"]) and (itercount < max_iter):
        
        # Update parameters.
        trace["learning_rate"].append(optimizer.prxgrad.param_groups[0]["lr"])
        trace["momentum"].append(optimizer.param_groups[0]["beta"])
        loss = optimizer.step(
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

    return trace


class BacktrackingProxGrad(Optimizer):
    """
    Implements Proximal Gradient Method with adaptive / backtracking learning rate.
    """
    def __init__(
            self, params, lr=required,
            lr_backtrack_factor=0.5, max_backtracks=1000,
            lr_tuning_patience=10
        ):

        # Initialize param groups.
        self.max_backtracks = 1000
        self._steps_since_reject = 0
        super().__init__(
            params,
            dict(
                lr=lr,
                lr_backtrack_factor=lr_backtrack_factor,
                lr_tuning_patience=lr_tuning_patience
            ),
        )

        # Add parameter copies to param groups for backtracking.
        for group in self.param_groups:
            group['params_copy'] = [torch.empty_like(p.data) for p in group['params']]

    @torch.no_grad()
    def step(self, closure, prox_operator):

        # Store initial parameter values in case we need to backtrack.
        for group in self.param_groups:
            for p, c in zip(group['params'], group['params_copy']):
                c.copy_(p)

        # Enable gradient calculations, and compute the initial loss.
        with torch.enable_grad():
            self.zero_grad()
            loss = closure()
            loss.backward()

        # Store new loss and number of times learning rate has backtracked.
        new_loss = None
        num_backtracks = 0

        # Main loop.
        while (new_loss is None) or (new_loss > loss):

            # Take gradient step.
            for group in self.param_groups:
                for p in group['params']:
                    p.add_(p.grad, alpha=-group['lr'])

            # Apply proximal operator.
            prox_operator()

            # Evaluate loss (gradients disabled).
            new_loss = closure()

            # Check if loss increased.
            if new_loss > loss:
                
                # Reject parameter update.
                self._steps_since_reject = 0

                # Decrease learning rate, backtrack parameters.
                for group in self.param_groups:
                    group['lr'] *= group['lr_backtrack_factor']
                    for p, c in zip(group['params'], group['params_copy']):
                        p.copy_(c)

                # Quit if backtracking has failed.
                num_backtracks += 1
                if num_backtracks > self.max_backtracks:
                    warnings.warn("Backtracking Line Search Failed...")
                    return loss

            # Otherwise, loss improved.
            else:

                # Increment counter for successful param updates.
                self._steps_since_reject += 1

                # See if we can get away with increasing the learning rate.
                for group in self.param_groups:
                    if self._steps_since_reject > group['lr_tuning_patience']:
                        group['lr'] /= group['lr_backtrack_factor']

                # Finish parameter update.
                return new_loss

        return new_loss


class AcceleratedProxGrad(Optimizer):
    """
    Accelerated Proximal Gradient Method with restarting for nonconvex problems.
    """
    def __init__(
            self, params,
            lr=required, init_momentum=required,
            lr_backtrack_factor=0.5,
            momentum_backtrack_factor=0.8
        ):

        # Create Proximal Gradient optimizer.
        self.prxgrad = BacktrackingProxGrad(
            params,
            lr=lr,
            lr_backtrack_factor=lr_backtrack_factor
        )

        # Initialize param_groups.
        super().__init__(params, dict(
            beta=init_momentum, t=momentum_backtrack_factor))

        # For each group, add a 'beta' momentum parameter
        # and a copy of the parameters for backtracking
        # and momentum extrapolation.
        for group in self.param_groups:
            group['xk'] = [torch.empty_like(p) for p in group['params']]
            group['xk1_m_xk'] = [torch.empty_like(p) for p in group['params']]

    @torch.no_grad()
    def step(self, closure, prox_operator):

        # Set up calculation for extrapolation direction.
        for group in self.param_groups:
            for xk, xk1_m_xk in zip(group['xk'], group['xk1_m_xk']):
                xk1_m_xk.copy_(xk.data)

        # Take a proximal gradient step.
        loss0 = self.prxgrad.step(closure, prox_operator)

        # Save parameters and extrapolate.
        for group in self.param_groups:

            # Store current parameters for next iteration.
            for xk, xk1_m_xk, p in zip(group['xk'], group['xk1_m_xk'], group['params']):
                xk.copy_(p.data)
                xk1_m_xk.sub_(p.data)
            
            # Extrapolation step.
            for xk1_m_xk, p in zip(group['xk1_m_xk'], group['params']):
                p.sub_(xk1_m_xk, alpha=group['beta'])

        # Apply prox operator, then evaluate loss.
        prox_operator()
        loss1 = closure()

        # Choose between extrapolated step and non-extrapolated step.
        if loss1 <= loss0:
            # Extrapolation was accepted, increase momentum.
            for group in self.param_groups:
                group['beta'] = min(1.0, group['beta'] / group['t'])

            return loss1

        # Otherwise, loss0 < loss1, reject momentum update.
        else:
            # Reject extrapolation and decrease momentum.
            for group in self.param_groups:
                group['beta'] *= group['t']

                # Backtrack parameters.
                for xk, p in zip(group['xk'], group['params']):
                    p.copy_(xk.data)

            return loss0
