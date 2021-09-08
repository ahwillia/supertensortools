import string
import numpy as np
import torch
from torch import nn
from opt_einsum import contract
from supertensortools.tensors import AnnotatedTensor, CategoricalVariable, ScalarVariable


class TensorModel(nn.Module):

    def __init__(self, *, Xs, rank, wx, ys=None, wy=None, shared_axes=[]):

        # Initialize nn.Module
        super(TensorModel, self).__init__()

        # Check that Xs is a list/tuple.
        if not isinstance(Xs, (list, tuple)):
            raise ValueError(
                "Argument 'Xs' should be a list/tuple of "
                "AnnotatedTensor objects."
            )
        
        # Check that all Xs are AnnotatedTensor objects.
        if not all([isinstance(v, AnnotatedTensor) for v in Xs]):
            raise ValueError(
                "Each element of 'Xs' should be an "
                "AnnotatedTensor object."
            )

        # Build list of tensor names.
        self.tensor_names = [X.name for X in Xs]

        # Check that there are no duplicate tensor names.
        if len(self.tensor_names) != len(set(self.tensor_names)):
            raise ValueError(
                "AnnotatedTensor objects in 'Xs' have duplicate "
                "tensor names."
            )

        # Build list of axis names.
        self.all_axes = []
        for X in Xs:
            self.all_axes += X.axes
        self.all_axes = sorted(list(np.unique(self.all_axes)))

        # Check shared_axes are appropriate.
        if not isinstance(shared_axes, (list, tuple)):
            raise ValueError(
                "'shared_axes' should be a list/tuple of strings."
            )
        for axis in shared_axes:
            if axis not in self.all_axes:
                raise ValueError(
                    f"'shared_axes' specified '{axis}' as shared, but "
                    f"'{axis}' is not a named axis on any AnnotatedTensor."
                )

        # Check that ys is None or a list/tuple.
        if (ys is not None) and (not isinstance(ys, (list, tuple))):
            raise ValueError(
                "Argument 'ys' should be a list/tuple of "
                "CategoricalVariable or ScalarVariable objects."
            )

        # Check that all ys are categorical or scalar variable objects.
        if not all([isinstance(v, (CategoricalVariable, ScalarVariable)) for v in ys]):
            raise ValueError(
                "Each element of 'ys' should be a CategoricalVariable "
                "or ScalarVariable object."
            )

        # Check that ys have appropriate axis names and values.
        for y in ys:
            if y.tensor not in self.tensor_names:
                raise ValueError(
                    f"Specified tensor name '{y.tensor}' for dependent "
                    "variable does not appear in list of AnnotatedTensors."
                )
            elif y.axis not in (Xs[self.tensor_names.index(y.tensor)].axes):
                raise ValueError(
                    f"Specified axis name '{y.axis}' does not appear "
                    f"in axes of tensor '{y.tensor}'."
                )

        # Check inputs.
        if not isinstance(rank, int):
            raise ValueError(
                "'rank' should be an int."
            )
        if not isinstance(wx, (list, tuple)):
            raise ValueError(
                "'wx' must be a list or tuple of floats."
            )
        if len(wx) != len(Xs):
            raise ValueError(
                "'wx' must be the same length as 'Xs'."
            )
        if (ys is not None) and (not isinstance(wy, (list, tuple))):
            raise ValueError(
                "'ys' was specified, so 'wy' must be "
                "a list or tuple of floats."
            )
        if (ys is not None) and (len(wy) != len(ys)):
            raise ValueError(
                "'wy' must be the same length as 'ys'."
            )

        # Store data
        self.wx = wx
        self.wy = wy
        self.Xs = Xs
        self.ys = ys
        self.tensor_names = [X.name for X in Xs]

        # Initialize list of factor matrix parameters.
        self._factor_params = torch.nn.ParameterList([])

        # Initialize shared factor matrices.
        self.shared_axes = shared_axes
        self.shared_factors = dict()
        for axis in shared_axes:
            dims = []
            for X in Xs:
                if axis in X.axes:
                    dims.append(X.shape[axis])
            if len(dims) == 0:
                warnings.warn(f"Shared axis '{axis}' doesn't appear.")
                continue
            elif np.unique(dims).size != 1:
                raise ValueError(
                    f"Shared axis '{axis}' has inconsistent dimension "
                    "across tensors."
                )
            else:
                self.shared_factors[axis] = len(self._factor_params)
                self._factor_params.append(nn.Parameter(
                    torch.randn(rank, dims[0])
                ))

        # Initialize non-shared factor matrices.
        self.factors = dict()
        for X in Xs:
            self.factors[X.name] = dict()
            for axis in X.axes:
                if axis in self.shared_factors.keys():
                    self.factors[X.name][axis] = self.shared_factors[axis]
                else:
                    self.factors[X.name][axis] = len(self._factor_params)
                    self._factor_params.append(nn.Parameter(
                        torch.randn(rank, X.shape[axis])
                    ))

        # For nonnegative factors, project onto positive orthant.
        self.apply_prox()

        # Rescale the factors to match the norm of the data.
        with torch.no_grad():
            
            # First, scale shared factors.
            mean_sqnorm = sum([X.sqnorm for X in Xs]) / len(Xs)
            mean_ndim = sum([X.ndim for X in Xs]) / len(Xs)
            for axis in shared_axes:
                fctr = self._factor_params[self.shared_factors[axis]]
                fctr /= torch.norm(shared_factors[axis])
                fctr *= mean_sqnorm ** (1 / mean_ndim) 

            # Then, scale non-shared factors.
            for X in Xs:
                fctrs = [self._factor_params[i] for i in self.factors[X.name].values()]
                est_sqnorm = torch.sum(torch.prod(
                    torch.stack([f @ f.T for f in fctrs]), axis=0
                ))
                _n = sum([(axis not in self.shared_axes) for axis in X.axes])
                for axis, index in self.factors[X.name].items():
                    if axis not in self.shared_axes:
                        fctr = self._factor_params[index]
                        fctr *= (X.sqnorm / est_sqnorm) ** (1 / _n)

        # Initialize readout weights and biases.
        if ys is None:
            self.readouts = None
            self.biases = None
        else:
            self.readouts = nn.ParameterList([])
            self.biases = nn.ParameterList([])
            for y in ys:
                if y.components is None:
                    self.readouts.append(nn.Parameter(
                        1e-4 * torch.randn(y.num_features, rank)
                    ))
                    self.biases.append(nn.Parameter(
                        torch.zeros(y.num_features)
                    ))
                else:
                    self.readouts.append(nn.Parameter(
                        1e-4 * torch.randn(y.num_features, len(y.components))
                    ))
                    self.biases.append(nn.Parameter(
                        torch.zeros(y.num_features)
                    ))

    @torch.no_grad()
    def apply_prox(self):
        for X in self.Xs:
            for i, axis in enumerate(X.axes):
                if X.nonneg[i]:
                    fctr = self._factor_params[self.factors[X.name][axis]]
                    fctr.relu_()

    def reconstruction_loss(self):
        loss = torch.tensor(0.)
        for X, w in zip(self.Xs, self.wx):
            fctrs = []
            for axis in X.axes:
                fctrs.append(self._factor_params[self.factors[X.name][axis]])
            loss += w * X.reconstruction_loss(fctrs)
        return loss

    def decoding_loss(self):
        loss = torch.tensor(0.)
        if self.ys is not None:
            for y, w, A, b in zip(self.ys, self.wy, self.readouts, self.biases):
                
                # Find the tensor associated with dependent variable y.
                X = self.Xs[self.tensor_names.index(y.tensor)]

                # Find low-rank factors, skipping axis that is being predicted.
                fctr_idx = [self.factors[y.tensor][axis] for axis in X.axes if (axis != y.axis)]
                fctrs = [self._factor_params[i] for i in fctr_idx]

                # Optionally, restrict our prediction to a limited number of components.
                if y.components is not None:
                    fctrs = [torch.index_select(f, 0, y.components) for f in fctrs]

                # Tensor contract over non-predicted axes. For example, if X is a 4d
                # array and we are predicting the axis in the second position, the
                # einsum contraction string is "abcd,za,zc,zd->bz", where 'z' is the
                # index over the model rank and 'b' is the index which is predicted.
                xidx = string.ascii_lowercase[:X.ndim]
                lridx = ["z" + c for c in xidx]
                rhs = lridx.pop(X.axes.index(y.axis))
                lhs = xidx + "," + ",".join(lridx)
                hidden = contract(lhs + "->" + rhs, X.data, *fctrs)

                # Now 'hidden' is an array with shape == (rank, n_obs) or if
                # y.components is specified, shape == (len(y.components), n_obs).
                # We apply an affine transformation to achieve an output with
                # shape (n_obs, num_features). For example, if y is a categorical
                # predicted variable, num_features == num_classes.
                out = (A @ hidden + b[:, None]).T
                loss += w * y.decoding_loss(out)

        return loss

    def total_loss(self):
        return self.reconstruction_loss() + self.decoding_loss()


