from opt_einsum import contract
import string
import torch
from torch.nn.functional import cross_entropy, mse_loss

class AnnotatedTensorShape:

    def __init__(self, *, axes, dimensions):
        self.axes = axes
        self.dimensions = dimensions

    def __getitem__(self, index):
        if isinstance(index, str):
            index = self.axes.index(index)
        return self.dimensions[index]

class AnnotatedTensor:
    """
    Holds a torch.Tensor object with labeled axes and
    a specified noise model.
    """

    def __init__(self, *, name, data, axes, nonneg, noise="KL"):

        if not isinstance(name, str):
            raise ValueError(
                "'name' should be a string."
            )

        if not isinstance(data, torch.Tensor):
            data = torch.tensor(data)

        if (not isinstance(axes, (list, tuple))) or (data.ndim != len(axes)):
            raise ValueError(
                "'axes' should be a list/tuple of strings, "
                "and 'len(axes)' should equal 'data.ndim'."
            )

        if (not isinstance(nonneg, (list, tuple))) or (data.ndim != len(axes)):
            raise ValueError(
                "'nonneg' should be a list/tuple of True/False values."
            )

        if noise not in ("gaussian", "poisson", "KL"):
            raise ValueError(
                "Specified 'noise' was not recognized."
            )

        self.name = name
        self.data = data
        self.axes = axes
        self.noise = noise
        self.shape = AnnotatedTensorShape(axes=axes, dimensions=data.shape)
        self.ndim = data.ndim
        self.nonneg = nonneg
        self.sqnorm = torch.vdot(
            data.view(-1), data.view(-1)
        )
        if self.ndim > 23:
            raise ValueError("Tensor has too many dimensions.")

    def reconstruction_loss(self, factors):

        if self.noise == "gaussian":
            targ_string = string.ascii_lowercase[:self.ndim]
            fctrs_string = ",".join("z" + c for c in targ_string)
            inner_prod = contract(
                targ_string + "," + fctrs_string + "->", self.data, *factors
            )

            fctrs_sqnrm = torch.sum(torch.prod(
                torch.stack([f @ f.T for f in factors]), axis=0
            ))
            return (self.sqnorm - 2 * inner_prod + fctrs_sqnrm) / self.sqnorm

        elif (self.noise == "poisson") or (self.noise == "KL"):
            targ_string = string.ascii_lowercase[:self.ndim]
            fctrs_string = ",".join("z" + c for c in targ_string)
            Xhat = contract(fctrs_string + "->" + targ_string, *factors)
            return torch.mean(Xhat - self.data * torch.log(Xhat))

        else:
            raise NotImplementedError()


class CategoricalVariable:
    """
    Specifies a Categorical dependent variable.
    """

    def __init__(
            self, *, tensor, axis, data, num_classes, components=None, weighted_loss=True
        ):
        if not isinstance(data, torch.Tensor):
            raise ValueError(
                "'data' must be a torch.Tensor object."
            )
        if data.ndim != 1:
            raise ValueError(
                "'data' must be a 1d array of integer labels."
            )
        if not isinstance(tensor, str):
            raise ValueError(
                "'tensor' should be a string corresponding to the 'name' of an AnnotatedTensor."
            )
        if not isinstance(axis, str):
            raise ValueError(
                "'axis' should be a string corresponding to an axis of an AnnotatedTensor."
            )
        if not isinstance(num_classes, int):
            raise ValueError(
                "'num_classes' should be an integer."
            )
        if torch.max(data) >= num_classes:
            raise ValueError(
                "'data' array has more integer labels than specified by 'num_classes'."
            )
        if components is not None:
            if not isinstance(components, torch.Tensor):
                components = torch.tensor(components)
            if components.ndim != 1:
                raise ValueError(
                    "'components' should be a 1d array of integers."
                )

        self.data = data
        self.tensor = tensor
        self.axis = axis
        self.num_features = num_classes
        self.components = components
        self.weighted_loss = weighted_loss

        if weighted_loss:
            self.weights = torch.tensor(
                [1 / (num_classes * max(1, torch.sum(data == i).item())) for i in range(num_classes)],
                device=data.device
            )
        else:
            self.weights = None

    def decoding_loss(self, logits):
        if self.weighted_loss:
            return cross_entropy(logits, self.data, weight=self.weights, reduction="sum")
        else:
            return cross_entropy(logits, self.data)


class ScalarVariable:
    """
    Specifies a Scalar dependent variable.
    """

    def __init__(self, *, tensor, axis, data, components=None):
        if not instance(data, torch.Tensor):
            raise ValueError(
                "'data' must be a torch.Tensor object."
            )
        if data.ndim > 2:
            raise ValueError(
                "'data' must be a 1d or 2d array."
            )
        if data.ndim == 1:
            data = data[:, None]
        if not isinstance(tensor, str):
            raise ValueError(
                "'tensor' should be a string corresponding to the 'name' of an AnnotatedTensor."
            )
        if not isinstance(axis, str):
            raise ValueError(
                "'axis' should be a string corresponding to an axis of an AnnotatedTensor."
            )
        self.data = data
        self.tensor = tensor
        self.axis = axis
        self.num_features = data.shape[1]

    def decoding_loss(self, pred):
        return mse_loss(pred, self.data)
