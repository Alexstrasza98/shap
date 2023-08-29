import numpy as np
import torch

from .._explainer import Explainer


class PyTorchDeep(Explainer):
    def __init__(self, model, data):
        """
        Initialize the model and reference data.

        :param model: A pytorch model that accept a list of inputs, each element inside the list should be a tensor
        :param data: Reference data, it should be a list of tensors, each element has shape [batch_size, ...]
        """
        self.model = model.eval()
        self.data = data

        # check if we have multiple inputs
        self.multi_input = False
        if isinstance(data, list):
            self.multi_input = True
        if not isinstance(data, list):
            data = [data]

        self.expected_value = None  # to keep the DeepExplainer base happy
        self.multi_output = False
        self.num_outputs = 1
        with torch.no_grad():
            outputs = model(*data)

            # also get the device everything is running on
            self.device = outputs.device
            if outputs.shape[1] > 1:
                self.multi_output = True
                self.num_outputs = outputs.shape[1]
            self.expected_value = outputs.mean(0).cpu().numpy()

    def gradient(self, idx, X):
        # Run inference
        self.model.zero_grad()
        outputs = self.model(*X)

        # Get gradients
        selected = [val for val in outputs[:, idx]]
        grads = []
        for idx, x in enumerate(X):
            if x.requires_grad:
                grad = torch.autograd.grad(
                    selected,
                    x,
                    retain_graph=True if idx + 1 < len(X) else None,
                    allow_unused=True,
                )[0]
            else:
                grad = None

            if grad is not None:
                grad = grad.cpu().numpy()
            else:
                grad = torch.zeros_like(X[idx]).cpu().numpy()
            grads.append(grad)

        return grads

    def shap_values(self, X, ranked_outputs=None, output_rank_order="max"):
        """
        Compute shap values referring to specific input samples.

        :param X: A list of inputs that got passed into the model. Each element should have dimensions as [batch_size, ...]
        :param ranked_outputs: If ranked_outputs is None, compute SHAP values for all of the outputs, else compute values for specific number of outputs in ranked order.
        :param output_rank_order: Method to rank the model outputs, should be in ["max", "min", "max_abs"]
        :return: A list of SHAP values for all the output nodes.
        """
        # Ensure that if we have multiple inputs
        if not self.multi_input:
            assert not isinstance(X, list), "Expected a single tensor model input!"
            X = [X]
        else:
            assert isinstance(X, list), "Expected a list of model inputs!"

        # Detach and transfer to same device
        # X = [x.detach().to(self.device) for x in X]

        if ranked_outputs is not None and self.multi_output:
            with torch.no_grad():
                model_output_values = self.model(*X)
            # rank and determine the model outputs that we will explain
            if output_rank_order == "max":
                _, model_output_ranks = torch.sort(model_output_values, descending=True)
            elif output_rank_order == "min":
                _, model_output_ranks = torch.sort(
                    model_output_values, descending=False
                )
            elif output_rank_order == "max_abs":
                _, model_output_ranks = torch.sort(
                    torch.abs(model_output_values), descending=True
                )
            else:
                assert False, "output_rank_order must be max, min, or max_abs!"
            model_output_ranks = model_output_ranks[:, :ranked_outputs]
        else:
            # model_output_ranks is in shape [batch_size, num_outputs], and each row ranges from 0 to num_outputs - 1
            model_output_ranks = (
                torch.ones((X[0].shape[0], self.num_outputs)).int()
                * torch.arange(0, self.num_outputs).int()
            )

        # add the gradient handles
        handles = self.add_handles(self.model, add_interim_values, deeplift_grad)

        # intialize return
        output_phis = []
        # iterate through each output node (i)
        for i in range(model_output_ranks.shape[1]):
            phis = []
            # initialize return value for current output node as same shape to input
            for k in range(len(X)):
                phis.append(np.zeros(X[k].shape))

            # iterate through each sample (j)
            for j in range(X[0].shape[0]):
                # tile the inputs to line up with the background data samples
                # repeat first dimention until the same number of reference, keep other dimensions the same
                tiled_X = [
                    X[l][j : j + 1].repeat(
                        (self.data[l].shape[0],)
                        + tuple([1 for _ in range(len(X[l].shape) - 1)])
                    )
                    for l in range(len(X))
                ]
                joint_x = [
                    torch.cat((tiled_X[l], self.data[l]), dim=0) for l in range(len(X))
                ]
                # get the id for which output node we are explaining
                feature_ind = model_output_ranks[j, i]
                # run attribution computation graph
                sample_phis = self.gradient(feature_ind, joint_x)
                # assign the attributions to the right part of the output arrays
                for l in range(len(X)):
                    phis[l][j] = (
                        (
                            torch.from_numpy(
                                sample_phis[l][self.data[l].shape[0] :]
                            ).to(self.device)
                            * (X[l][j : j + 1] - self.data[l])
                        )
                        .cpu()
                        .detach()
                        .numpy()
                        .mean(0)
                    )
            output_phis.append(phis[0] if not self.multi_input else phis)

        # cleanup; remove all gradient handles
        for handle in handles:
            handle.remove()
        self.remove_attributes(self.model)

        if not self.multi_output:
            return output_phis[0]
        elif ranked_outputs is not None:
            return output_phis, model_output_ranks
        else:
            return output_phis

    def add_target_handle(self, layer):
        input_handle = layer.register_forward_hook(get_target_input)
        self.target_handle = input_handle

    def add_handles(self, model, forward_handle, backward_handle):
        """
        Add handles to all non-container layers in the model
        Recursively for non-container layers

        :param model: The model to add handles to
        :param forward_handle: The forward hook to add
        :param backward_handle: The backward hook to add
        :return: A list of all handles added
        """
        handles_list = []
        model_children = list(model.children())
        if model_children:
            for child in model_children:
                handles_list.extend(
                    self.add_handles(child, forward_handle, backward_handle)
                )
        else:  # leaves
            handles_list.append(model.register_forward_hook(forward_handle))
            handles_list.append(model.register_full_backward_hook(backward_handle))
        return handles_list

    def remove_attributes(self, model):
        """
        Removes the x and y attributes which were added by the forward handles
        Recursively searches for non-container layers

        :param model: The model to remove attributes from
        """
        for child in model.children():
            if "nn.modules.container" in str(type(child)):
                self.remove_attributes(child)
            else:
                try:
                    del child.x
                except AttributeError:
                    pass
                try:
                    del child.y
                except AttributeError:
                    pass


# Module hooks
def deeplift_grad(module, grad_input, grad_output):
    """
    The backward hook which computes the deeplift gradient for an nn.Module

    :param module: The module to compute the gradient for
    :param grad_input: The gradient of the input
    :param grad_output: The gradient of the output
    """
    # get the module type
    module_type = module.__class__.__name__
    # check the module is supported
    if module_type in op_handler:
        if op_handler[module_type].__name__ not in ["passthrough", "linear_1d"]:
            return op_handler[module_type](module, grad_input, grad_output)
    else:
        print(f"Warning: unrecognized nn.Module: {module_type}")
        return grad_input


def add_interim_values(module, input, output):
    """
    The forward hook used to save interim tensors, detached from the graph
    Used to calculate the multipliers

    :param module: The module to add the forward hook to
    :param input: The input to the module
    :param output: The output of the module
    """
    # delete saved tensors if they exist
    try:
        del module.x
    except AttributeError:
        pass
    try:
        del module.y
    except AttributeError:
        pass

    # get module type
    module_type = module.__class__.__name__
    if module_type in op_handler:
        func_name = op_handler[module_type].__name__
        # First, check for cases where we don't need to save the x and y tensors
        if func_name == "passthrough":
            pass
        else:
            # check only the 0th input varies
            for i in range(len(input)):
                if i != 0 and type(output) is tuple:
                    assert input[i] == output[i], "Only the 0th input may vary!"

            # if a new method is added, it must be added here too. This ensures tensors
            # are only saved if necessary
            if func_name in ["maxpool", "nonlinear_1d"]:
                # only save tensors if necessary
                if type(input) is tuple:
                    setattr(module, "x", torch.nn.Parameter(input[0].detach()))
                else:
                    setattr(module, "x", torch.nn.Parameter(input.detach()))
                if type(output) is tuple:
                    setattr(module, "y", torch.nn.Parameter(output[0].detach()))
                else:
                    setattr(module, "y", torch.nn.Parameter(output.detach()))
            if module_type in failure_case_modules:
                input[0].register_hook(deeplift_tensor_grad)


def get_target_input(module, input, output):
    """A forward hook which saves the tensor - attached to its graph.
    Used if we want to explain the interim outputs of a model
    """
    try:
        del module.target_input
    except AttributeError:
        pass
    setattr(module, "target_input", input)


# From the documentation: "The current implementation will not have the presented behavior for
# complex Module that perform many operations. In some failure cases, grad_input and grad_output
# will only contain the gradients for a subset of the inputs and outputs.
# The tensor hook below handles such failure cases (currently, MaxPool1d). In such cases, the deeplift
# grad should still be computed, and then appended to the complex_model_gradients list. The tensor hook
# will then retrieve the proper gradient from this list.


failure_case_modules = ["MaxPool1d"]


def deeplift_tensor_grad(grad):
    return_grad = complex_module_gradients[-1]
    del complex_module_gradients[-1]
    return return_grad


complex_module_gradients = []


def passthrough(module, grad_input, grad_output):
    """No change made to gradients"""
    return None


def maxpool(module, grad_input, grad_output):
    pool_to_unpool = {
        "MaxPool1d": torch.nn.functional.max_unpool1d,
        "MaxPool2d": torch.nn.functional.max_unpool2d,
        "MaxPool3d": torch.nn.functional.max_unpool3d,
    }
    pool_to_function = {
        "MaxPool1d": torch.nn.functional.max_pool1d,
        "MaxPool2d": torch.nn.functional.max_pool2d,
        "MaxPool3d": torch.nn.functional.max_pool3d,
    }
    delta_in = (
        module.x[: int(module.x.shape[0] / 2)] - module.x[int(module.x.shape[0] / 2) :]
    )
    dup0 = [2] + [1 for i in delta_in.shape[1:]]
    # we also need to check if the output is a tuple
    y, ref_output = torch.chunk(module.y, 2)
    cross_max = torch.max(y, ref_output)
    diffs = torch.cat([cross_max - ref_output, y - cross_max], 0)

    # all of this just to unpool the outputs
    with torch.no_grad():
        _, indices = pool_to_function[module.__class__.__name__](
            module.x,
            module.kernel_size,
            module.stride,
            module.padding,
            module.dilation,
            module.ceil_mode,
            True,
        )
        xmax_pos, rmax_pos = torch.chunk(
            pool_to_unpool[module.__class__.__name__](
                grad_output[0] * diffs,
                indices,
                module.kernel_size,
                module.stride,
                module.padding,
                list(module.x.shape),
            ),
            2,
        )
    org_input_shape = grad_input[0].shape  # for the maxpool 1d
    grad_input = [None for _ in grad_input]
    grad_input[0] = torch.where(
        torch.abs(delta_in) < 1e-7,
        torch.zeros_like(delta_in),
        (xmax_pos + rmax_pos) / delta_in,
    ).repeat(dup0)
    if module.__class__.__name__ == "MaxPool1d":
        complex_module_gradients.append(grad_input[0])
        # the grad input that is returned doesn't matter, since it will immediately be
        # be overridden by the grad in the complex_module_gradient
        grad_input[0] = torch.ones(org_input_shape)
    return tuple(grad_input)


def linear_1d(module, grad_input, grad_output):
    """No change made to gradients."""
    return None


def nonlinear_1d(module, grad_input, grad_output):
    delta_out = (
        module.y[: int(module.y.shape[0] / 2)] - module.y[int(module.y.shape[0] / 2) :]
    )

    delta_in = (
        module.x[: int(module.x.shape[0] / 2)] - module.x[int(module.x.shape[0] / 2) :]
    )
    dup0 = [2] + [1 for i in delta_in.shape[1:]]
    # handles numerical instabilities where delta_in is very small by
    # just taking the gradient in those cases
    grads = [None for _ in grad_input]
    grads[0] = torch.where(
        torch.abs(delta_in.repeat(dup0)) < 1e-6,
        grad_input[0],
        grad_output[0] * (delta_out / delta_in).repeat(dup0),
    )
    return tuple(grads)


def split(module, grad_input, grad_output):
    grad_input[0][:, 129:, :] = 0
    return grad_input


op_handler = {}

# passthrough ops, where we make no change to the gradient
op_handler["Dropout3d"] = passthrough
op_handler["Dropout2d"] = passthrough
op_handler["Dropout"] = passthrough
op_handler["AlphaDropout"] = passthrough

op_handler["Conv1d"] = linear_1d
op_handler["Conv2d"] = linear_1d
op_handler["Conv3d"] = linear_1d
op_handler["ConvTranspose1d"] = linear_1d
op_handler["ConvTranspose2d"] = linear_1d
op_handler["ConvTranspose3d"] = linear_1d
op_handler["Linear"] = linear_1d
op_handler["AvgPool1d"] = linear_1d
op_handler["AvgPool2d"] = linear_1d
op_handler["AvgPool3d"] = linear_1d
op_handler["AdaptiveAvgPool1d"] = linear_1d
op_handler["AdaptiveAvgPool2d"] = linear_1d
op_handler["AdaptiveAvgPool3d"] = linear_1d
op_handler["BatchNorm1d"] = linear_1d
op_handler["BatchNorm2d"] = linear_1d
op_handler["BatchNorm3d"] = linear_1d
op_handler["LayerNorm"] = linear_1d

op_handler["LeakyReLU"] = nonlinear_1d
op_handler["ReLU"] = nonlinear_1d
op_handler["ELU"] = nonlinear_1d
op_handler["Sigmoid"] = nonlinear_1d
op_handler["Tanh"] = nonlinear_1d
op_handler["Softplus"] = nonlinear_1d
op_handler["Softmax"] = nonlinear_1d

op_handler["MaxPool1d"] = maxpool
op_handler["MaxPool2d"] = maxpool
op_handler["MaxPool3d"] = maxpool

op_handler["Split"] = split
