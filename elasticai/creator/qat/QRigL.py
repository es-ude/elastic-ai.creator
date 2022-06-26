""" Original: version:https://github.com/nollied/rigl-torch """
import numpy as np
import torch

from elasticai.creator.qat.masks import random_mask_4d


class IndexMaskHook:
    def __init__(self, layer, scheduler):
        self.layer = layer
        self.scheduler = scheduler
        self.dense_grad = None

    def __name__(self):
        return "IndexMaskHook"

    def __eq__(self, other):
        a = isinstance(other, IndexMaskHook) & (
            all(self.layer.shape == other.layer.shape)
        )
        return isinstance(other, IndexMaskHook) & (
            all(self.layer.shape == other.layer.shape)
        )

    @torch.no_grad()
    def __call__(self, grad):
        mask = self.scheduler.backward_masks[self.layer]

        # only calculate dense_grads when necessary
        if self.scheduler.check_if_backward_hook_should_accumulate_grad():
            if self.dense_grad is None:
                # initialize as all 0s so we can do a rolling average
                self.dense_grad = torch.zeros_like(grad)
            self.dense_grad += grad / self.scheduler.grad_accumulation_n
        else:
            self.dense_grad = None

        return grad * mask


def _create_step_wrapper(scheduler, optimizer):
    _unwrapped_step = optimizer.step

    def _wrapped_step():
        _unwrapped_step()
        scheduler.reset_momentum()
        scheduler.apply_mask_to_weights()

    optimizer.step = _wrapped_step


class QRigLScheduler:
    """
    Creates a Rigl scheduler for the selected Conv2d layers. This works similarly to the
    original Rigl [Evci et al 2021] https://arxiv.org/pdf/1911.11134.pdf , with the difference that it has a fixed
    number of connections per channel rather than a per layer fan in. This works by changing the connections,
    by editing a mask every few steps .This is useful in precalculated applications. It replaces the optimizer step
    every few steps. So it should be called like:
    if scheduler():
        optimizer.step()
    Args:
        layers. The Conv2d layers
        optimizer: a torch optimizer it will be wrapped by a function
        num connections: the fixed number of connections per channel
        delta: how often the update step should be done
        alpha: times cosine annealing determines which fraction of weights should get shifted.
        T_end: when the mask should be fixed, given by a fraction of total batches   default expected usage: 0.75 * epochs * len(trainLoader)
        grad_accumulation: added by the Rigl_torch author, takes the averaging score of n batches when determining the growth scores
    """

    def __init__(
        self,
        weighted_layers: list[torch.nn.Module],
        optimizer: torch.optim.Optimizer,
        num_connections: int,
        T_end,
        delta=100,
        alpha=0.3,
        grad_accumulation_n=1,
        state_dict=None,
    ):

        self.layers = weighted_layers
        self.optimizer = optimizer
        self.backward_masks = []
        _create_step_wrapper(self, optimizer)
        self.weights_in_layer = [layer.weight.numel() for layer in self.layers]
        self.num_connections = num_connections
        if state_dict is not None:
            self.load_state_dict(state_dict)
            self.apply_mask_to_weights()

        else:

            # randomly sparsify model according to S
            self.random_sparsify()

            assert 0 < grad_accumulation_n < delta
            # scheduler keeps a log of how many times it's called. this is how it does its scheduling
            self.step = 0
            self.rigl_steps = 0
            # define the actual schedule
            self.delta = delta
            self.alpha = alpha
            self.T_end = T_end
            self.grad_accumulation_n = grad_accumulation_n
        self.backward_hook_objects = []
        for i, layer in enumerate(self.layers):
            if getattr(layer.weight, "_has_rigl_backward_hook", False):
                raise Exception(
                    "This model already has been registered to a RigLScheduler."
                )

            self.backward_hook_objects.append(IndexMaskHook(i, self))
            layer.weight.register_hook(self.backward_hook_objects[-1])
            setattr(layer.weight, "_has_rigl_backward_hook", True)
        self.apply_mask_to_weights()

    def state_dict(self):
        obj = {
            "num_connections": self.num_connections,
            "weights_in_layer": self.weights_in_layer,
            "delta": self.delta,
            "alpha": self.alpha,
            "T_end": self.T_end,
            "grad_accumulation_n": self.grad_accumulation_n,
            "step": self.step,
            "rigl_steps": self.rigl_steps,
            "backward_masks": self.backward_masks,
        }

        return obj

    def __eq__(self, other):
        return (
            isinstance(other, QRigLScheduler)
            & (self.num_connections == other.num_connections)
            & (self.delta == other.delta)
            & (self.T_end == other.T_end)
            & (self.alpha == other.alpha)
            & (self.grad_accumulation_n == other.grad_accumulation_n)
            & (self.weights_in_layer == other.weights_in_layer)
            & (
                torch.all(
                    list(
                        a == b
                        for a, b in zip(self.backward_masks, other.backward_masks)
                    )[0]
                ).item()
            )
        )

    def load_state_dict(self, state_dict):
        for k, v in state_dict.items():
            if type(v) == dict:
                self.load_state_dict(v)
            setattr(self, k, v)

    @torch.no_grad()
    def random_sparsify(self):
        self.backward_masks = []
        for layer_index, layer in enumerate(self.layers):
            # if sparsity is 0%, skip

            mask = random_mask_4d(
                out_channels=layer.out_channels,
                kernel_size=layer.kernel_size,
                in_channels=layer.in_channels,
                groups=layer.groups,
                params_per_channel=self.num_connections,
            )
            layer.weight *= mask
            self.backward_masks.append(mask)

    def __str__(self):
        s = "RigLScheduler(\n"
        s += "layers=%i,\n" % len(self.weights_in_layer)

        # calculate the number of non-zero elements out of the total number of elements
        N_str = "["
        S_str = "["
        sparsity_percentages = []
        total_params = 0
        total_conv_params = 0
        total_nonzero = 0
        total_conv_nonzero = 0

        for N, mask, layer in zip(
            self.weights_in_layer, self.backward_masks, self.layers
        ):
            actual_S = torch.sum(layer.weight[mask == 0] == 0).item()
            N_str += "%i/%i, " % (N - actual_S, N)
            sp_p = float(N - actual_S) / float(N) * 100
            S_str += "%.2f%%, " % sp_p
            sparsity_percentages.append(sp_p)
            total_params += N
            total_nonzero += N - actual_S
            # if not is_linear:
            total_conv_nonzero += N - actual_S
            total_conv_params += N

        N_str = N_str[:-2] + "]"
        S_str = S_str[:-2] + "]"

        s += "nonzero_params=" + N_str + ",\n"
        s += "nonzero_percentages=" + S_str + ",\n"
        s += (
            "total_nonzero_params="
            + (
                "%i/%i (%.2f%%)"
                % (
                    total_nonzero,
                    total_params,
                    float(total_nonzero) / float(total_params) * 100,
                )
            )
            + ",\n"
        )
        s += (
            "total_CONV_nonzero_params="
            + (
                "%i/%i (%.2f%%)"
                % (
                    total_conv_nonzero,
                    total_conv_params,
                    float(total_conv_nonzero) / float(total_conv_params) * 100,
                )
            )
            + ",\n"
        )
        s += "step=" + str(self.step) + ",\n"
        s += "num_rigl_steps=" + str(self.rigl_steps) + ",\n"

        return s + ")"

    @torch.no_grad()
    def reset_momentum(self):
        # raise NotImplementedError("Not compatible with momentum yet")
        for layer, mask in zip(self.layers, self.backward_masks):

            param_state = self.optimizer.state[layer.weight]
            if "momentum_buffer" in param_state:
                # mask the momentum matrix
                buf = param_state["momentum_buffer"]
                buf *= mask

    @torch.no_grad()
    def apply_mask_to_weights(self):
        for layer, mask in zip(self.layers, self.backward_masks):
            layer.weight *= mask

    @torch.no_grad()
    def apply_mask_to_gradients(self):
        for layer, mask in zip(self.layers, self.backward_masks):
            # if sparsity is 0%, skip
            layer.weight.grad *= mask

    def check_if_backward_hook_should_accumulate_grad(self):
        """
        Used by the backward hooks. Basically just checks how far away the next rigl step is,
        if it's within `self.grad_accumulation_n` steps, return True.
        """

        if self.step >= self.T_end:
            return False

        steps_til_next_rigl_step = self.delta - (self.step % self.delta)
        return steps_til_next_rigl_step <= self.grad_accumulation_n

    def cosine_annealing(self):
        return self.alpha / 2 * (1 + np.cos((self.step * np.pi) / self.T_end))

    def __call__(self):
        self.step += 1
        if (self.step % self.delta) == 0 and self.step < self.T_end:  # check schedule
            self._rigl_step()
            self.rigl_steps += 1
            return False
        return True

    @torch.no_grad()
    def _rigl_step(self):
        drop_fraction = self.cosine_annealing()

        for l, layer in enumerate(self.layers):
            weight_sizes = layer.weight.shape
            if (
                int(self.num_connections * drop_fraction) <= 0
                or weight_sizes[1] * weight_sizes[2] * weight_sizes[3]
                <= self.num_connections
            ):
                continue

            channel_masks = []
            channel_new_weights = []
            channels = layer.weight.shape[0]
            for channel in range(channels):
                current_mask = self.backward_masks[l][channel]

                # calculate raw scores
                score_drop = torch.abs(layer.weight[channel])
                score_grow = torch.abs(
                    self.backward_hook_objects[l].dense_grad[channel]
                )

                # calculate drop/grow quantities
                n_total = self.weights_in_layer[l] // channels
                n_ones = torch.sum(current_mask).item()
                n_prune = int(n_ones * drop_fraction)
                n_keep = n_ones - n_prune

                # create drop mask
                _, sorted_indices = torch.topk(score_drop.view(-1), k=n_total)
                new_values = torch.where(
                    torch.arange(n_total, device=layer.weight.device) < n_keep,
                    torch.ones_like(sorted_indices),
                    torch.zeros_like(sorted_indices),
                )
                mask1 = new_values.scatter(0, sorted_indices, new_values)

                # flatten grow scores
                score_grow = score_grow.view(-1)

                # set scores of the enabled connections(ones) to min(s) - 1, so that they have the lowest scores
                score_grow_lifted = torch.where(
                    mask1 == 1,
                    torch.ones_like(mask1) * (torch.min(score_grow) - 1),
                    score_grow,
                )

                # create grow mask
                _, sorted_indices = torch.topk(score_grow_lifted, k=n_total)
                new_values = torch.where(
                    torch.arange(n_total, device=layer.weight.device) < n_prune,
                    torch.ones_like(sorted_indices),
                    torch.zeros_like(sorted_indices),
                )
                mask2 = new_values.scatter(0, sorted_indices, new_values)

                mask2_reshaped = torch.reshape(mask2, current_mask.shape)
                grow_tensor = torch.zeros_like(layer.weight[0])

                new_connections = (mask2_reshaped == 1) & (current_mask == 0)

                # update new weights to be initialized as zeros and update the weight tensors
                new_weights = torch.where(
                    new_connections.to(layer.weight.device),
                    grow_tensor,
                    layer.weight[channel],
                )
                channel_new_weights.append(new_weights)
                mask_combined = torch.reshape(mask1 + mask2, current_mask.shape).bool()
                channel_masks.append(mask_combined)

            layer.weight.data = torch.stack(channel_new_weights, 0)

            current_mask = self.backward_masks[l]
            # update the mask
            current_mask.data = torch.stack(channel_masks)

        self.reset_momentum()
        self.apply_mask_to_weights()
        self.apply_mask_to_gradients()
