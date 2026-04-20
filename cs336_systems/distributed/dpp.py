from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn


class NaiveDDP(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._broadcast_module_state()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _broadcast_module_state(self) -> None:
        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

    def finish_gradient_synchronization(self) -> None:
        world_size = dist.get_world_size()
        if world_size == 1:
            return

        for parameter in self.module.parameters():
            if parameter.grad is None:
                continue
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
            parameter.grad.div_(world_size)

    def finish_gradient_sync(self) -> None:
        self.finish_gradient_synchronization()


# Preserve the existing misspelled class name used by the local adapter wiring.
NaiveDPP = NaiveDDP

class MinDPPFlat(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self._broadcast_module_state()

    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _broadcast_module_state(self) -> None:
        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

    def finish_gradient_synchronization(self) -> None:
        world_size = dist.get_world_size()
        if world_size == 1:
            return

        params_with_grad = [
            parameter for parameter in self.module.parameters() if parameter.grad is not None
        ]
        if not params_with_grad:
            return

        grads = [parameter.grad for parameter in params_with_grad]
        flat_grad = torch._utils._flatten_dense_tensors(grads)
        dist.all_reduce(flat_grad, op=dist.ReduceOp.SUM)
        flat_grad.div_(world_size)

        reduced_grads = torch._utils._unflatten_dense_tensors(flat_grad, grads)
        for parameter, reduced_grad in zip(params_with_grad, reduced_grads):
            parameter.grad.copy_(reduced_grad)
        
    def finish_gradient_sync(self) -> None:
        self.finish_gradient_synchronization()


class DDPOverlap(nn.Module):
    def __init__(self, module: nn.Module):
        super().__init__()
        self.module = module
        self.handles = []
        self._broadcast_module_state()
        self._register_grad_hooks()


    def forward(self, *args, **kwargs):
        return self.module(*args, **kwargs)

    def _broadcast_module_state(self) -> None:
        for parameter in self.module.parameters():
            dist.broadcast(parameter.data, src=0)
        for buffer in self.module.buffers():
            dist.broadcast(buffer.data, src=0)

    def _register_grad_hooks(self) -> None:
        for parameter in self.module.parameters():
            if parameter.requires_grad:
                parameter.register_post_accumulate_grad_hook(self.grad_hook)

    def finish_gradient_synchronization(self) -> None:
        for handle in self.handles:
            handle.wait()
        self.handles.clear()
        params = [parameter for parameter in self.module.parameters() if parameter.grad is not None]
        world_size = dist.get_world_size()
        if world_size > 1:
            for parameter in params:
                parameter.grad.div_(world_size)

    def finish_gradient_sync(self) -> None:
        self.finish_gradient_synchronization()
        
    def grad_hook(self, param: torch.Tensor) -> None:
        world_size = dist.get_world_size()
        if world_size == 1:
            return
        handle = dist.all_reduce(param.grad, op=dist.ReduceOp.SUM, async_op=True)
        self.handles.append(handle)
