# SPDX-License-Identifier: Apache-2.0
# Parallel state and collective helpers for the MiniMax H3 visual VAE.
import os
import math
import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.autograd import Function
from torch.distributed import group, ReduceOp


def get_group_rank(group_size):
    global_rank = int(os.environ["RANK"])
    group_rank = global_rank % group_size
    return group_rank


_parallel_state = {}

# The torch.autograd.Function subclasses below keep their backward() methods
# to satisfy the autograd.Function contract; only the forward paths are
# exercised in this inference-only bundle.


def get_parallel_state():
    return _parallel_state


class _AllGather(Function):
    @staticmethod
    def forward(ctx, group, tensor):
        tensor = tensor.contiguous()
        ctx.group = group
        group_size = dist.get_world_size(group=group)
        out_tensor_list = [torch.empty_like(tensor) for _ in range(group_size)]
        dist.all_gather(out_tensor_list, tensor, group=group)
        return tuple(out_tensor_list)

    @staticmethod
    def backward(ctx, *grad_outputs):
        rank = dist.get_rank(group=ctx.group)
        gx = torch.empty_like(grad_outputs[rank])
        gx = gx.contiguous()
        grad_outputs = tuple(t.contiguous() for t in grad_outputs)
        dist.reduce_scatter(gx, list(grad_outputs), op=ReduceOp.SUM, group=ctx.group)
        return (None, gx)


@torch.compiler.disable
def all_gather(tensor, group=group.WORLD):
    return _AllGather.apply(group, tensor)


class _AllGatherVarShape(Function):
    @staticmethod
    def forward(ctx, group, tensor):
        tensor = tensor.contiguous()
        ctx.group = group
        ctx.original_shape = tensor.shape

        shape_info = torch.tensor(
            list(tensor.shape), dtype=torch.long, device=tensor.device
        )

        shape_list = [
            torch.empty_like(shape_info)
            for _ in range(dist.get_world_size(group=group))
        ]
        dist.all_gather(shape_list, shape_info, group=group)

        all_shapes = [tuple(shape_tensor.tolist()) for shape_tensor in shape_list]
        ctx.all_shapes = all_shapes

        flat_tensor = tensor.flatten()
        max_size = max(math.prod(s) for s in all_shapes)

        if flat_tensor.numel() < max_size:
            padded = torch.zeros(max_size, dtype=tensor.dtype, device=tensor.device)
            padded[: flat_tensor.numel()] = flat_tensor
            flat_tensor = padded

        gathered_flat = [torch.empty_like(flat_tensor) for _ in range(len(all_shapes))]
        dist.all_gather(gathered_flat, flat_tensor, group=group)

        return tuple(
            t[: math.prod(shape)].reshape(shape)
            for t, shape in zip(gathered_flat, all_shapes)
        )

    @staticmethod
    def backward(ctx, *grad_outputs):
        rank = dist.get_rank(group=ctx.group)

        grad_input = grad_outputs[rank]
        if grad_input is None:
            return None, torch.zeros(
                ctx.original_shape, device=next(iter(grad_outputs)).device
            )

        max_size = max(math.prod(shape) for shape in ctx.all_shapes)
        padded_grads = []

        for grad, shape in zip(grad_outputs, ctx.all_shapes):
            if grad is not None:
                flat_grad = grad.flatten()
            else:
                flat_grad = torch.zeros(
                    math.prod(shape),
                    dtype=grad_input.dtype,
                    device=grad_input.device,
                )

            if flat_grad.numel() < max_size:
                padded = torch.zeros(
                    max_size, dtype=flat_grad.dtype, device=flat_grad.device
                )
                padded[: flat_grad.numel()] = flat_grad
                padded_grads.append(padded)
            else:
                padded_grads.append(flat_grad)

        result_grad = torch.empty_like(padded_grads[0])
        dist.reduce_scatter(result_grad, padded_grads, op=ReduceOp.SUM, group=ctx.group)

        original_size = math.prod(ctx.original_shape)
        return None, result_grad[:original_size].reshape(ctx.original_shape)


@torch.compiler.disable
def all_gather_var_shape(tensor, group=group.WORLD):
    return _AllGatherVarShape.apply(group, tensor)


class _AllReduce(Function):
    @staticmethod
    def forward(ctx, _input, op, group):
        ctx.group = group
        ctx.op = op
        _input = _input.clone()
        dist.all_reduce(_input, op=op, group=group)
        return _input

    @staticmethod
    def backward(ctx, grad_output):
        grad_output = grad_output.clone()
        dist.all_reduce(grad_output, op=ctx.op, group=ctx.group)
        return grad_output, None, None


@torch.compiler.disable
def all_reduce(input_, op, group):
    return _AllReduce.apply(input_, op, group)


class _AlltoAllSingle(Function):
    @staticmethod
    def forward(ctx, group, input):
        ctx.group = group

        world_size = dist.get_world_size(group=group)
        if world_size == 1:
            return input

        input = input.contiguous()
        output = torch.empty_like(input)
        dist.all_to_all_single(
            output,
            input,
            group=group,
        )
        return output

    @staticmethod
    def backward(ctx, grad_output):
        return (None, _AlltoAllSingle.apply(ctx.group, grad_output))


@torch.compiler.disable
def all_to_all_single(input, group=group.WORLD):
    return _AlltoAllSingle.apply(group, input)



@torch.compiler.disable
def get_subseq(input, sp_size=None):
    if sp_size is None:
        state = get_parallel_state()
        if not state.get("sp_enabled", False):
            return input
        sp_size = state["sp_size"]
        sp_rank = state["sp_rank"]
    else:
        sp_rank = get_group_rank(sp_size)

    if sp_size == 1:
        return input

    if input.shape[1] % sp_size != 0:
        raise ValueError(
            f"Input shape {input.shape} is not divisible by sp_size {sp_size}"
        )

    return torch.chunk(input, sp_size, dim=1)[sp_rank]


@torch.compiler.disable
def gather_subseq(input, sp_size=None, local_process_group=None):
    if sp_size is None:
        state = get_parallel_state()
        if not state.get("sp_enabled", False):
            return input
        sp_size = state["sp_size"]
        local_process_group = state["sp_process_group"]

    if sp_size == 1:
        return input

    output = all_gather(input, group=local_process_group)
    output = torch.cat(output, dim=1)
    return output


@torch.compiler.disable
def all_to_all_4D(
    input: torch.tensor,
    scatter_idx: int = 2,
    gather_idx: int = 1,
    group=None,
):
    assert (
        input.dim() == 4
    ), f"input must be 4D tensor, got {input.dim()} and shape {input.shape}"

    if group is None:
        seq_world_size = 1
    else:
        seq_world_size = dist.get_world_size(group)

    if seq_world_size == 1:
        return input

    if scatter_idx == 2 and gather_idx == 1:
        bs, shard_seqlen, hc, hs = input.shape
        seqlen = shard_seqlen * seq_world_size
        shard_hc = hc // seq_world_size

        input_t = (
            input.reshape(bs, shard_seqlen, seq_world_size, shard_hc, hs)
            .transpose(0, 2)
            .contiguous()
        )

        output = all_to_all_single(input_t, group=group)
        output = output.reshape(seqlen, bs, shard_hc, hs)
        output = output.transpose(0, 1).contiguous().reshape(bs, seqlen, shard_hc, hs)
        return output

    elif scatter_idx == 1 and gather_idx == 2:
        bs, seqlen, shard_hc, hs = input.shape
        hc = shard_hc * seq_world_size
        shard_seqlen = seqlen // seq_world_size

        input_t = (
            input.reshape(bs, seq_world_size, shard_seqlen, shard_hc, hs)
            .transpose(0, 3)
            .transpose(0, 1)
            .contiguous()
            .reshape(seq_world_size, shard_hc, shard_seqlen, bs, hs)
        )

        output = all_to_all_single(input_t, group=group)
        output = output.reshape(hc, shard_seqlen, bs, hs)
        output = output.transpose(0, 2).contiguous().reshape(bs, shard_seqlen, hc, hs)
        return output
    else:
        raise RuntimeError("scatter_idx must be 1 or 2 and gather_idx must be 1 or 2")



@torch.compiler.disable
def exchange_borders(
    input_, padding, pad_mode, sp_rank, sp_size, group, dim=-1, async_op=False
):
    if async_op and input_.requires_grad:
        raise ValueError("async_op is not supported backward, check previous commits")

    slice_indices = [slice(None)] * input_.ndim
    slice_indices[dim] = slice(None, padding)
    first_tensor = input_[tuple(slice_indices)].contiguous()

    slice_indices[dim] = slice(-padding, None)
    last_tensor = input_[tuple(slice_indices)].contiguous()

    if async_op:
        first_borders = [torch.empty_like(first_tensor) for _ in range(sp_size)]
        last_borders = [torch.empty_like(last_tensor) for _ in range(sp_size)]

        handle_first = dist.all_gather(
            first_borders, first_tensor, group=group, async_op=True
        )
        handle_last = dist.all_gather(
            last_borders, last_tensor, group=group, async_op=True
        )
    else:
        first_borders = all_gather(first_tensor, group=group)
        last_borders = all_gather(last_tensor, group=group)

    if dim < 0:
        pad_dim = -1 - dim
    else:
        pad_dim = input_.ndim - 1 - dim

    pad_size = [0] * ((input_.ndim - 2) * 2)
    pad_size[pad_dim * 2] = padding
    pad_size[pad_dim * 2 + 1] = padding
    output = F.pad(input_, pad_size, mode=pad_mode)

    slice_indices = [slice(None)] * input_.ndim
    slice_indices[dim] = slice(-padding, None)

    if async_op:
        handle_first.wait()

    if sp_rank < sp_size - 1:
        output[tuple(slice_indices)] = first_borders[sp_rank + 1]
    else:
        output[tuple(slice_indices)] += first_borders[0] * 0.0

    slice_indices = [slice(None)] * input_.ndim
    slice_indices[dim] = slice(None, padding)

    if async_op:
        handle_last.wait()

    if sp_rank > 0:
        output[tuple(slice_indices)] = last_borders[sp_rank - 1]
    else:
        output[tuple(slice_indices)] += last_borders[sp_size - 1] * 0.0

    return output


@torch.compiler.disable
def exchange_strides(
    input_, pad_mode, sp_rank, sp_size, group, dim=-1, async_op=False
):
    if async_op and input_.requires_grad:
        raise ValueError("async_op is not supported backward, check previous commits")

    if dim not in [-1, -2]:
        raise ValueError("dim must be -1 (W) or -2 (H) for exchange_strides")

    if dim == -1:
        if input_.ndim == 5:
            input_ = F.pad(input_, (0, 0, 0, 1, 0, 0), mode=pad_mode)
        elif input_.ndim == 4:
            input_ = F.pad(input_, (0, 0, 0, 1), mode=pad_mode)
        else:
            raise ValueError(f"Input must have 4 or 5 dimensions, got {input_.ndim}")

        left_border = input_[..., :1].contiguous()

        if async_op:
            left_borders = [torch.empty_like(left_border) for _ in range(sp_size)]
            handle = dist.all_gather(
                left_borders, left_border, group=group, async_op=True
            )
        else:
            left_borders = all_gather(left_border, group=group)

        if input_.ndim == 5:
            output = F.pad(input_, (0, 1, 0, 0, 0, 0), mode=pad_mode)
        elif input_.ndim == 4:
            output = F.pad(input_, (0, 1, 0, 0), mode=pad_mode)
        else:
            raise ValueError(f"Input must have 4 or 5 dimensions, got {input_.ndim}")

        if async_op:
            handle.wait()

        if sp_rank != sp_size - 1:
            output[..., -1:] = left_borders[sp_rank + 1]
        else:
            output[..., -1:] += left_borders[0] * 0.0
    else:
        if input_.ndim == 5:
            input_ = F.pad(input_, (0, 1, 0, 0, 0, 0), mode=pad_mode)
        elif input_.ndim == 4:
            input_ = F.pad(input_, (0, 1, 0, 0), mode=pad_mode)
        else:
            raise ValueError(f"Input must have 4 or 5 dimensions, got {input_.ndim}")

        top_border = input_[..., :1, :].contiguous()

        if async_op:
            top_borders = [torch.empty_like(top_border) for _ in range(sp_size)]
            handle = dist.all_gather(
                top_borders, top_border, group=group, async_op=True
            )
        else:
            top_borders = all_gather(top_border, group=group)

        if input_.ndim == 5:
            output = F.pad(input_, (0, 0, 0, 1, 0, 0), mode=pad_mode)
        elif input_.ndim == 4:
            output = F.pad(input_, (0, 0, 0, 1), mode=pad_mode)
        else:
            raise ValueError(f"Input must have 4 or 5 dimensions, got {input_.ndim}")

        if async_op:
            handle.wait()

        if sp_rank != sp_size - 1:
            output[..., -1:, :] = top_borders[sp_rank + 1]
        else:
            output[..., -1:, :] += top_borders[0] * 0.0

    return output
