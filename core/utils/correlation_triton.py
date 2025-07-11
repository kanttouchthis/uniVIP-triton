#!/usr/bin/env python

import torch
import triton
import triton.language as tl


@triton.jit
def correlation_rearrange_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    batch_size,
    channels,
    height,
    width,
    padded_height,
    padded_width,
    BLOCK_SIZE: tl.constexpr,
):
    """Rearrange input tensor with padding for correlation computation."""
    pid = tl.program_id(axis=0)
    batch_id = tl.program_id(axis=1)
    channel_id = tl.program_id(axis=2)
    
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Calculate input indices
    input_base = batch_id * channels * height * width + channel_id * height * width
    input_indices = input_base + offsets
    
    # Load input values
    input_vals = tl.load(input_ptr + input_indices, mask=mask, other=0.0)
    
    # Calculate padded coordinates
    y_coords = offsets // width
    x_coords = offsets % width
    padded_y = y_coords + 4
    padded_x = x_coords + 4
    
    # Calculate output indices (rearranged layout)
    rearrange_indices = padded_y * padded_width + padded_x
    output_base = batch_id * channels * padded_height * padded_width
    output_indices = output_base + rearrange_indices * channels + channel_id
    
    # Store rearranged values
    tl.store(output_ptr + output_indices, input_vals, mask=mask)


@triton.jit
def correlation_forward_kernel(
    rbot0_ptr,
    rbot1_ptr,
    output_ptr,
    batch_size,
    channels,
    height,
    width,
    padded_height,
    padded_width,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute correlation between two rearranged tensors."""
    x_id = tl.program_id(axis=0)
    y_id = tl.program_id(axis=1)
    batch_id = tl.program_id(axis=2)
    
    # Position in padded tensor
    x1 = x_id + 4
    y1 = y_id + 4
    
    # Load patch from rbot0
    patch_base = batch_id * channels * padded_height * padded_width
    patch_offset = (y1 * padded_width + x1) * channels
    patch_indices = patch_base + patch_offset + tl.arange(0, BLOCK_SIZE)
    patch_mask = tl.arange(0, BLOCK_SIZE) < channels
    patch_data = tl.load(rbot0_ptr + patch_indices, mask=patch_mask, other=0.0)
    
    # Compute correlation for all 81 displacement combinations
    for top_channel in range(81):
        s2o = (top_channel % 9) - 4  # x displacement
        s2p = (top_channel // 9) - 4  # y displacement
        
        # Calculate corresponding position in rbot1
        x2 = x1 + s2o
        y2 = y1 + s2p
        
        # Load corresponding patch from rbot1
        patch1_offset = (y2 * padded_width + x2) * channels
        patch1_indices = patch_base + patch1_offset + tl.arange(0, BLOCK_SIZE)
        patch1_data = tl.load(rbot1_ptr + patch1_indices, mask=patch_mask, other=0.0)
        
        # Compute correlation
        correlation = tl.sum(patch_data * patch1_data) / channels
        
        # Store result
        output_idx = (batch_id * 81 * height * width + 
                     top_channel * height * width + 
                     y_id * width + x_id)
        tl.store(output_ptr + output_idx, correlation)


@triton.jit
def correlation_backward_first_kernel(
    rbot0_ptr,
    rbot1_ptr,
    grad_output_ptr,
    grad_first_ptr,
    batch_id,
    channels,
    height,
    width,
    padded_height,
    padded_width,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute gradient with respect to first input."""
    pid = tl.program_id(axis=0)
    
    # Calculate position in original tensor
    n_elements = channels * height * width
    element_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = element_id < n_elements
    
    channel_id = element_id % channels
    spatial_id = element_id // channels
    y_pos = spatial_id // width
    x_pos = spatial_id % width
    
    # Position in padded tensor
    padded_y = y_pos + 4
    padded_x = x_pos + 4
    
    # Calculate gradient bounds
    xmin = tl.maximum(0, padded_x - 4)
    xmax = tl.minimum(width - 1, padded_x - 4)
    ymin = tl.maximum(0, padded_y - 4)
    ymax = tl.minimum(height - 1, padded_y - 4)
    
    grad_sum = tl.zeros_like(element_id, dtype=tl.float32)
    
    # Iterate over all possible displacements
    for p in range(-4, 5):
        for o in range(-4, 5):
            # Get rbot1 data
            rbot1_y = padded_y + p
            rbot1_x = padded_x + o
            rbot1_idx = (batch_id * channels * padded_height * padded_width + 
                        (rbot1_y * padded_width + rbot1_x) * channels + channel_id)
            rbot1_val = tl.load(rbot1_ptr + rbot1_idx, mask=mask, other=0.0)
            
            # Calculate displacement index
            op = (p + 4) * 9 + (o + 4)
            
            # Accumulate gradient over valid positions
            for y in range(ymin, ymax + 1):
                for x in range(xmin, xmax + 1):
                    grad_idx = (batch_id * 81 * height * width + 
                               op * height * width + y * width + x)
                    grad_val = tl.load(grad_output_ptr + grad_idx)
                    grad_sum += grad_val * rbot1_val
    
    # Normalize and store gradient
    grad_sum = grad_sum / channels
    output_idx = (batch_id * channels * height * width + 
                 channel_id * height * width + y_pos * width + x_pos)
    tl.store(grad_first_ptr + output_idx, grad_sum, mask=mask)


@triton.jit
def correlation_backward_second_kernel(
    rbot0_ptr,
    rbot1_ptr,
    grad_output_ptr,
    grad_second_ptr,
    batch_id,
    channels,
    height,
    width,
    padded_height,
    padded_width,
    BLOCK_SIZE: tl.constexpr,
):
    """Compute gradient with respect to second input."""
    pid = tl.program_id(axis=0)
    
    # Calculate position in original tensor
    n_elements = channels * height * width
    element_id = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = element_id < n_elements
    
    channel_id = element_id % channels
    spatial_id = element_id // channels
    y_pos = spatial_id // width
    x_pos = spatial_id % width
    
    # Position in padded tensor
    padded_y = y_pos + 4
    padded_x = x_pos + 4
    
    grad_sum = tl.zeros_like(element_id, dtype=tl.float32)
    
    # Iterate over all possible displacements
    for p in range(-4, 5):
        for o in range(-4, 5):
            # Calculate bounds for this displacement
            xmin = tl.maximum(0, padded_x - 4 - o)
            xmax = tl.minimum(width - 1, padded_x - 4 - o)
            ymin = tl.maximum(0, padded_y - 4 - p)
            ymax = tl.minimum(height - 1, padded_y - 4 - p)
            
            # Check if bounds are valid
            if xmax >= 0 and ymax >= 0 and xmin <= width - 1 and ymin <= height - 1:
                # Get rbot0 data
                rbot0_y = padded_y - p
                rbot0_x = padded_x - o
                rbot0_idx = (batch_id * channels * padded_height * padded_width + 
                            (rbot0_y * padded_width + rbot0_x) * channels + channel_id)
                rbot0_val = tl.load(rbot0_ptr + rbot0_idx, mask=mask, other=0.0)
                
                # Calculate displacement index
                op = (p + 4) * 9 + (o + 4)
                
                # Accumulate gradient over valid positions
                for y in range(ymin, ymax + 1):
                    for x in range(xmin, xmax + 1):
                        grad_idx = (batch_id * 81 * height * width + 
                                   op * height * width + y * width + x)
                        grad_val = tl.load(grad_output_ptr + grad_idx)
                        grad_sum += grad_val * rbot0_val
    
    # Normalize and store gradient
    grad_sum = grad_sum / channels
    output_idx = (batch_id * channels * height * width + 
                 channel_id * height * width + y_pos * width + x_pos)
    tl.store(grad_second_ptr + output_idx, grad_sum, mask=mask)


def triton_correlation_rearrange(input_tensor, output_tensor):
    """Rearrange input tensor with padding using Triton."""
    batch_size, channels, height, width = input_tensor.shape
    padded_height, padded_width = height + 8, width + 8
    n_elements = height * width
    
    BLOCK_SIZE = 256
    grid = (
        triton.cdiv(n_elements, BLOCK_SIZE),
        batch_size,
        channels,
    )
    
    correlation_rearrange_kernel[grid](
        input_tensor,
        output_tensor,
        n_elements,
        batch_size,
        channels,
        height,
        width,
        padded_height,
        padded_width,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def triton_correlation_forward(rbot0, rbot1, output):
    """Compute correlation using Triton."""
    batch_size, channels, height, width = output.shape[0], rbot0.shape[3], output.shape[2], output.shape[3]
    padded_height, padded_width = height + 8, width + 8
    
    BLOCK_SIZE = min(1024, triton.next_power_of_2(channels))
    grid = (width, height, batch_size)
    
    correlation_forward_kernel[grid](
        rbot0,
        rbot1,
        output,
        batch_size,
        channels,
        height,
        width,
        padded_height,
        padded_width,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def triton_correlation_backward_first(rbot0, rbot1, grad_output, grad_first, batch_id):
    """Compute gradient with respect to first input using Triton."""
    batch_size, channels, height, width = grad_first.shape
    padded_height, padded_width = height + 8, width + 8
    n_elements = channels * height * width
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    correlation_backward_first_kernel[grid](
        rbot0,
        rbot1,
        grad_output,
        grad_first,
        batch_id,
        channels,
        height,
        width,
        padded_height,
        padded_width,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def triton_correlation_backward_second(rbot0, rbot1, grad_output, grad_second, batch_id):
    """Compute gradient with respect to second input using Triton."""
    batch_size, channels, height, width = grad_second.shape
    padded_height, padded_width = height + 8, width + 8
    n_elements = channels * height * width
    
    BLOCK_SIZE = 256
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    correlation_backward_second_kernel[grid](
        rbot0,
        rbot1,
        grad_output,
        grad_second,
        batch_id,
        channels,
        height,
        width,
        padded_height,
        padded_width,
        BLOCK_SIZE=BLOCK_SIZE,
    )


class _FunctionCorrelation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, first, second):
        # Create padded rearranged tensors
        rbot0 = first.new_zeros([first.shape[0], first.shape[2] + 8, first.shape[3] + 8, first.shape[1]])
        rbot1 = first.new_zeros([first.shape[0], first.shape[2] + 8, first.shape[3] + 8, first.shape[1]])

        ctx.save_for_backward(first, second, rbot0, rbot1)

        assert first.is_contiguous(), "First input must be contiguous"
        assert second.is_contiguous(), "Second input must be contiguous"

        # Create output tensor
        output = first.new_zeros([first.shape[0], 81, first.shape[2], first.shape[3]])

        if first.is_cuda:
            # Rearrange inputs with padding
            triton_correlation_rearrange(first, rbot0)
            triton_correlation_rearrange(second, rbot1)
            
            # Compute correlation
            triton_correlation_forward(rbot0, rbot1, output)
        else:
            raise NotImplementedError("CPU implementation not available")

        return output

    @staticmethod
    def backward(ctx, grad_output):
        first, second, rbot0, rbot1 = ctx.saved_tensors

        assert grad_output.is_contiguous(), "Gradient output must be contiguous"

        grad_first = None
        grad_second = None
        
        if ctx.needs_input_grad[0]:
            grad_first = first.new_zeros_like(first)
        if ctx.needs_input_grad[1]:
            grad_second = first.new_zeros_like(first)

        if first.is_cuda:
            if grad_first is not None:
                for batch_id in range(first.shape[0]):
                    triton_correlation_backward_first(rbot0, rbot1, grad_output, grad_first, batch_id)

            if grad_second is not None:
                for batch_id in range(first.shape[0]):
                    triton_correlation_backward_second(rbot0, rbot1, grad_output, grad_second, batch_id)
        else:
            raise NotImplementedError("CPU implementation not available")

        return grad_first, grad_second


def FunctionCorrelation(tenFirst, tenSecond):
    """Backwards compatible function interface."""
    return _FunctionCorrelation.apply(tenFirst, tenSecond)


class ModuleCorrelation(torch.nn.Module):
    """Backwards compatible module interface."""
    def __init__(self):
        super(ModuleCorrelation, self).__init__()

    def forward(self, tenFirst, tenSecond):
        return _FunctionCorrelation.apply(tenFirst, tenSecond)