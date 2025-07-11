#!/usr/bin/env python

import torch
import triton
import triton.language as tl


@triton.jit
def softsplat_forward_kernel(
    input_ptr, flow_ptr, output_ptr,
    n_batch, n_channels, height, width,
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    flow_batch_stride, flow_channel_stride, flow_height_stride, flow_width_stride,
    output_batch_stride, output_channel_stride, output_height_stride, output_width_stride,
    BLOCK_SIZE: tl.constexpr
):
    """Triton kernel for softsplat forward pass"""
    pid = tl.program_id(axis=0)
    
    # Calculate total number of elements to process
    total_elements = n_batch * n_channels * height * width
    
    # Calculate indices for this thread
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    
    # Decompose linear index into batch, channel, y, x coordinates
    batch_idx = idx // (n_channels * height * width)
    remaining = idx % (n_channels * height * width)
    channel_idx = remaining // (height * width)
    remaining = remaining % (height * width)
    y_idx = remaining // width
    x_idx = remaining % width
    
    # Get input value
    input_offset = (batch_idx * input_batch_stride + 
                   channel_idx * input_channel_stride + 
                   y_idx * input_height_stride + 
                   x_idx * input_width_stride)
    input_val = tl.load(input_ptr + input_offset, mask=mask)
    
    # Get flow values (flow has 2 channels: x and y displacement)
    flow_x_offset = (batch_idx * flow_batch_stride + 
                    0 * flow_channel_stride + 
                    y_idx * flow_height_stride + 
                    x_idx * flow_width_stride)
    flow_y_offset = (batch_idx * flow_batch_stride + 
                    1 * flow_channel_stride + 
                    y_idx * flow_height_stride + 
                    x_idx * flow_width_stride)
    
    flow_x = tl.load(flow_ptr + flow_x_offset, mask=mask)
    flow_y = tl.load(flow_ptr + flow_y_offset, mask=mask)
    
    # Calculate output coordinates
    output_x = x_idx.to(tl.float32) + flow_x
    output_y = y_idx.to(tl.float32) + flow_y
    
    # Calculate bilinear interpolation coordinates
    nw_x = tl.floor(output_x).to(tl.int32)
    nw_y = tl.floor(output_y).to(tl.int32)
    ne_x = nw_x + 1
    ne_y = nw_y
    sw_x = nw_x
    sw_y = nw_y + 1
    se_x = nw_x + 1
    se_y = nw_y + 1
    
    # Calculate bilinear weights
    nw_weight = (se_x.to(tl.float32) - output_x) * (se_y.to(tl.float32) - output_y)
    ne_weight = (output_x - sw_x.to(tl.float32)) * (sw_y.to(tl.float32) - output_y)
    sw_weight = (ne_x.to(tl.float32) - output_x) * (output_y - ne_y.to(tl.float32))
    se_weight = (output_x - nw_x.to(tl.float32)) * (output_y - nw_y.to(tl.float32))
    
    # Splat to output using atomic operations
    # Northwest
    nw_valid = (nw_x >= 0) & (nw_x < width) & (nw_y >= 0) & (nw_y < height)
    nw_output_offset = (batch_idx * output_batch_stride + 
                       channel_idx * output_channel_stride + 
                       nw_y * output_height_stride + 
                       nw_x * output_width_stride)
    nw_value = input_val * nw_weight
    tl.atomic_add(output_ptr + nw_output_offset, nw_value, mask=mask & nw_valid)
    
    # Northeast
    ne_valid = (ne_x >= 0) & (ne_x < width) & (ne_y >= 0) & (ne_y < height)
    ne_output_offset = (batch_idx * output_batch_stride + 
                       channel_idx * output_channel_stride + 
                       ne_y * output_height_stride + 
                       ne_x * output_width_stride)
    ne_value = input_val * ne_weight
    tl.atomic_add(output_ptr + ne_output_offset, ne_value, mask=mask & ne_valid)
    
    # Southwest
    sw_valid = (sw_x >= 0) & (sw_x < width) & (sw_y >= 0) & (sw_y < height)
    sw_output_offset = (batch_idx * output_batch_stride + 
                       channel_idx * output_channel_stride + 
                       sw_y * output_height_stride + 
                       sw_x * output_width_stride)
    sw_value = input_val * sw_weight
    tl.atomic_add(output_ptr + sw_output_offset, sw_value, mask=mask & sw_valid)
    
    # Southeast
    se_valid = (se_x >= 0) & (se_x < width) & (se_y >= 0) & (se_y < height)
    se_output_offset = (batch_idx * output_batch_stride + 
                       channel_idx * output_channel_stride + 
                       se_y * output_height_stride + 
                       se_x * output_width_stride)
    se_value = input_val * se_weight
    tl.atomic_add(output_ptr + se_output_offset, se_value, mask=mask & se_valid)


@triton.jit
def softsplat_backward_input_kernel(
    input_ptr, flow_ptr, grad_output_ptr, grad_input_ptr,
    n_batch, n_channels, height, width,
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    flow_batch_stride, flow_channel_stride, flow_height_stride, flow_width_stride,
    grad_output_batch_stride, grad_output_channel_stride, grad_output_height_stride, grad_output_width_stride,
    grad_input_batch_stride, grad_input_channel_stride, grad_input_height_stride, grad_input_width_stride,
    BLOCK_SIZE: tl.constexpr
):
    """Triton kernel for softsplat backward pass (input gradients)"""
    pid = tl.program_id(axis=0)
    
    total_elements = n_batch * n_channels * height * width
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    
    # Decompose linear index
    batch_idx = idx // (n_channels * height * width)
    remaining = idx % (n_channels * height * width)
    channel_idx = remaining // (height * width)
    remaining = remaining % (height * width)
    y_idx = remaining // width
    x_idx = remaining % width
    
    # Get flow values
    flow_x_offset = (batch_idx * flow_batch_stride + 
                    0 * flow_channel_stride + 
                    y_idx * flow_height_stride + 
                    x_idx * flow_width_stride)
    flow_y_offset = (batch_idx * flow_batch_stride + 
                    1 * flow_channel_stride + 
                    y_idx * flow_height_stride + 
                    x_idx * flow_width_stride)
    
    flow_x = tl.load(flow_ptr + flow_x_offset, mask=mask)
    flow_y = tl.load(flow_ptr + flow_y_offset, mask=mask)
    
    # Calculate output coordinates
    output_x = x_idx.to(tl.float32) + flow_x
    output_y = y_idx.to(tl.float32) + flow_y
    
    # Calculate bilinear interpolation coordinates
    nw_x = tl.floor(output_x).to(tl.int32)
    nw_y = tl.floor(output_y).to(tl.int32)
    ne_x = nw_x + 1
    ne_y = nw_y
    sw_x = nw_x
    sw_y = nw_y + 1
    se_x = nw_x + 1
    se_y = nw_y + 1
    
    # Calculate bilinear weights
    nw_weight = (se_x.to(tl.float32) - output_x) * (se_y.to(tl.float32) - output_y)
    ne_weight = (output_x - sw_x.to(tl.float32)) * (sw_y.to(tl.float32) - output_y)
    sw_weight = (ne_x.to(tl.float32) - output_x) * (output_y - ne_y.to(tl.float32))
    se_weight = (output_x - nw_x.to(tl.float32)) * (output_y - nw_y.to(tl.float32))
    
    # Accumulate gradients
    grad_input_val = tl.zeros_like(input_ptr, dtype=tl.float32)
    
    # Northwest
    nw_valid = (nw_x >= 0) & (nw_x < width) & (nw_y >= 0) & (nw_y < height)
    nw_grad_offset = (batch_idx * grad_output_batch_stride + 
                     channel_idx * grad_output_channel_stride + 
                     nw_y * grad_output_height_stride + 
                     nw_x * grad_output_width_stride)
    nw_grad = tl.load(grad_output_ptr + nw_grad_offset, mask=mask & nw_valid, other=0.0)
    grad_input_val += nw_grad * nw_weight
    
    # Northeast
    ne_valid = (ne_x >= 0) & (ne_x < width) & (ne_y >= 0) & (ne_y < height)
    ne_grad_offset = (batch_idx * grad_output_batch_stride + 
                     channel_idx * grad_output_channel_stride + 
                     ne_y * grad_output_height_stride + 
                     ne_x * grad_output_width_stride)
    ne_grad = tl.load(grad_output_ptr + ne_grad_offset, mask=mask & ne_valid, other=0.0)
    grad_input_val += ne_grad * ne_weight
    
    # Southwest
    sw_valid = (sw_x >= 0) & (sw_x < width) & (sw_y >= 0) & (sw_y < height)
    sw_grad_offset = (batch_idx * grad_output_batch_stride + 
                     channel_idx * grad_output_channel_stride + 
                     sw_y * grad_output_height_stride + 
                     sw_x * grad_output_width_stride)
    sw_grad = tl.load(grad_output_ptr + sw_grad_offset, mask=mask & sw_valid, other=0.0)
    grad_input_val += sw_grad * sw_weight
    
    # Southeast
    se_valid = (se_x >= 0) & (se_x < width) & (se_y >= 0) & (se_y < height)
    se_grad_offset = (batch_idx * grad_output_batch_stride + 
                     channel_idx * grad_output_channel_stride + 
                     se_y * grad_output_height_stride + 
                     se_x * grad_output_width_stride)
    se_grad = tl.load(grad_output_ptr + se_grad_offset, mask=mask & se_valid, other=0.0)
    grad_input_val += se_grad * se_weight
    
    # Store gradient
    grad_input_offset = (batch_idx * grad_input_batch_stride + 
                        channel_idx * grad_input_channel_stride + 
                        y_idx * grad_input_height_stride + 
                        x_idx * grad_input_width_stride)
    tl.store(grad_input_ptr + grad_input_offset, grad_input_val, mask=mask)


@triton.jit
def softsplat_backward_flow_kernel(
    input_ptr, flow_ptr, grad_output_ptr, grad_flow_ptr,
    n_batch, n_channels, height, width,
    input_batch_stride, input_channel_stride, input_height_stride, input_width_stride,
    flow_batch_stride, flow_channel_stride, flow_height_stride, flow_width_stride,
    grad_output_batch_stride, grad_output_channel_stride, grad_output_height_stride, grad_output_width_stride,
    grad_flow_batch_stride, grad_flow_channel_stride, grad_flow_height_stride, grad_flow_width_stride,
    BLOCK_SIZE: tl.constexpr
):
    """Triton kernel for softsplat backward pass (flow gradients)"""
    pid = tl.program_id(axis=0)
    
    total_elements = n_batch * 2 * height * width  # 2 channels for flow (x, y)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < total_elements
    
    # Decompose linear index
    batch_idx = idx // (2 * height * width)
    remaining = idx % (2 * height * width)
    flow_channel_idx = remaining // (height * width)
    remaining = remaining % (height * width)
    y_idx = remaining // width
    x_idx = remaining % width
    
    # Get flow values
    flow_x_offset = (batch_idx * flow_batch_stride + 
                    0 * flow_channel_stride + 
                    y_idx * flow_height_stride + 
                    x_idx * flow_width_stride)
    flow_y_offset = (batch_idx * flow_batch_stride + 
                    1 * flow_channel_stride + 
                    y_idx * flow_height_stride + 
                    x_idx * flow_width_stride)
    
    flow_x = tl.load(flow_ptr + flow_x_offset, mask=mask)
    flow_y = tl.load(flow_ptr + flow_y_offset, mask=mask)
    
    # Calculate output coordinates
    output_x = x_idx.to(tl.float32) + flow_x
    output_y = y_idx.to(tl.float32) + flow_y
    
    # Calculate bilinear interpolation coordinates
    nw_x = tl.floor(output_x).to(tl.int32)
    nw_y = tl.floor(output_y).to(tl.int32)
    ne_x = nw_x + 1
    ne_y = nw_y
    sw_x = nw_x
    sw_y = nw_y + 1
    se_x = nw_x + 1
    se_y = nw_y + 1
    
    # Calculate weight derivatives
    nw_weight_dx = tl.zeros_like(flow_x)
    nw_weight_dy = tl.zeros_like(flow_x)
    ne_weight_dx = tl.zeros_like(flow_x)
    ne_weight_dy = tl.zeros_like(flow_x)
    sw_weight_dx = tl.zeros_like(flow_x)
    sw_weight_dy = tl.zeros_like(flow_x)
    se_weight_dx = tl.zeros_like(flow_x)
    se_weight_dy = tl.zeros_like(flow_x)
    
    if flow_channel_idx == 0:  # x flow channel
        nw_weight_dx = -(se_y.to(tl.float32) - output_y)
        ne_weight_dx = (sw_y.to(tl.float32) - output_y)
        sw_weight_dx = -(output_y - ne_y.to(tl.float32))
        se_weight_dx = (output_y - nw_y.to(tl.float32))
    else:  # y flow channel
        nw_weight_dy = -(se_x.to(tl.float32) - output_x)
        ne_weight_dy = -(output_x - sw_x.to(tl.float32))
        sw_weight_dy = (ne_x.to(tl.float32) - output_x)
        se_weight_dy = (output_x - nw_x.to(tl.float32))
    
    # Accumulate flow gradients across all input channels
    grad_flow_val = tl.zeros_like(flow_x)
    
    for c in range(n_channels):
        # Get input value
        input_offset = (batch_idx * input_batch_stride + 
                       c * input_channel_stride + 
                       y_idx * input_height_stride + 
                       x_idx * input_width_stride)
        input_val = tl.load(input_ptr + input_offset, mask=mask)
        
        # Northwest
        nw_valid = (nw_x >= 0) & (nw_x < width) & (nw_y >= 0) & (nw_y < height)
        nw_grad_offset = (batch_idx * grad_output_batch_stride + 
                         c * grad_output_channel_stride + 
                         nw_y * grad_output_height_stride + 
                         nw_x * grad_output_width_stride)
        nw_grad = tl.load(grad_output_ptr + nw_grad_offset, mask=mask & nw_valid, other=0.0)
        if flow_channel_idx == 0:
            grad_flow_val += input_val * nw_grad * nw_weight_dx
        else:
            grad_flow_val += input_val * nw_grad * nw_weight_dy
        
        # Northeast
        ne_valid = (ne_x >= 0) & (ne_x < width) & (ne_y >= 0) & (ne_y < height)
        ne_grad_offset = (batch_idx * grad_output_batch_stride + 
                         c * grad_output_channel_stride + 
                         ne_y * grad_output_height_stride + 
                         ne_x * grad_output_width_stride)
        ne_grad = tl.load(grad_output_ptr + ne_grad_offset, mask=mask & ne_valid, other=0.0)
        if flow_channel_idx == 0:
            grad_flow_val += input_val * ne_grad * ne_weight_dx
        else:
            grad_flow_val += input_val * ne_grad * ne_weight_dy
        
        # Southwest
        sw_valid = (sw_x >= 0) & (sw_x < width) & (sw_y >= 0) & (sw_y < height)
        sw_grad_offset = (batch_idx * grad_output_batch_stride + 
                         c * grad_output_channel_stride + 
                         sw_y * grad_output_height_stride + 
                         sw_x * grad_output_width_stride)
        sw_grad = tl.load(grad_output_ptr + sw_grad_offset, mask=mask & sw_valid, other=0.0)
        if flow_channel_idx == 0:
            grad_flow_val += input_val * sw_grad * sw_weight_dx
        else:
            grad_flow_val += input_val * sw_grad * sw_weight_dy
        
        # Southeast
        se_valid = (se_x >= 0) & (se_x < width) & (se_y >= 0) & (se_y < height)
        se_grad_offset = (batch_idx * grad_output_batch_stride + 
                         c * grad_output_channel_stride + 
                         se_y * grad_output_height_stride + 
                         se_x * grad_output_width_stride)
        se_grad = tl.load(grad_output_ptr + se_grad_offset, mask=mask & se_valid, other=0.0)
        if flow_channel_idx == 0:
            grad_flow_val += input_val * se_grad * se_weight_dx
        else:
            grad_flow_val += input_val * se_grad * se_weight_dy
    
    # Store flow gradient
    grad_flow_offset = (batch_idx * grad_flow_batch_stride + 
                       flow_channel_idx * grad_flow_channel_stride + 
                       y_idx * grad_flow_height_stride + 
                       x_idx * grad_flow_width_stride)
    tl.store(grad_flow_ptr + grad_flow_offset, grad_flow_val, mask=mask)


class _FunctionSoftsplat(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, flow):
        ctx.save_for_backward(input, flow)
        
        batch_size, n_channels, height, width = input.shape
        assert flow.shape == (batch_size, 2, height, width), f"Flow shape {flow.shape} doesn't match expected {(batch_size, 2, height, width)}"
        
        assert input.is_contiguous(), "Input must be contiguous"
        assert flow.is_contiguous(), "Flow must be contiguous"
        
        output = torch.zeros_like(input)
        
        if input.is_cuda:
            total_elements = batch_size * n_channels * height * width
            BLOCK_SIZE = 256
            num_blocks = triton.cdiv(total_elements, BLOCK_SIZE)
            
            softsplat_forward_kernel[(num_blocks,)](
                input, flow, output,
                batch_size, n_channels, height, width,
                input.stride(0), input.stride(1), input.stride(2), input.stride(3),
                flow.stride(0), flow.stride(1), flow.stride(2), flow.stride(3),
                output.stride(0), output.stride(1), output.stride(2), output.stride(3),
                BLOCK_SIZE=BLOCK_SIZE
            )
        else:
            raise NotImplementedError("CPU implementation not available")
        
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        input, flow = ctx.saved_tensors
        batch_size, n_channels, height, width = input.shape
        
        assert grad_output.is_contiguous(), "Gradient output must be contiguous"
        
        grad_input = torch.zeros_like(input) if ctx.needs_input_grad[0] else None
        grad_flow = torch.zeros_like(flow) if ctx.needs_input_grad[1] else None
        
        if input.is_cuda:
            BLOCK_SIZE = 256
            
            if grad_input is not None:
                total_elements = batch_size * n_channels * height * width
                num_blocks = triton.cdiv(total_elements, BLOCK_SIZE)
                
                softsplat_backward_input_kernel[(num_blocks,)](
                    input, flow, grad_output, grad_input,
                    batch_size, n_channels, height, width,
                    input.stride(0), input.stride(1), input.stride(2), input.stride(3),
                    flow.stride(0), flow.stride(1), flow.stride(2), flow.stride(3),
                    grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
                    grad_input.stride(0), grad_input.stride(1), grad_input.stride(2), grad_input.stride(3),
                    BLOCK_SIZE=BLOCK_SIZE
                )
            
            if grad_flow is not None:
                total_elements = batch_size * 2 * height * width
                num_blocks = triton.cdiv(total_elements, BLOCK_SIZE)
                
                softsplat_backward_flow_kernel[(num_blocks,)](
                    input, flow, grad_output, grad_flow,
                    batch_size, n_channels, height, width,
                    input.stride(0), input.stride(1), input.stride(2), input.stride(3),
                    flow.stride(0), flow.stride(1), flow.stride(2), flow.stride(3),
                    grad_output.stride(0), grad_output.stride(1), grad_output.stride(2), grad_output.stride(3),
                    grad_flow.stride(0), grad_flow.stride(1), grad_flow.stride(2), grad_flow.stride(3),
                    BLOCK_SIZE=BLOCK_SIZE
                )
        else:
            raise NotImplementedError("CPU implementation not available")
        
        return grad_input, grad_flow


def FunctionSoftsplat(tenInput, tenFlow, tenMetric, strType):
    """
    Backward compatible function interface for softsplat
    """
    assert tenMetric is None or tenMetric.shape[1] == 1
    assert strType in ['summation', 'average', 'linear', 'softmax']
    
    if strType == 'average':
        tenInput = torch.cat([tenInput, tenInput.new_ones(tenInput.shape[0], 1, tenInput.shape[2], tenInput.shape[3])], 1)
    
    elif strType == 'linear':
        tenInput = torch.cat([tenInput * tenMetric, tenMetric], 1)
    
    elif strType == 'softmax':
        tenInput = torch.cat([tenInput * tenMetric.exp(), tenMetric.exp()], 1)
    
    tenOutput = _FunctionSoftsplat.apply(tenInput, tenFlow)
    
    if strType != 'summation':
        tenNormalize = tenOutput[:, -1:, :, :]
        tenNormalize = torch.where(tenNormalize == 0.0, torch.ones_like(tenNormalize), tenNormalize)
        tenOutput = tenOutput[:, :-1, :, :] / tenNormalize
    
    return tenOutput


class ModuleSoftsplat(torch.nn.Module):
    """
    Backward compatible module interface for softsplat
    """
    def __init__(self, strType):
        super(ModuleSoftsplat, self).__init__()
        self.strType = strType
    
    def forward(self, tenInput, tenFlow, tenMetric):
        return FunctionSoftsplat(tenInput, tenFlow, tenMetric, self.strType)