import gc
import torch
from torch import nn
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

def module_list_reconstruction(model, layers):
    # Free the existing layers first
    model.model.layers = None
    gc.collect()

    # Update the number of hidden layers in the model's configuration
    model.config.num_hidden_layers = len(layers)

    # Prepare to replace and copy layers simultaneously
    with torch.no_grad():
        # Create and copy new layers directly into the model
        model.model.layers = nn.ModuleList()
        for idx, old_layer in enumerate(layers):
            # Create a new layer according to the configuration, and copy parameters from the old layer
            new_layer = LlamaDecoderLayer(model.config, idx).to(model.device.type).to(model.config.torch_dtype)
            for param_old, param_new in zip(old_layer.parameters(), new_layer.parameters()):
                param_new.data.copy_(param_old.data)
            # Append the newly created and populated layer to the model's layers
            model.model.layers.append(new_layer)

    return True

def layer_avg_stack(model,layers):
    # Stack parameters from all layers for efficient computation
    stacked_params = {}

    # Iterate over each layer and parameters, stacking them
    for layer in layers:
        for name, param in layer.named_parameters():
            if name not in stacked_params:
                stacked_params[name] = []
            stacked_params[name].append(param.data)

    # Convert lists to tensors and compute the mean
    stacked_params = {name: torch.stack(p_list).mean(0) for name, p_list in stacked_params.items()}

    # Apply the averaged weights to a specific layer
    target_layer = LlamaDecoderLayer(model.config, 0).to(model.device.type).to(model.config.torch_dtype)
    for name, param in target_layer.named_parameters():
        if name in stacked_params:
            param.data.copy_(stacked_params[name])

    return target_layer

