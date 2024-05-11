from layersfusion.layers.operations import *
from layersfusion.utils import *
from transformers.models.llama import LlamaForCausalLM
from transformers import LlamaTokenizer
import torch

if __name__ == "__main__":
    model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"

    model = LlamaForCausalLM.from_pretrained(
        pretrained_model_name_or_path=model_id,device_map="mps",torch_dtype=torch.bfloat16
    )
    tokenizer = LlamaTokenizer.from_pretrained(model_id)

    first_part = list(model.model.layers[:12])

    fusion_part1 = list(model.model.layers[12:14])

    cont_part = list(model.model.layers[14:24])

    fusion_part2 = list(model.model.layers[24:27] )

    last_part = list( model.model.layers[27:] )


    fused_layers = first_part + [ layer_avg_stack(model,fusion_part1) ] + cont_part + [ layer_avg_stack(model,fusion_part2) ] + last_part


    module_list_reconstruction(model,fused_layers)
