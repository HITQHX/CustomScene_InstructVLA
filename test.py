import torch
from vla.instructvla_eagle_dual_sys_v2_meta_query_v2 import load, load_vla
from PIL import Image
import numpy as np

model_path = '/root/autodl-tmp/InstructVLA/model/instructvla_finetune_v2_xlora_freeze_head_instruction/checkpoints/step-013500-epoch-01-loss=0.1093.pt'

# Load Stage-2 (Generalist) model
model = load_vla(model_path, stage="stage2").eval().to(torch.bfloat16).cuda()

messages = [
    {"content": "You are a helpful assistant."},  # system
    {
        "role": "user",
        "content": "Can you describe the main idea of this image?",
        "image": [{'np_array': np.asarray(Image.open("./asset/teaser.png"))}]
    }
]

# Preprocess input
inputs = model.processor.prepare_input(dict(prompt=messages))
autocast_dtype = torch.bfloat16

with torch.autocast("cuda", dtype=autocast_dtype, enabled=True):
    output = model.vlm.generate(
        input_ids=inputs['input_ids'].cuda(),
        attention_mask=inputs['attention_mask'].cuda(),
        pixel_values=inputs['pixel_values'].cuda(),
        max_new_tokens=200,
        output_hidden_states=False,
    )

response = model.processor.tokenizer.decode(output[0])
print(response)