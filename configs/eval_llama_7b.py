from mmengine.config import read_base

with read_base():
    from .datasets.humaneval.humaneval_gen import humaneval_datasets
datasets = [*humaneval_datasets]

from opencompass.models import HuggingFaceCausalLM

# LLaMA 7B
llama7b=dict(
        type=HuggingFaceCausalLM,
        abbr='llama-7b-hf',
        path="/home/hhl/pretrained_models/llama-7b-hf-transformers-4.29",
        tokenizer_path='/home/hhl/pretrained_models/llama-7b-hf-transformers-4.29',
        tokenizer_kwargs=dict(padding_side='left',
                              truncation_side='left',
                              use_fast=False,
                              ),
        max_out_len=100,
        max_seq_len=2048,
        batch_size=8,
        model_kwargs=dict(device_map='auto'),
        batch_padding=False, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1),
    )

models =[llama7b]