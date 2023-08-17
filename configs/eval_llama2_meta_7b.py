from mmengine.config import read_base

with read_base():
    from .models.llama2_7b_chat import models

    from .datasets.humaneval.humaneval_gen import humaneval_datasets
datasets = [*humaneval_datasets]

#     from .datasets.mmlu.mmlu_gen import mmlu_datasets
# datasets = [*mmlu_datasets]


# from opencompass.models import HuggingFaceCausalLM