from transformers import AutoTokenizer
import transformers
import torch
import json

model = "meta-llama/Llama-2-7b-hf"

json_file_path = './datasets/squadv2/concatenated_300_sentences.json'

with open(json_file_path, 'r') as input_texts:
    squad_data = json.load(input_texts)

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

pipeline.enable_cpu_offload()
#nope! 

sequences = pipeline(
    squad_data[0],
    do_sample=True,
    top_k=10,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)

print(sequences[0])