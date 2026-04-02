from transformers import AutoTokenizer, AutoModelForCausalLM


## Model IDs: meta-llama/Llama-3.1-8B
## Models IDs: inceptionai/Jais-2-8B-Chat

def get_model(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    return tokenizer, model