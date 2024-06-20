from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria
import torch
from PIL import Image

# Define the prompt template
def apply_prompt_template(prompt):
    s = (
        '<|system|>\nA chat between a curious user and an artificial intelligence assistant. '
        "The assistant gives helpful, detailed, and polite answers to the user's questions.<|end|>\n"
        f'<|user|>\n<image>\n{prompt}<|end|>\n<|assistant|>\n'
    )
    return s 

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[32007]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids      

# Load models
model_name_or_path = "./xgen-mm-phi3-mini-instruct-r-v1"
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_TYPE = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, trust_remote_code=True).eval().to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False)
image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = model.update_special_tokens(tokenizer)

def run_chat():
    while True:
        image_path = input("Image path >>>>> ")
        if image_path == '':
            print('You did not enter an image path, the following will be a plain text conversation.')
            image = None
        else:
            print(image_path)
            image = Image.open(image_path).convert('RGB')

        history = []

        while True:
            query = input("Human: ")
            if query == "clear":
                break

            if image is None:
                print("Image not provided. Please provide an image path to continue.")
                break

            inputs = image_processor([image], return_tensors="pt", image_aspect_ratio='anyres')
            image_size = [image.size]
            prompt = apply_prompt_template(query)
            language_inputs = tokenizer([prompt], return_tensors="pt")
            inputs.update(language_inputs)
            inputs = {name: tensor.to(DEVICE) for name, tensor in inputs.items()}

            gen_kwargs = {
                "max_new_tokens": 768,
                "pad_token_id": tokenizer.pad_token_id,
                "do_sample": False,
                "num_beams": 1,
                "stopping_criteria": [EosListStoppingCriteria()],
            }

            with torch.no_grad():
                outputs = model.generate(**inputs, image_size=image_size, **gen_kwargs)
                response = tokenizer.decode(outputs[0], skip_special_tokens=True).split("<|end|>")[0]
                print("Assistant:", response)
            history.append((query, response))

if __name__ == "__main__":
    run_chat()
