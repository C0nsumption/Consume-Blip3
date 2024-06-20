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
model = AutoModelForVision2Seq.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True, use_fast=False, legacy=False)
image_processor = AutoImageProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
tokenizer = model.update_special_tokens(tokenizer)

# Craft a test sample
image_path = './assets/image.png'  # Change the image source to a local image
raw_image = Image.open(image_path).convert('RGB')
query = "Is there a banner in this image? If so, what color is it and what does it say? Ensure to include all letters and"

model = model.cuda()
inputs = image_processor([raw_image], return_tensors="pt", image_aspect_ratio='anyres')
prompt = apply_prompt_template(query)
language_inputs = tokenizer([prompt], return_tensors="pt")
inputs.update(language_inputs)
inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
generated_text = model.generate(**inputs, image_size=[raw_image.size],
                                pad_token_id=tokenizer.pad_token_id,
                                do_sample=False, max_new_tokens=768, top_p=None, num_beams=1,
                                stopping_criteria = [EosListStoppingCriteria()],
                                )
prediction = tokenizer.decode(generated_text[0], skip_special_tokens=True).split("<|end|>")[0]
print("==> prediction: ", prediction)
