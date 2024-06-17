import argparse
from transformers import AutoModelForVision2Seq, AutoTokenizer, AutoImageProcessor, StoppingCriteria
import torch
from PIL import Image
import os

class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[32007]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids

class ImageAnalyzer:
    def __init__(self, model_path):
        self.model = AutoModelForVision2Seq.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, use_fast=False, legacy=False)
        self.image_processor = AutoImageProcessor.from_pretrained(model_path, trust_remote_code=True)
        self.tokenizer = self.model.update_special_tokens(self.tokenizer)
        self.model = self.model.cuda()

    @staticmethod
    def apply_prompt_template(prompt):
        return (
            '\nA chat between a curious user and an artificial intelligence assistant. '
            "The assistant gives helpful, detailed, and polite answers to the user's questions.\n"
            f'\n<image>\n{prompt}\n\n'
        )

    def __call__(self, image_path, query, max_new_tokens=768, num_beams=1):
        raw_image = Image.open(image_path).convert('RGB')
        inputs = self.image_processor([raw_image], return_tensors="pt", image_aspect_ratio='anyres')
        prompt = self.apply_prompt_template(query)
        language_inputs = self.tokenizer([prompt], return_tensors="pt")
        inputs.update(language_inputs)
        inputs = {name: tensor.cuda() for name, tensor in inputs.items()}
        
        generated_text = self.model.generate(**inputs, image_size=[raw_image.size],
                                             pad_token_id=self.tokenizer.pad_token_id,
                                             do_sample=False, max_new_tokens=max_new_tokens, top_p=None, num_beams=num_beams,
                                             stopping_criteria=[EosListStoppingCriteria()])
        prediction = self.tokenizer.decode(generated_text[0], skip_special_tokens=True).split(".")[0]
        return prediction

    def analyze_image(self, image_path, query, max_new_tokens=768, num_beams=1, save_response=False):
        prediction = self.__call__(image_path, query, max_new_tokens, num_beams)
        print(f"==> {os.path.basename(image_path)}: {prediction}")

        if save_response:
            base_path, ext = os.path.splitext(image_path)
            response_path = f"{base_path}.txt"
            with open(response_path, "w") as f:
                f.write(prediction)
            print(f"Response saved to: {response_path}")

    def analyze_directory(self, directory_path, query, max_new_tokens=768, num_beams=1, save_response=False):
        for filename in os.listdir(directory_path):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_path = os.path.join(directory_path, filename)
                self.analyze_image(image_path, query, max_new_tokens, num_beams, save_response)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Image Analysis using Vision2Seq Model')
    parser.add_argument('path', type=str, help='Path to the image file or directory containing images')
    parser.add_argument('query', type=str, help='Query to ask about the image(s)')
    parser.add_argument('--max_new_tokens', type=int, default=768, help='Maximum number of new tokens to generate')
    parser.add_argument('--num_beams', type=int, default=1, help='Number of beams for beam search')
    parser.add_argument('--save_response', action='store_true', help='Save the response to a text file')
    args = parser.parse_args()

    analyzer = ImageAnalyzer("./xgen-mm-phi3-mini-instruct-r-v1")
    
    if os.path.isdir(args.path):
        analyzer.analyze_directory(args.path, args.query, args.max_new_tokens, args.num_beams, args.save_response)
    else:
        analyzer.analyze_image(args.path, args.query, args.max_new_tokens, args.num_beams, args.save_response)
