# API Documentation

This document provides detailed information about the classes and methods used in the BLIP3 autocaptioning tools.

## ImageAnalyzer Class

### Methods

#### `__init__(self, model_path)`

Initializes the ImageAnalyzer class with the specified model path.

- `model_path` (str): Path to the model.

#### `apply_prompt_template(prompt)`

Applies a prompt template to the given prompt.

- `prompt` (str): The prompt to apply the template to.
- Returns (str): The formatted prompt.

#### `__call__(self, image_path, query, max_new_tokens=768, num_beams=1)`

Generates a prediction for the given image and query.

- `image_path` (str): Path to the image file.
- `query` (str): Query to ask about the image.
- `max_new_tokens` (int, optional): Maximum number of new tokens to generate (default: 768).
- `num_beams` (int, optional): Number of beams for beam search (default: 1).
- Returns (str): The prediction.

#### `analyze_image(self, image_path, query, max_new_tokens=768, num_beams=1, save_response=False)`

Analyzes a single image and prints the prediction.

- `image_path` (str): Path to the image file.
- `query` (str): Query to ask about the image.
- `max_new_tokens` (int, optional): Maximum number of new tokens to generate (default: 768).
- `num_beams` (int, optional): Number of beams for beam search (default: 1).
- `save_response` (bool, optional): Save the response to a text file (default: False).

#### `analyze_directory(self, directory_path, query, max_new_tokens=768, num_beams=1, save_response=False)`

Analyzes all images in a directory and prints the predictions.

- `directory_path` (str): Path to the directory containing images.
- `query` (str): Query to ask about the images.
- `max_new_tokens` (int, optional): Maximum number of new tokens to generate (default: 768).
- `num_beams` (int, optional): Number of beams for beam search (default: 1).
- `save_response` (bool, optional): Save the responses to text files (default: False).

## EosListStoppingCriteria Class

### Methods

#### `__init__(self, eos_sequence=[32007])`

Initializes the EosListStoppingCriteria class with the specified end-of-sequence token.

- `eos_sequence` (list, optional): List of end-of-sequence tokens (default: [32007]).

#### `__call__(self, input_ids, scores, **kwargs)`

Checks if the end-of-sequence token is in the input IDs.

- `input_ids` (torch.LongTensor): Input IDs.
- `scores` (torch.FloatTensor): Scores.
- Returns (bool): True if the end-of-sequence token is in the input IDs, False otherwise.
