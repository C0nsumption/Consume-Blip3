# Usage Guide

This guide provides detailed instructions on how to use the BLIP3 autocaptioning tools.

## Running the Script

You can run the script on a single image or a directory containing multiple images.

### Single Image

To analyze a single image, run the following command:

```sh
python analyze.py path/to/image.jpg "Describe the image"
```

### Directory of Images

To analyze all images in a directory, run the following command:

```sh
python analyze.py path/to/directory "Describe the image"
```

### Saving Responses

To save the AI's responses to text files, add the `--save_response` flag:

```sh
python analyze.py path/to/image.jpg "Describe the image" --save_response
```

## Command Line Arguments

- `path`: Path to the image file or directory containing images.
- `query`: Query to ask about the image(s).
- `--max_new_tokens`: Maximum number of new tokens to generate (default: 768).
- `--num_beams`: Number of beams for beam search (default: 1).
- `--save_response`: Save the response to a text file.

## Example Outputs

### Single Image

Command:

```sh
python analyze.py example.jpg "Describe the image"
```

Output:

```
==> example.jpg: The image captures a serene scene of a white crane, its wings spread wide in a display of majesty, standing on the shore of a tranquil lake
```

### Directory of Images

Command:

```sh
python analyze.py images/ "Describe the images" --save_response
```

Output:

```
==> image1.jpg: The image captures a serene scene of a white crane, its wings spread wide in a display of majesty, standing on the shore of a tranquil lake
Response saved to: images/image1.txt
==> image2.jpg: The image portrays a woman lying on a grassy field
Response saved to: images/image2.txt
```

## Troubleshooting

If you encounter any issues, please check the following:

- Ensure that all dependencies are installed correctly.
- Verify that the paths to the images are correct.
- Check the console output for any error messages and follow the suggestions provided.

For further assistance, feel free to open an issue on the [GitHub repository](https://github.com/C0nsumption/Consume-Blip3/issues).

---

Happy autocaptioning!