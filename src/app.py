from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from PIL import Image
import io
import os
from datetime import datetime
from analyze import ImageAnalyzer  # Ensure the import path is correct

app = FastAPI()
analyzer = ImageAnalyzer("./xgen-mm-phi3-mini-instruct-r-v1")

# Function to create a folder with today's date and save images with incrementing filenames
def save_image_and_prediction(image: Image.Image, prediction: str, base_path="output"):
    # Get today's date in YYYY-MM-DD format
    date_str = datetime.now().strftime("%Y-%m-%d")
    # Create a directory for today's date
    date_dir = os.path.join(base_path, date_str)
    os.makedirs(date_dir, exist_ok=True)

    # Get the next available index for images
    existing_files = [f for f in os.listdir(date_dir) if f.endswith(".png")]
    if existing_files:
        existing_files.sort()
        next_index = int(existing_files[-1].split(".")[0]) + 1
    else:
        next_index = 0

    # Save the image with an incrementing filename
    image_filename = f"{next_index:06}.png"  # Incrementing filenames with zero padding
    image_filepath = os.path.join(date_dir, image_filename)
    image.save(image_filepath)
    print(f"Saved image: {image_filepath}")

    # Save the prediction to a text file
    prediction_filename = f"{next_index:06}.txt"
    prediction_filepath = os.path.join(date_dir, prediction_filename)
    with open(prediction_filepath, "w") as f:
        f.write(prediction)
    print(f"Saved prediction: {prediction_filepath}")

@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), query: str = Form(...), max_new_tokens: int = Form(768), num_beams: int = Form(1), save_response: bool = Form(False)):
    image = Image.open(io.BytesIO(await file.read())).convert('RGB')
    prediction = analyzer(image, query, max_new_tokens, num_beams)

    if save_response:
        save_image_and_prediction(image, prediction)

    return JSONResponse(content={"prediction": prediction})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
