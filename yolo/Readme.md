# YOLO
Yolo is a object detection model. One use case of it is to detect number plate of vehicles.

We can train it on our custom data to particular type of object.

In this example, I have used it to detect book name and its author. It finds the block containing Author Name and Book Name.

## YOLO + OCR 🚀
Using YOLO increase the efficiency of OCR. Because using YOLO we can detect blocks containing the target text, now the OCR can read that block and give the text on it. YOLO increase the performance by reducing the scope of search and extra noise in the image/video.



## This folder
In this folder, I have written code to use YOLO + OCR to detect author name and book name. I have followed https://medium.com/@tapanbabbar/enhance-ocr-with-a-custom-yolov10-ollama-llama-3-1-d13747164c96 to implement it.

## Steps
1. Train YOLO Model for a specific use case
2. Use Model to detect target blocks
3. Use OCR model to extract text from target blocks
4. Use LLM to understand extracted text better and output result in a particular format.


## Learning
1. Training a model takes lot of hours
2. Trained model can be deployed.
3. To use ollama, we need to pull model in the local machine. `ollama pull llama3` was run to use llama3
4. Training data matters a lot for good results.



