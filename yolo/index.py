from ultralytics import YOLO
import cv2
import easyocr
import ollama


# Load the YOLO10 pre-trained model
# model = YOLO("yolov10n.pt")

# Train the model
    # epochs denotes number of time model will be trained on the dataset
# model.train(data="datasets/data.yaml", epochs=50, imgsz=640)

# model.pt is the trained model
model = YOLO("model.pt")

# Initialize EasyOCR reader
reader = easyocr.Reader(["en"])  # Use English language model

# use model to detect block containing author name and book name

# Load the image
image_path = "book-cover.jpg"
image = cv2.imread(image_path)

# Run inference on the image
results = model(image)

text_on_book = ""
# Iterate through detected text blocks
for result in results:
    for box in result.boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])  # Get bounding box coordinates
        
        # Crop the detected text region
        text_region = image[y1:y2, x1:x2]

        # Convert to grayscale for better OCR
        gray = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)

        # Use EasyOCR to extract text
        extracted_text = reader.readtext(gray, detail=0)

        # Print extracted text
        detected_text = " ".join(extracted_text)  # Join words into a sentence
        print(f"Detected Text: {detected_text}")
        text_on_book += detected_text

        # Draw bounding box & label on image
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(image, detected_text, (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Show the output image
cv2.imshow("Detected Text", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally, save the output image
cv2.imwrite("output_with_text_easyocr.png", image)

# use LLM Model to get author name and book name from extracted text

# LLaMA prompt
prompt = f"""
        - Below is a text extracted from an OCR. The text may contain mentions of famous books and their corresponding authors.
        - Some words may be slightly misspelled or out of order.
        - Your task is to identify the book titles and corresponding authors from the text.
        - Output the text in the format: '<Book Name> By <Author Name>' '<Book Summary>'
        - Do not generate any other text except the book title, the author and the summary.

        TEXT:
        {text_on_book}
"""

# Use LLaMA to predict book & author
response = ollama.chat(
    model="llama3",
    messages=[{"role": "user", "content": prompt}]
)

# Extract and print response
response_text = response['message']['content'].strip()
print(f"\n📚 Predicted Book & Author:\n{response_text}")