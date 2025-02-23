import cv2
import easyocr
import numpy as np
from ultralytics import YOLO
from sort.sort import *  # Import SORT tracking

# Load YOLOv8 model
model = YOLO("yolov8n.pt")  # You can use a custom-trained model

# Load plate number detector model
licensePlateDetector = YOLO("license_plate_detector.pt")

# Initialize SORT Tracker
tracker = Sort()

# Initialize OCR
reader = easyocr.Reader(['en'])

def preprocess_plate(plate_img):
    """Preprocess the license plate for better OCR accuracy."""
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY_INV, 11, 2)
    resized = cv2.resize(thresh, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    kernel = np.ones((3, 3), np.uint8)
    processed = cv2.dilate(resized, kernel, iterations=1)
    return processed

def process_video(input_video, output_video=None):
    """Process video to detect, track, and recognize license plates."""
    cap = cv2.VideoCapture(input_video)
    width, height = int(cap.get(3)), int(cap.get(4))
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    # Define video writer if saving output
    if output_video:
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  # End of video
        
        # Run YOLOv8 detection
        results = model(frame)
        
        detections = []
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])  # Bounding box coordinates
                confidence = box.conf[0].item()  # Confidence score
                
                if confidence > 0.5:
                    detections.append([x1, y1, x2, y2, confidence])  # Add detection

        # Convert to NumPy array for SORT
        detections = np.array(detections)

        if detections is None or detections.size == 0:
            continue
        # Update tracker
        tracked_objects = tracker.update(detections)

        # detect vehicle and mark them
        for obj in tracked_objects:
            x1, y1, x2, y2, obj_id = map(int, obj)  # Object coordinates + ID

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {obj_id}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.8, (0, 255, 0), 2)

        # Detect license plates
        plate_results = licensePlateDetector(frame)

        for r in plate_results:
            for box in r.boxes.xyxy:
                x1, y1, x2, y2 = map(int, box[:4])

                # Crop license plate from frame
                plate_img = frame[y1:y2, x1:x2]

                # Check if the image is valid
                if plate_img is None or plate_img.size == 0:
                    print(f"Warning: Empty image for plate at ({x1}, {y1}, {x2}, {y2})")
                    continue

                # Run OCR to extract plate number
                plate_number = reader.readtext(plate_img, detail=0)

                # Draw plate bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # cv2.putText(frame, "Plate", (x1, y1 - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                # Display extracted plate number
                plate_text = " ".join(plate_number) if plate_number else "Unknown"
                cv2.putText(frame, plate_text, (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

        # Show the frame with detections
        cv2.imshow("License Plate Tracking", frame)
        cv2.waitKey(1)  # Add this to prevent errors

        # Save frame to output video
        if output_video:
            out.write(frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to stop
            break

    # Release resources
    cap.release()
    if output_video:
        out.release()
    cv2.destroyAllWindows()

# Run the function
process_video("a.mp4", "output_video.mp4")  # Replace with actual file paths
