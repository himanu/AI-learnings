from ultralytics import YOLO


# Load the YOLO10 pre-trained model
model = YOLO("yolov10n.pt")

# Train the model
    # epochs denotes number of time model will be trained on the dataset
model.train(data="datasets/data.yaml", epochs=50, imgsz=640)

# check the model performance
model.val()