### Summary
1. Load two model, one to detect vehicle and other to detect license plate.
2. Open video and process each frame
3. Detect vehicle and draw rectangle border around them
4. Detect plate, crop it and use Easy OCR to detect license plate no.
5. Draw rectangle around plate with label as plate no.
6. Write the updated frame to output video.

### Sort
1. Sort is used to give a unique identifier to moving object, in this vehicle.


### Improvements
1. Right now in the output video, the license plate number keeps on changing
2. It happens because OCR reads plate number differently in each frame.
3. So the improvement should remove this con.
4. We can remove it by first storing each frame info in a variable.
5. It would contain frame_id, car_id, plate_number, confidence.
6. Consider all frame, we will find the plate_number with maximum confidence.
7. Now we will read the input video again.
8. Process frame one by one.
9. And write license number with maximum confidence in the output video frame.
