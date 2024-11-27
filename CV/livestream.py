from model import Model, putTextOnImage 
import cv2
import time

cap = cv2.VideoCapture(0)

if not (cap.isOpened()):
  print("Could not open video device")
  
model = Model()

# Print the resolution of the camera
print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

#print the FPS of the camera
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    start_time = time.perf_counter_ns()
    
    # Process frame and get boxes
    boxes = model.processInput(frame)
    
    end_time = time.perf_counter_ns()
    time_diff = (end_time - start_time) * 1.0 / 1e9

    print(f"\rTime taken: {time_diff}", end="")
    fps = 1 / time_diff if time_diff > 0 else 0

    # Display FPS on the frame
    frame = putTextOnImage(frame, boxes)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
