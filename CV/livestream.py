from model import Model, putTextOnImage 
import cv2
import time

cap = cv2.VideoCapture(0, cv2.CAP_FFMPEG)
count = 0
while count < 10:
  cap = cv2.VideoCapture(count)

  if not (cap.isOpened()):
    print("Could not open video device ", count)
    count += 1
  
  break
print("It was opened on video device ", count)
  
model = Model()

# Set the camera format to MJPG
fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MJPG FOURCC code
cap.set(cv2.CAP_PROP_FOURCC, fourcc)

# Set the resolution to 800x600
WIDTH = 1280
HEIGHT = 800
print(cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH))
print(cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT))


cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0) 
cap.set(cv2.CAP_PROP_EXPOSURE, 5)

# Saturate image
cap.set(cv2.CAP_PROP_SATURATION, 128)  # Replace 'value' with a range from 0 to 100

# Set desired FPS
desired_fps = 100
cap.set(cv2.CAP_PROP_FPS, desired_fps)

# Print the resolution of the camera
print(f"Resolution: {cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

#print the FPS of the camera
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")

while True:
    prev_time = time.perf_counter_ns()
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
      
    # Process frame and get boxes
    boxes = model.processInput(frame)
    
    time_diff = (time.perf_counter_ns() - prev_time) * 1.0 / 1e9

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
