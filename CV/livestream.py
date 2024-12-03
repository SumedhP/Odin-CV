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

frame_count = 0
prev_second = time.perf_counter_ns() // 1e9

model_times = []

while True:
    frame_count += 1

    if(time.perf_counter_ns() // 1e9 != prev_second):
        import subprocess
        temp1 = subprocess.run(['cat', '/sys/devices/virtual/thermal/thermal_zone1/temp'], stdout=subprocess.PIPE)
        temp2 = subprocess.run(['cat', '/sys/devices/virtual/thermal/thermal_zone2/temp'], stdout=subprocess.PIPE)
        temp3 = subprocess.run(['cat', '/sys/devices/virtual/thermal/thermal_zone3/temp'], stdout=subprocess.PIPE)
        # Get values and average
        temp1 = int(temp1.stdout) / 1000.0
        temp2 = int(temp2.stdout) / 1000.0
        temp3 = int(temp3.stdout) / 1000.0
        print(f"Temperature: {temp1}°C, {temp2}°C, {temp3}°C")
        print(f"Average temperature: {(temp1 + temp2 + temp3) / 3}°C")
        print(f"Frames in the last second: {frame_count}")
        
        print(f"Average processing time: {sum(model_times) / len(model_times) / 1e6} ms")
        
        frame_count = 0
        prev_second = time.perf_counter_ns() // 1e9
        model_times = []
        
    ret, frame = cap.read()
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
      
    # Process frame and get boxes
    start_time = time.perf_counter_ns()
    boxes = model.processInput(frame)
    end_time = time.perf_counter_ns()
    model_times.append(end_time - start_time)
    
    # Display FPS on the frame
    frame = putTextOnImage(frame, boxes)

    # Show frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
