import cv2
import threading
import queue
import time
from model import Model, putTextOnImage

class FrameBuffer:
    def __init__(self, buffer_size=10):
        self.buffer = queue.Queue(maxsize=buffer_size)
        self.cap = cv2.VideoCapture(0)  # Use your video source (e.g., file or camera)
        # Set the camera format to MJPG
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')  # MJPG FOURCC code
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        # Set the resolution to 800x600
        WIDTH = 1280
        HEIGHT = 800
        print(self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH))
        print(self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT))


        self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0) 
        self.cap.set(cv2.CAP_PROP_EXPOSURE, 5)

        # Saturate image
        self.cap.set(cv2.CAP_PROP_SATURATION, 128)  # Replace 'value' with a range from 0 to 100

        # Set desired FPS
        desired_fps = 100
        self.cap.set(cv2.CAP_PROP_FPS, desired_fps)

        # Print the resolution of the camera
        print(f"Resolution: {self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)}x{self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)}")

        #print the FPS of the camera
        print(f"FPS: {self.cap.get(cv2.CAP_PROP_FPS)}")
        
        self.stop_thread = False
        self.capture_thread = threading.Thread(target=self.capture_frames)
        self.capture_thread.start()

    def capture_frames(self):
        prev_second = time.perf_counter_ns() // 1e9
        frame_count = 0
        
        
        while not self.stop_thread:
            ret, frame = self.cap.read()
            frame_count += 1
            
            if(time.perf_counter_ns() // 1e9 != prev_second):
                print(f"Frames in the last second: {frame_count} ------------------------------------------")
                frame_count = 0
                prev_second = time.perf_counter_ns() // 1e9
            
            if ret:
                if self.buffer.full():
                    self.buffer.get()  # Remove the oldest frame if buffer is full
                self.buffer.put(frame)

    def get_frame(self):
        # Get the next frame from the buffer (non-blocking)
        if not self.buffer.empty():
            return self.buffer.get()
        return None

    def stop(self):
        self.stop_thread = True
        self.capture_thread.join()
        self.cap.release()

def printTemperatures():
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

# Example usage
frame_buffer = FrameBuffer(buffer_size=10)

model = Model()

frame_count = 0
prev_second = time.perf_counter_ns() // 1e9
model_times = []

while True:
    frame = frame_buffer.get_frame()
    if frame is not None:
        frame_count += 1

        if(time.perf_counter_ns() // 1e9 != prev_second):
            # printTemperatures()
            print(f"Frames processed in the last second: {frame_count} ooooooooooooooooooooooooo")
            if(len(model_times) > 0):
                print(f"Average processing time: {sum(model_times) / len(model_times) / 1e6} ms")
                print("Max processing time: ", max(model_times) / 1e6, "ms")
                print("Cummulative time: ", sum(model_times) / 1e9, "s")
            print()
            
            frame_count = 0
            prev_second = time.perf_counter_ns() // 1e9
            model_times = []
        
        # Process frame and get boxes
        start_time = time.perf_counter_ns()
        boxes = model.processInput(frame)
        end_time = time.perf_counter_ns()
        model_times.append(end_time - start_time)
        
        # Display FPS on the frame
        frame = putTextOnImage(frame, boxes)
        
        # Process the frame
        cv2.imshow('Processed Frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Stop the capture thread when done
frame_buffer.stop()
cv2.destroyAllWindows()
