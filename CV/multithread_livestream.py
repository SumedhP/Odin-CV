import cv2
import time
import multiprocessing
from model import Model, putTextOnImage

def setupCamera():
  cap = cv2.VideoCapture(0, cv2.CAP_FFMPEG)
  count = 0
  while count < 10:
      cap = cv2.VideoCapture(count)
      if not cap.isOpened():
          print("Could not open video device ", count)
          count += 1
      else:
          break
  print("It was opened on video device ", count)
  
  # Set the camera format to MJPG
  fourcc = cv2.VideoWriter_fourcc(*'MJPG')
  cap.set(cv2.CAP_PROP_FOURCC, fourcc)

  # Set the resolution to 800x600
  WIDTH = 1280
  HEIGHT = 800
  cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
  cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)

  # Set desired FPS
  desired_fps = 100
  cap.set(cv2.CAP_PROP_FPS, desired_fps)

  # Set other camera properties
  cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
  cap.set(cv2.CAP_PROP_EXPOSURE, 5)
  cap.set(cv2.CAP_PROP_SATURATION, 128)

  return cap

# Function to capture frames from the camera
def capture_frames(cap, frame_queue):
    print("Started capturing frames")
    while True:
        ret, frame = cap.read()
        print("Captured frame", frame.shape)
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame_queue.put(frame)

# Function to process frames using the model
def process_frames(frame_queue, model):
    model_times = []
    frame_count = 0
    prev_second = time.perf_counter_ns() // 1e9
    
    print("Started processing frames")

    while True:
        # print("In loop to process frames")
        # print("The frame queue is empty: ", frame_queue.empty())
        if not frame_queue.empty():
            print("OMG WE FOUND A FRAME")
            frame = frame_queue.get()
            
            cv2.imshow('frame1', frame)
            cv2.waitKey(1)

            # Process frame and get boxes
            print("About to run the model")
            start_time = time.perf_counter_ns()
            boxes = model.processInput(frame)
            end_time = time.perf_counter_ns()
            
            print("Managed to run the model")

            model_times.append(end_time - start_time)
            
            print("It took ", end_time - start_time, " to run the model")

            # Display FPS on the frame
            frame = putTextOnImage(frame, boxes)

            print("We are about to show the frame")
            # Show frame
            cv2.imshow('frame2', frame)
            print("We showed the frame")

            frame_count += 1

            if time.perf_counter_ns() // 1e9 != prev_second:
                print(f"Frames in the last second: {frame_count}")
                print(f"Average processing time: {sum(model_times) / len(model_times) / 1e6} ms")
                frame_count = 0
                prev_second = time.perf_counter_ns() // 1e9
                model_times = []

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def main():
    cap = setupCamera()
    model = Model()

    # Create a queue to hold frames
    frame_queue = multiprocessing.Queue(10)

    # Create processes
    capture_process = multiprocessing.Process(target=capture_frames, args=(cap, frame_queue))
    process_process = multiprocessing.Process(target=process_frames, args=(frame_queue, model))

    # Start processes
    capture_process.start()
    process_process.start()

    # Wait for processes to finish
    capture_process.join()
    process_process.join()

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
