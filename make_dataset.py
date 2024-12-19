import cv2
import os

INPUT_FOLDER = "../2_17"
OUTPUT_FOLDER = "../2_17_resized"

# Make output folder
if not os.path.exists(OUTPUT_FOLDER):
  os.makedirs(OUTPUT_FOLDER)

# Write to txt file written file paths
OUTPUT_TXT = "2_17.txt"

for file in os.listdir(INPUT_FOLDER):
  img = cv2.imread(os.path.join(INPUT_FOLDER, file))
  
  img = img[:416, :416]
  
  cv2.imwrite(os.path.join(OUTPUT_FOLDER, file), img)
  
  with open(OUTPUT_TXT, "a") as f:
    f.write(os.path.join(OUTPUT_FOLDER, file) + "\n")

