import cv2
import numpy as np
from Match import Match, Point


# Goal is to take in a set of points on the screen and return the world position of the object using solvePnP


# import camera matrix and distortion coefficients from calibration
CALIBRATION_PATH = "../camera_calibration_calibdb.json"

# get the values from the calibration file
import json
with open(CALIBRATION_PATH, "r") as f:
    calibration = json.load(f)

camera_matrix = calibration["camera_matrix"]["data"]
camera_matrix = np.array(camera_matrix).reshape((3, 3))

distortion_coefficients = calibration["distortion_coefficients"]["data"]
distortion_coefficients = np.array(distortion_coefficients).reshape((5, 1))

print("Camera Matrix:\n", camera_matrix)
print()
print("Distortion matrix:\n", distortion_coefficients)


# Actual plate size is 135mm x 135mm
plate_size = 135

def world_position(detected_plate: Match):
    image_points = [
        (detected_plate.points[0].x, detected_plate.points[0].y),
        (detected_plate.points[1].x, detected_plate.points[1].y),
        (detected_plate.points[2].x, detected_plate.points[2].y),
        (detected_plate.points[3].x, detected_plate.points[3].y),
    ]

    # The plate is 135mm x 135mm
    object_points = [
        (0, 0, 0),
        (plate_size, 0, 0),
        (plate_size, plate_size, 0),
        (0, plate_size, 0),
    ]

    success, rvec, tvec = cv2.solvePnP(
        object_points,
        image_points,
        camera_matrix,
        distortion_coefficients,
    )
    
    print(success)

    return rvec, tvec  
