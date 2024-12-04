import cv2

from Match import Match, Point

# Goal is to take in a set of points on the screen and return the world position of the object using solvePnP

camera_matrix = [
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]

distortion_coefficients = [0.0, 0.0, 0.0, 0.0, 0.0]

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
