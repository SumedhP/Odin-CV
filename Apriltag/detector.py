"""
This model takes an image and returns the pose of any detected AprilTags in the image.
"""
import cv2

TAG_FAMILY = cv2.aruco.DICT_APRILTAG_36h11

detector_settings = cv2.aruco.DetectorParameters()
detector_settings.aprilTagQuadDecimate = 3
detector_settings.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG

# This one is an unknonwn parameter, internet it makes better detection, prob slower
detector_settings.adaptiveThreshWinSizeStep = 1 

detector = cv2.aruco.ArucoDetector(TAG_FAMILY, detector_settings)

def detect(img):
  pass
