import cv2

"""we use media pipe for getting the points of elements of body,we use pretrain NN"""
import mediapipe as mp
import numpy as np
import math
from scipy import io
import time

'''
function angle between points give our the degree between two points 
also we have edge case if the angle negative will be 180 (limits of servo) 
'''


def angleBetweenPoints(a, b):
    deltaY = b["y"] - a["y"]
    deltaX = b["x"] - a["x"]
    degree = math.atan2(deltaX, deltaY) * 180 / math.pi
    if degree < 0:
        return 180
    return degree


'''
main function for the simulation 
'''


def simulation():
    """ init sleep 9 seconds to setting down the camera """
    time.sleep(9)
    print("Starting...")

    """setting empty arrays for r2,r3,r4 will be save the degrees according to time"""
    r2 = []
    r3 = []
    r4 = []

    """consts"""
    """limit the video of max frames for short video"""
    MAX_FRAMES = 250
    mp_pose = mp.solutions.pose
    font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale
    fontScale = 1
    # Blue color in BGR
    color = (0, 255, 0)
    # Line thickness of 2 px
    thickness = 2
    line_thickness = 2

    """starting video from webcam"""
    # For webcam input:
    cap = cv2.VideoCapture(0)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))

    """we record our video to debugging and analyze"""
    out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 10, (frame_width, frame_height))

    """use the model"""
    with mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5) as pose:

        """open camera untill MAX_FRAMES is positive"""
        while cap.isOpened() and MAX_FRAMES >= 0:
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                break

            MAX_FRAMES -= 1

            """we formatting the image to RGB for the model"""
            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False

            """predict our image and getting results"""
            results = pose.process(image)

            height, width, deep = image.shape

            if results.pose_landmarks is None:
                continue

            """taking relative points from the model"""
            leftArm = {"x": int(results.pose_landmarks.landmark[11].x * width),
                       "y": int(results.pose_landmarks.landmark[11].y * height)}
            leftElbow = {"x": int(results.pose_landmarks.landmark[13].x * width),
                         "y": int(results.pose_landmarks.landmark[13].y * height)}
            leftHand = {"x": int(results.pose_landmarks.landmark[15].x * width),
                        "y": int(results.pose_landmarks.landmark[15].y * height)}
            leftHandler = {"x": int(results.pose_landmarks.landmark[19].x * width),
                           "y": int(results.pose_landmarks.landmark[19].y * height)}

            """calculate the angle between the arm and elbow"""
            init_angle = angleBetweenPoints(leftArm, leftElbow)
            middle_angle = angleBetweenPoints(leftElbow, leftHand)
            high_angle = angleBetweenPoints(leftHand, leftHandler)
            """append the results to r2,r3,r4 array of results ,
            we subtract 180 degrees because the simulation Z axis is Top
             and in video Y axis is on top so we shift the axis by 180 degrees """



            """drawing circles on images, the points of arm, elbow and hand"""
            image = cv2.circle(image, (leftArm["x"], leftArm["y"]), 10, (50, 0, 0), thickness)
            image = cv2.circle(image, (leftElbow["x"], leftElbow["y"]), 10, (100, 0, 0), thickness)
            image = cv2.circle(image, (leftHand["x"], leftHand["y"]), 10, (200, 0, 0), thickness)
            image = cv2.circle(image, (leftHandler["x"], leftHandler["y"]), 10, (250, 0, 0), thickness)

            # Draw the pose annotation on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            """drawing lines between points"""
            cv2.line(image, (leftArm["x"], leftArm["y"]), (leftElbow["x"], leftElbow["y"]), (0, 100, 0),
                     thickness=line_thickness)
            cv2.line(image, (leftElbow["x"], leftElbow["y"]), (leftHand["x"], leftHand["y"]), (0, 200, 0),
                     thickness=line_thickness)
            cv2.line(image, (leftHand["x"], leftHand["y"]), (leftHandler["x"], leftHandler["y"]), (0, 255, 0),
                     thickness=line_thickness)

            """drawing text of angles and X and Y positions"""
            image = cv2.putText(image,
                                str(int(init_angle)) + str("   x: " + str(leftArm["x"]) + ", y:" + str(leftArm["y"])),
                                (leftArm["x"], leftArm["y"]), font,
                                fontScale, color, thickness, cv2.LINE_AA)

            image = cv2.putText(image, str(int(middle_angle)) + str(
                "   x: " + str(leftElbow["x"]) + ", y:" + str(leftElbow["y"])), (leftElbow["x"], leftElbow["y"]), font,
                                fontScale, color, thickness, cv2.LINE_AA)

            image = cv2.putText(image,
                                str(int(high_angle)) + str("   x: " + str(leftHand["x"]) + ", y:" + str(leftHand["y"])),
                                (leftHand["x"], leftHand["y"]), font,
                                fontScale, color, thickness, cv2.LINE_AA)

            r2.append(180 - int(init_angle))
            r3.append(180 - int(middle_angle))
            r4.append(180 - int(high_angle))

            """show image"""
            cv2.imshow('MediaPipe Pose', image)
            out.write(image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    """shut down the video and record"""
    out.release()
    cap.release()

    """return the results"""
    return [r2, r3, r4]


"""Main function, running the simulation ,
getting the results and save them on matlab Matrix that called out.mat"""
[r2, r3, r4] = simulation()
r2 = np.array(r2)
r3 = np.array(r3)
r4 = np.array(r4)
io.savemat('./out.mat', mdict={'r2': r2, 'r3': r3, 'r4': r4})
