#!/usr/bin/env python3

import rospy
import rospkg

import cv2
import mediapipe as mp

from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

from hand_landmarker.msg import HandLandmark, HandLandmarks
from geometry_msgs.msg import Point

# Google's code for drawing landmarks on image
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def draw_landmarks_on_image(rgb_image, detection_result):
  hand_landmarks_list = detection_result.hand_landmarks
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    hand_landmarks = hand_landmarks_list[idx]
    handedness = handedness_list[idx]

    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape
    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image

from std_msgs.msg import Header
class MsgHeaderHelper:
    def __init__(self):
        self.seq = 0
    
    def header(self, frame_id = '', increment_seq = True):
        header = Header()
        header.seq = self.seq
        header.stamp = rospy.Time.now()
        header.frame_id = frame_id

        if increment_seq: self.seq += 1
        return header

class MPHandLandmarker:
    def __init__(self):
        self.landmarker = mp.tasks.vision.HandLandmarker.create_from_options(
            mp.tasks.vision.HandLandmarkerOptions(
                base_options = mp.tasks.BaseOptions(
                    model_asset_path = rospkg.RosPack().get_path('hand_landmarker') + '/models/hand_landmarker.task'
                ),
                running_mode = mp.tasks.vision.RunningMode.LIVE_STREAM,
                result_callback = self.landmarker_cb
            )
        )

        self.bridge = CvBridge()

        self.last_img_stamp = 0

        self.image_sub = rospy.Subscriber('image', Image, self.image_cb)
        self.landmarks_pub = rospy.Publisher('landmarks', HandLandmarks, queue_size=10)
        self.landmarks_img_pub = rospy.Publisher('image_lms', Image, queue_size=10)

        self.landmarks_hdr = MsgHeaderHelper()
    
    def image_cb(self, data):
        try:
            img = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(f'cannot bridge input image: {e}')
            return
        
        img_mp = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
        stamp = int(data.header.stamp.to_sec() * 1000)
        if stamp <= self.last_img_stamp:
            rospy.logwarn(f'dropping received image (recv timestamp {stamp} <= last timestamp {self.last_img_stamp})')
            return
        self.last_img_stamp = stamp
        self.frame_id = data.header.frame_id
        self.landmarker.detect_async(img_mp, int(data.header.stamp.to_sec() * 1000))
    
    def landmarker_cb(self, results, img_out, timestamp):
        # rospy.loginfo(f'found {len(results.hand_landmarks)} hands')
        header = self.landmarks_hdr.header(self.frame_id)

        img_annotated = draw_landmarks_on_image(img_out.numpy_view(), results)
        try:
            img_annotated_msg = self.bridge.cv2_to_imgmsg(img_annotated, 'bgr8')
            img_annotated_msg.header = header
            self.landmarks_img_pub.publish(img_annotated_msg)
        except CvBridgeError as e:
            rospy.logerr(f'cannot bridge landmarks image: {e}')
        
        self.landmarks_pub.publish(HandLandmarks(
            header = header,
            landmarks = [
                HandLandmark(
                    right = handedness[0].category_name == 'Right',
                    lms = [
                        Point(lm.x, lm.y, lm.z)
                        for lm in lms
                    ],
                    world_lms = [
                        Point(lm.x, lm.y, lm.z)
                        for lm in world_lms
                    ]
                )
                for lms, world_lms, handedness in zip(results.hand_landmarks, results.hand_world_landmarks, results.handedness)
            ]
        ))
def main():
    rospy.init_node('mp_landmarker', anonymous=True)
    node = MPHandLandmarker()

    rospy.loginfo('spinning')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo('shutting down')

if __name__ == '__main__':
    main()