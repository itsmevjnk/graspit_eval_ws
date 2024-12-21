#!/usr/bin/env python3

import rospy
import numpy as np
import cv2

import message_filters
from hand_landmarker.msg import HandLandmarks, HandLandmark
from hand_calibration.msg import Homography
from std_msgs.msg import Float32
from sensor_msgs.msg import CameraInfo

class LmPerspective:
    def __init__(self):
        self.resolution = None
        self.resolution_sub = rospy.Subscriber('pixels_per_metre', Float32, self.resolution_cb)

        self.lm_pub = rospy.Publisher('landmarks_corr', HandLandmarks, queue_size=10)

        self.sub = message_filters.ApproximateTimeSynchronizer([
            message_filters.Subscriber('landmarks', HandLandmarks),
            message_filters.Subscriber('homography', Homography),
            message_filters.Subscriber('camera_info', CameraInfo)
        ], 10, 0.25)
        self.sub.registerCallback(self.lm_cb)
    
    def resolution_cb(self, data):
        self.resolution = data.data
        rospy.loginfo(f'received image resolution: {self.resolution} pix/m')

    def lm_cb(self, landmarks, homography, cam_info):
        if self.resolution is None:
            rospy.logwarn('dropping message: no image resolution received yet')
            return
        
        M = np.array(homography.homography, dtype=np.float32).reshape(3, 3) # homography matrix
        for hand in landmarks.landmarks:
            pts_in = np.array([[ [lm.x * cam_info.width, lm.y * cam_info.height] for lm in hand.lms ]], dtype=np.float32)
            # print(pts_in.shape)
            pts_out = (cv2.perspectiveTransform(pts_in, M)[0] / self.resolution).tolist()

            for lm, pt_out in zip(hand.lms, pts_out):
                lm.x, lm.y = pt_out
            
            # hand_length = np.sqrt((hand.lms[0].x - hand.lms[12].x)**2 + (hand.lms[0].y - hand.lms[12].y)**2)
            # rospy.loginfo(f'hand length (assuming flat hand): {hand_length:.4f} m')
        
        self.lm_pub.publish(landmarks)
        
        
def main():
    rospy.init_node('lm_perspective', anonymous=True)
    node = LmPerspective()

    rospy.loginfo('spinning')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo('shutting down')

if __name__ == '__main__':
    main()