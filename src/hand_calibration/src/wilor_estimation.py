#!/usr/bin/env python3

import rospy

import cv2
import torch
import numpy as np
from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline

import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from hand_calibration.msg import HandShape
from hand_landmarker.msg import HandLandmarks

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

class WiLorEstimation:
    def __init__(self):
        self.bridge = CvBridge()

        rospy.loginfo(f'CUDA availability: {torch.cuda.is_available()}')
        self.torch_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.pipeline = WiLorHandPose3dEstimationPipeline(device=self.torch_dev, dtype=torch.float32)

        self.pub_header = MsgHeaderHelper()
        self.pub = rospy.Publisher('shape', HandShape, queue_size=10)

        self.sub = message_filters.ApproximateTimeSynchronizer([
            message_filters.Subscriber('landmarks', HandLandmarks),
            message_filters.Subscriber('image', Image)
        ], 1, 0.25)
        self.sub.registerCallback(self.image_cb)
    
    def calc_length_scale(self, mp_a, mp_b, wilor_a, wilor_b):
        mp_len = np.sqrt((mp_a.x - mp_b.x)**2 + (mp_a.y - mp_b.y)**2)
        wilor_len = np.linalg.norm(wilor_a - wilor_b)
        rospy.loginfo(f'keypoint distance: MediaPipe {mp_len:.4f} m, WiLor {wilor_len:.4f} au')
        return mp_len / wilor_len

    def image_cb(self, landmarks, image):
        if len(landmarks.landmarks) == 0:
            # rospy.logwarn('no landmarks received - possibly no hands in view')
            return
        if len(landmarks.landmarks) > 1:
            rospy.logwarn('more than one hand detected - cannot estimate shape')
            return
        
        try:
            img = self.bridge.imgmsg_to_cv2(image, 'rgb8')
        except CvBridgeError as e:
            rospy.logerr(f'cannot bridge input image: {e}')
            return
        
        wilor_out = self.pipeline.predict(img)[0] # we only have one hand here

        wilor_right = wilor_out['is_right'] > 0.5 # TODO: is this really a probability?
        if wilor_right != landmarks.landmarks[0].right:
            rospy.logwarn(f"discarding detection: handedness discrepancy (WiLor: {wilor_right}, MediaPipe: {landmarks.landmarks[0].right})")
            return
        
        mano_betas = wilor_out['wilor_preds']['betas'].flatten()
        mano_joints = wilor_out['wilor_preds']['pred_keypoints_3d'].reshape(-1, 3)

        mp_joints = landmarks.landmarks[0].lms
        scale = np.mean([
            self.calc_length_scale(mp_joints[0], mp_joints[5], mano_joints[0], mano_joints[5]),
            self.calc_length_scale(mp_joints[0], mp_joints[17], mano_joints[0], mano_joints[17]),
        ]).item()
        rospy.loginfo(f'average model scale factor: {scale:.4f} m/au')

        self.pub.publish(HandShape(
            header = self.pub_header.header(image.header.frame_id),
            right = wilor_right,
            betas = mano_betas.tolist(),
            scale = scale
        ))

def main():
    rospy.init_node('wilor_estimation', anonymous=True)
    node = WiLorEstimation()

    rospy.loginfo('spinning')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo('shutting down')

if __name__ == '__main__':
    main()
