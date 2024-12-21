#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import apriltag
from collections import namedtuple

from std_msgs.msg import Float32
from sensor_msgs.msg import Image
from hand_calibration.msg import Homography
from cv_bridge import CvBridge, CvBridgeError

# tag config
APRILTAG_FAMILY = 'tag36h11'
APRILTAG_SIZE = 0.036
TagConfig = namedtuple('TagConfig', ['cx', 'cy'])
APRILTAG_TAGS = {
    0: TagConfig(0.000,  0.140),
    1: TagConfig(0.220,  0.140),
    2: TagConfig(0.000,  0.000),
    3: TagConfig(0.220,  0.000),
}
OUTPUT_PIXELS_PER_METRE = 2000
OUTPUT_WIDTH = (max([tag.cx for tag in APRILTAG_TAGS.values()]) + APRILTAG_SIZE) * OUTPUT_PIXELS_PER_METRE
OUTPUT_HEIGHT = (max([tag.cy for tag in APRILTAG_TAGS.values()]) + APRILTAG_SIZE) * OUTPUT_PIXELS_PER_METRE

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

class PreprocessWorkspace:
    def __init__(self):
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber('image', Image, self.image_cb)
        self.tag_detector = apriltag.Detector(apriltag.DetectorOptions(families=APRILTAG_FAMILY))
    
        self.detections_pub = rospy.Publisher('tag_detections', Image, queue_size=10)
        self.corr_pub = rospy.Publisher('image_corr', Image, queue_size=10)

        self.homography_pub = rospy.Publisher('homography', Homography, queue_size=10)

        self.detections_hdr = MsgHeaderHelper() # header for detections images

        rospy.Publisher('pixels_per_metre', Float32, queue_size=1, latch=True).publish(OUTPUT_PIXELS_PER_METRE) # send out image resolution

    def image_cb(self, data):
        try:
            img = self.bridge.imgmsg_to_cv2(data, 'bgr8')
        except CvBridgeError as e:
            rospy.logerr(f'cannot bridge image: {e}')
            return
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        tags = self.tag_detector.detect(img_gray)
        if len(tags) == 0:
            rospy.logwarn('no tag detected')
            return
        
        img_detections = img.copy()

        corner_coords = dict(); centre_coords = dict() # coordinates of each tag's corners and centre (in pixels)
        for tag in tags:
            if tag.tag_id not in APRILTAG_TAGS:
                rospy.logwarn(f'found tag {tag.tag_id} not in config')
                continue

            (ptA, ptB, ptC, ptD) = tag.corners
            corner_coords[tag.tag_id] = (
                (int(ptA[0]), int(ptA[1])),
                (int(ptB[0]), int(ptB[1])),
                (int(ptC[0]), int(ptC[1])),
                (int(ptD[0]), int(ptD[1]))
            )
            centre_coords[tag.tag_id] = (int(tag.center[0]), int(tag.center[1]))

            # draw detections
            for i, point in enumerate(corner_coords[tag.tag_id]):
                cv2.circle(img_detections, point, 5, [(0,0,255),(0,255,0),(255,0,0),(0,255,255)][i], -1)
            cv2.circle(img_detections, centre_coords[tag.tag_id], 5, (255,255,0), -1)
        
        header = self.detections_hdr.header(data.header.frame_id)

        try:
            img_detections_msg = self.bridge.cv2_to_imgmsg(img_detections, 'bgr8')
            img_detections_msg.header = header
            self.detections_pub.publish(img_detections_msg)
        except CvBridgeError as e:
            rospy.logerr(f'cannot bridge detections image for publishing: {e}')

        # rospy.loginfo(f'{len(centre_coords)} tag(s) detected')

        # apply perspective transform
        HALF_SIZE = APRILTAG_SIZE / 2
        # if len(centre_coords) == 4: # all 4 tags picked up - use their centre points for transform
        #     src_pts = np.array([
        #         centre_coords[id] for id in centre_coords
        #     ], dtype=np.float32).reshape(-1, 2)
        #     dst_pts = (np.array([
        #         [APRILTAG_TAGS[id].cx, APRILTAG_TAGS[id].cy] for id in centre_coords
        #     ], dtype=np.float32).reshape(-1, 2) + HALF_SIZE) * OUTPUT_PIXELS_PER_METER
        # else: # pick random tag and use its corners for transform (less reliable)
        #     id = random.choice(list(corner_coords.keys()))
        #     print(f'using tag {id} for transform')
        #     src_pts = np.array(corner_coords[id], dtype=np.float32).reshape(-1, 2)
        #     dst_pts = (np.array([
        #         [APRILTAG_TAGS[id].cx - HALF_SIZE, APRILTAG_TAGS[id].cy + HALF_SIZE], # A
        #         [APRILTAG_TAGS[id].cx + HALF_SIZE, APRILTAG_TAGS[id].cy + HALF_SIZE], # B
        #         [APRILTAG_TAGS[id].cx + HALF_SIZE, APRILTAG_TAGS[id].cy - HALF_SIZE], # C
        #         [APRILTAG_TAGS[id].cx - HALF_SIZE, APRILTAG_TAGS[id].cy - HALF_SIZE], # D
        #     ], dtype=np.float32).reshape(-1, 2) + HALF_SIZE) * OUTPUT_PIXELS_PER_METER
        # M = cv2.getPerspectiveTransform(src_pts, dst_pts)
        src_pts = np.array([
            (list(corner_coords[id]) + [list(centre_coords[id])]) for id in centre_coords
        ], dtype=np.float32).reshape(-1, 2)
        dst_pts = (np.array([
            [
                [APRILTAG_TAGS[id].cx - HALF_SIZE, APRILTAG_TAGS[id].cy + HALF_SIZE], # A
                [APRILTAG_TAGS[id].cx + HALF_SIZE, APRILTAG_TAGS[id].cy + HALF_SIZE], # B
                [APRILTAG_TAGS[id].cx + HALF_SIZE, APRILTAG_TAGS[id].cy - HALF_SIZE], # C
                [APRILTAG_TAGS[id].cx - HALF_SIZE, APRILTAG_TAGS[id].cy - HALF_SIZE], # D
                [APRILTAG_TAGS[id].cx, APRILTAG_TAGS[id].cy] # centre
            ] for id in centre_coords
        ], dtype=np.float32).reshape(-1, 2) + HALF_SIZE) * OUTPUT_PIXELS_PER_METRE
        M, _ = cv2.findHomography(src_pts, dst_pts)

        self.homography_pub.publish(Homography(
            header = header,
            homography = M.flatten().tolist()
        ))

        img_corr = cv2.warpPerspective(img, M, (int(OUTPUT_WIDTH), int(OUTPUT_HEIGHT)))
        img_corr = cv2.flip(img_corr, 0) # flip image to account for coordinate frame difference
        try:
            img_corr_msg = self.bridge.cv2_to_imgmsg(img_corr, 'bgr8')
            header.frame_id += '_corr'; img_corr_msg.header = header
            self.corr_pub.publish(img_corr_msg)
        except CvBridgeError as e:
            rospy.logerr(f'cannot bridge corrected image for publishing: {e}')

def main():
    rospy.init_node('preprocess_workspace', anonymous=True)
    node = PreprocessWorkspace()

    rospy.loginfo('spinning')
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo('shutting down')

if __name__ == '__main__':
    main()
