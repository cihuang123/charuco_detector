#!/usr/bin/env python
import rospy
import math
import numpy as np
import time
import tf
import tf.transformations as tr
from scipy.stats import gmean
from std_msgs.msg import Bool, Float64, Empty
from sensor_msgs.msg import Joy
from geometry_msgs.msg import PoseStamped, Pose, TransformStamped, Transform
from std_srvs.srv import Trigger, TriggerResponse
from sensor_msgs.msg import JointState

def pose_to_pq(msg):
    """Convert a C{geometry_msgs/Pose} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.position.x, msg.position.y, msg.position.z])
    q = np.array([msg.orientation.x, msg.orientation.y,
                  msg.orientation.z, msg.orientation.w])
    return p, q


def pose_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/PoseStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return pose_to_pq(msg.pose)


def transform_to_pq(msg):
    """Convert a C{geometry_msgs/Transform} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
    q = np.array([msg.rotation.x, msg.rotation.y,
                  msg.rotation.z, msg.rotation.w])
    return p, q


def transform_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/TransformStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return transform_to_pq(msg.transform)


def msg_to_se3(msg):
    """Conversion from geometric ROS messages into SE(3)

    @param msg: Message to transform. Acceptable types - C{geometry_msgs/Pose}, C{geometry_msgs/PoseStamped},
    C{geometry_msgs/Transform}, or C{geometry_msgs/TransformStamped}
    @return: a 4x4 SE(3) matrix as a numpy array
    @note: Throws TypeError if we receive an incorrect type.
    """
    if isinstance(msg, Pose):
        p, q = pose_to_pq(msg)
    elif isinstance(msg, PoseStamped):
        p, q = pose_stamped_to_pq(msg)
    elif isinstance(msg, Transform):
        p, q = transform_to_pq(msg)
    elif isinstance(msg, TransformStamped):
        p, q = transform_stamped_to_pq(msg)
    else:
        raise TypeError("Invalid type for conversion to SE(3)")
    norm = np.linalg.norm(q)
    if np.abs(norm - 1.0) > 1e-3:
        raise ValueError(
            "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                str(q), np.linalg.norm(q)))
    elif np.abs(norm - 1.0) > 1e-6:
        q = q / norm
    g = tr.quaternion_matrix(q)
    g[0:3, -1] = p
    return g

class get_cam_pose():

    def __init__(self):
        self.top_camera_tag_1 = tf.TransformListener()
        self.left_camera_tag_1 = tf.TransformListener()

        start = rospy.Service("/record_left_cam_pose", Trigger, self.record_pose)

        get_avg = rospy.Service("/get_left_cam_pose", Trigger, self.get_pose)

        self.lock = True
        self.top_cam_tag_1_pose = Pose()
        self.left_cam_tag_1_pose = Pose()
        self.pos_x = []
        self.pos_y = []
        self.pos_z = []

        self.rot_x = []
        self.rot_y = []
        self.rot_z = []


    def record_pose(self, req):
        res = TriggerResponse()
        try:
            while self.lock == True:
                try:
                    (top_cam_tag_1_trans, top_cam_tag_1_rot) = self.top_camera_tag_1.lookupTransform('/map', '/charuco', rospy.Time(0))
                    (left_cam_tag_1_trans, left_cam_tag_1_rot) = self.left_camera_tag_1.lookupTransform('/charuco_left','/camera_left_color_optical_frame', rospy.Time(0))
                    self.lock = False
                except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException):
                    self.lock = True
                    continue

            # ************************** top_cam_tag_1 **************************
            self.top_cam_tag_1_pose.position.x = top_cam_tag_1_trans[0]
            self.top_cam_tag_1_pose.position.y = top_cam_tag_1_trans[1]
            self.top_cam_tag_1_pose.position.z = top_cam_tag_1_trans[2]
            
            self.top_cam_tag_1_pose.orientation.x = top_cam_tag_1_rot[0]
            self.top_cam_tag_1_pose.orientation.y = top_cam_tag_1_rot[1]
            self.top_cam_tag_1_pose.orientation.z = top_cam_tag_1_rot[2]
            self.top_cam_tag_1_pose.orientation.w = top_cam_tag_1_rot[3]
            top_cam_tag_1_matrix = msg_to_se3(self.top_cam_tag_1_pose)

            # ************************** left_cam_tag_1 **************************
            self.left_cam_tag_1_pose.position.x = left_cam_tag_1_trans[0]
            self.left_cam_tag_1_pose.position.y = left_cam_tag_1_trans[1]
            self.left_cam_tag_1_pose.position.z = left_cam_tag_1_trans[2]
            
            self.left_cam_tag_1_pose.orientation.x = left_cam_tag_1_rot[0]
            self.left_cam_tag_1_pose.orientation.y = left_cam_tag_1_rot[1]
            self.left_cam_tag_1_pose.orientation.z = left_cam_tag_1_rot[2]
            self.left_cam_tag_1_pose.orientation.w = left_cam_tag_1_rot[3]
            left_cam_tag_1_matrix = msg_to_se3(self.left_cam_tag_1_pose)

            map_left_cam_1_matrix = np.matmul(top_cam_tag_1_matrix, left_cam_tag_1_matrix)

            # print("X = ", (map_left_cam_1_matrix[0][3] + map_left_cam_2_matrix[0][3])/2)
            # print("Y = ", (map_left_cam_1_matrix[1][3] + map_left_cam_2_matrix[1][3])/2)
            # print("Z = ", (map_left_cam_1_matrix[2][3] + map_left_cam_2_matrix[2][3])/2)

            self.pos_x.append(map_left_cam_1_matrix[0][3])
            self.pos_y.append(map_left_cam_1_matrix[1][3])
            self.pos_z.append(map_left_cam_1_matrix[2][3])
            # print(map_left_cam_1_matrix)
            # print(map_left_cam_2_matrix)
            al, be, ga = tr.euler_from_matrix(map_left_cam_1_matrix, 'szyx')
            # print("z,y,x = ", (al/math.pi)*180, (be/math.pi)*180, (ga/math.pi)*180)
            if (al <0):
                al = -al
            if (be <0):
                be = -be
            if (ga <0):
                ga = -ga

            self.rot_x.append((ga/math.pi)*180)
            self.rot_y.append((be/math.pi)*180)
            self.rot_z.append((al/math.pi)*180)
            # print("Raw Rotation (z,y,x) = ", self.rot_z, self.rot_y, self.rot_x)

            self.lock = True

            res.success = True
        except (rospy.ServiceException, rospy.ROSException) as e:
            res.success = False
            print("Service call failed: %s"%e)

        return res
    
    def get_pose(self, req):
        res = TriggerResponse()
        if (len(self.pos_x) > 4):
            res.success = True
            print("Position (X,Y,Z) = ", -gmean(self.pos_y), gmean(self.pos_z), gmean(self.pos_x))
            print("Rotation (x,y,z) = ", gmean(self.rot_x), gmean(self.rot_z) - 90, gmean(self.rot_y))

        else:
            res.success = False

        return res

if __name__ == '__main__':
    rospy.init_node('get_left_camera')
    test = get_cam_pose()
    rospy.spin()
