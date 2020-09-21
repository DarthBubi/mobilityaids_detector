#! /usr/bin/env python

import cv2
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace, models, dyndep
import numpy as np
import os
import sys
import net_helper

import multiclass_tracking
from multiclass_tracking.tracker import Tracker
from multiclass_tracking.viz import Visualizer
from image_handler import ImageHandler
from inside_box_filter import filter_inside_boxes
from publisher import Publisher

from dynamic_reconfigure.server import Server
from mobilityaids_detector.cfg import TrackingParamsConfig
from sensor_msgs.msg import CameraInfo, Image
from cv_bridge import CvBridge
import rospy
import tf


class DetectorNode(object):

    def __init__(self):
        self.classnames = ["background", "person", "crutches",
                           "walking_frame", "wheelchair", "push_wheelchair"]

        detectron_ops_lib = net_helper.get_detectron_ops_lib()
        dyndep.InitOpsLibrary(detectron_ops_lib)

        model_path = rospy.get_param("~model_path")
        self.fixed_frame = rospy.get_param('~fixed_frame', 'odom')
        self.tracking = rospy.get_param('~tracking', True)
        self.filter_detections = rospy.get_param('~filter_inside_boxes', True)
        self.inside_box_ratio = rospy.get_param('~inside_box_ratio', 0.8)
        camera_topic = rospy.get_param(
            '~camera_topic', '/camera/color/image_raw')
        camera_info_topic = rospy.get_param(
            '~camera_info_topic', '/camera/color/camera_info')

        self.net = caffe2_pb2.NetDef()
        with open(os.path.join(model_path, "model.pb"), "rb") as f:
            self.net.ParseFromString(f.read())

        self.init_net = caffe2_pb2.NetDef()
        with open(os.path.join(model_path, "model_init.pb"), "rb") as f:
            self.init_net.ParseFromString(f.read())

        workspace.ResetWorkspace()
        workspace.RunNetOnce(self.init_net)
        for op in self.net.op:
            for blob_in in op.input:
                if not workspace.HasBlob(blob_in):
                    workspace.CreateBlob(blob_in)
        workspace.CreateNet(self.net)

        # initialize subscribers
        rospy.Subscriber(camera_topic, Image,
                         self.image_callback, queue_size=1)
        rospy.Subscriber(camera_info_topic, CameraInfo,
                         self.cam_info_callback, queue_size=1)

        # image queues
        self.last_received_image = None  # set from image topic
        self.last_processed_image = None  # set from image topic
        self.new_image = False

        self.cam_calib = None  # set from camera info
        self.camera_frame = None  # set from camera info

        bridge = CvBridge()
        self.publisher = Publisher(self.classnames, bridge)
        observation_model = np.loadtxt(os.path.join(
            model_path, "observation_model.txt"), delimiter=',')
        ekf_sensor_noise = np.loadtxt(os.path.join(
            model_path, "meas_cov.txt"), delimiter=',')
        self.tracker = Tracker(
            ekf_sensor_noise, observation_model, use_hmm=True)
        self.tfl = tf.TransformListener()
        self.image_handler = ImageHandler(bridge, 540, 960)
        Server(TrackingParamsConfig, self.reconfigure_callback)
        thresholds = {}
        with open(os.path.join(model_path, "AP_thresholds.txt")) as f:
            for line in f:
                (key, val) = line.split(',')
                thresholds[key] = float(val)
        self.cla_thresholds = thresholds

    def reconfigure_callback(self, config, level):

        pos_cov_threshold = config["pos_cov_threshold"]
        mahalanobis_threshold = config["mahalanobis_max_dist"]
        euclidean_threshold = config["euclidean_max_dist"]

        accel_noise = config["accel_noise"]
        height_noise = config["height_noise"]
        init_vel_sigma = config["init_vel_sigma"]
        hmm_transition_prob = config["hmm_transition_prob"]

        use_hmm = config["use_hmm"]

        self.tracker.set_thresholds(pos_cov_threshold, mahalanobis_threshold,
                                    euclidean_threshold)

        self.tracker.set_tracking_config(accel_noise, height_noise,
                                         init_vel_sigma, hmm_transition_prob,
                                         use_hmm)

        return config

    def get_trafo_odom_in_cam(self):
        trafo_odom_in_cam = None

        if self.camera_frame is not None:

            try:
                time = self.last_processed_image.header.stamp
                self.tfl.waitForTransform(
                    self.camera_frame, self.fixed_frame, time, rospy.Duration(0.5))
                pos, quat = self.tfl.lookupTransform(
                    self.camera_frame, self.fixed_frame, time)

                trans = tf.transformations.translation_matrix(pos)
                rot = tf.transformations.quaternion_matrix(quat)

                trafo_odom_in_cam = np.dot(trans, rot)

            except Exception as e:
                rospy.logerr(e)

        else:
            rospy.logerr(
                "camera frame not set, cannot get trafo between camera and fixed frame")

        return trafo_odom_in_cam

    def run_model_pb(self, im):
        input_blobs = net_helper._prepare_blobs(
            im, [[[102.9801, 115.9465, 122.7717]]], 540, 960)

        gpu_blobs = ['data']

        for k, v in input_blobs.items():
            workspace.FeedBlob(
                core.ScopedName(k),
                v,
                net_helper.get_device_option_cuda() if k in gpu_blobs else
                net_helper.get_device_option_cpu()
            )

        try:
            workspace.RunNet(self.net.name)
            scores = workspace.FetchBlob("score_nms")
            cls_prob = workspace.FetchBlob(
                core.ScopedName('cls_prob')).squeeze()
            classids = workspace.FetchBlob("class_nms")
            boxes = workspace.FetchBlob("bbox_nms")
            depths = workspace.FetchBlob("depth_pred").squeeze()
            pred_boxes = workspace.FetchBlob("pred_bbox").squeeze()

            # Get depth predictions per class
            num_classes = len(self.classnames)
            depths = net_helper.get_depth_nms_predictions(
                pred_boxes, depths, cls_prob, num_classes)

        except Exception as e:
            print("Running pb model failed.\n{}".format(e))
            # may not detect anything at all
            R = 0
            scores = np.zeros((R,), dtype=np.float32)
            boxes = np.zeros((R, 4), dtype=np.float32)
            classids = np.zeros((R,), dtype=np.float32)
            depths = np.zeros((R,), dtype=np.float32)

        boxes = np.column_stack((boxes, scores))
        detections = []

        for i in range(len(classids)):
            detection = {}

            detection["bbox"] = list(map(int, boxes[i, :4]))
            detection["score"] = boxes[i, -1]
            detection["depth"] = depths[i]
            detection["category_id"] = int(classids[i])

            if detection["score"] > self.cla_thresholds[self.classnames[detection["category_id"]]]:
                detections.append(detection)

        if self.filter_detections:
            filter_inside_boxes(
                detections, inside_ratio_thresh=self.inside_box_ratio)

        return detections

    def update_tracker(self, detections, trafo_odom_in_cam, dt):
        if dt is not None:
            self.tracker.predict(dt)

        if (trafo_odom_in_cam is not None) and (self.cam_calib is not None):
            self.tracker.update(detections, trafo_odom_in_cam, self.cam_calib)

    def process_last_image(self):
        if self.new_image:

            dt = None
            if self.last_processed_image is not None:
                dt = (self.last_received_image.header.stamp -
                      self.last_processed_image.header.stamp).to_sec()
            self.last_processed_image = self.last_received_image

            image = self.image_handler.get_image(self.last_processed_image)

            detections = self.run_model_pb(image)

            trafo_odom_in_cam = self.get_trafo_odom_in_cam()

            if self.tracking:
                self.update_tracker(detections, trafo_odom_in_cam, dt)

            # publish messages
            self.publisher.publish_results(image, self.last_processed_image.header,
                                           detections, self.tracker, self.cam_calib,
                                           trafo_odom_in_cam, self.fixed_frame,
                                           tracking=self.tracking)
            self.new_image = False

    def get_cam_calib(self, camera_info):
        cam_calib = {}

        # camera calibration
        cam_calib["fx"] = camera_info.K[0]
        cam_calib["cx"] = camera_info.K[2]
        cam_calib["fy"] = camera_info.K[4]
        cam_calib["cy"] = camera_info.K[5]

        return cam_calib

    def cam_info_callback(self, camera_info):
        if self.cam_calib is None:
            rospy.loginfo("camera info received")
            self.cam_calib = self.get_cam_calib(camera_info)
            self.camera_frame = camera_info.header.frame_id

    def image_callback(self, image):
        self.last_received_image = image
        self.new_image = True


if __name__ == "__main__":
    rospy.init_node('mobilityaids_detector')
    detector_node = DetectorNode()
    rate = rospy.Rate(30)
    rospy.loginfo("Waiting for images...")

    while not rospy.is_shutdown():
        detector_node.process_last_image()
        rate.sleep()
