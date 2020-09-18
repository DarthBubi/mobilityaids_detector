import cv2
from caffe2.proto import caffe2_pb2
from caffe2.python import core
import detectron.utils.cython_nms as cython_nms
import numpy as np
import os
import sys

_CMAKE_INSTALL_PREFIX = '/usr/local'

def get_detectron_ops_lib():
    """Retrieve Detectron ops library."""
    # Candidate prefixes for the detectron ops lib path
    prefixes = [_CMAKE_INSTALL_PREFIX, sys.prefix, sys.exec_prefix] + sys.path
    # Search for detectron ops lib
    for prefix in prefixes:
        ops_path = os.path.join(prefix, 'lib/libcaffe2_detectron_ops_gpu.so')
        if os.path.exists(ops_path):
            # TODO(ilijar): Switch to using a logger
            print('Found Detectron ops lib: {}'.format(ops_path))
            break
    assert os.path.exists(ops_path), \
        ('Detectron ops lib not found; make sure that your Caffe2 '
         'version includes Detectron module')
    return ops_path

def get_device_option_cpu():
    device_option = core.DeviceOption(caffe2_pb2.CPU)
    return device_option


def get_device_option_cuda(gpu_id=0):
    device_option = caffe2_pb2.DeviceOption()
    device_option.device_type = caffe2_pb2.CUDA
    device_option.device_id = gpu_id
    return device_option

def _prepare_blobs(im, pixel_means, target_size, max_size):
    """ Reference: blob.prep_im_for_blob() """

    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape

    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])
    im_scale = float(target_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    im = cv2.resize(
        im, None, None, fx=im_scale, fy=im_scale, interpolation=cv2.INTER_LINEAR
    )

    blob = np.zeros([1, im.shape[0], im.shape[1], 3], dtype=np.float32)
    blob[0, :, :, :] = im
    channel_swap = (0, 3, 1, 2)  # swap channel to (k, c, h, w)
    blob = blob.transpose(channel_swap)

    blobs = {}
    blobs["data"] = blob
    blobs["im_info"] = np.array(
        [[blob.shape[2], blob.shape[3], im_scale]], dtype=np.float32
    )
    return blobs

def nms(dets, thresh):
    """Apply classic DPM-style greedy NMS."""
    if dets.shape[0] == 0:
        return []
    return cython_nms.nms(dets, thresh)

def get_depth_nms_predictions(pred_boxes, depths, cls_prob, num_classes):
    cls_depths = [[] for _ in range(num_classes)]
    cls_boxes = [[] for _ in range(num_classes)]
    for j in range(1, num_classes):
        inds = np.where(cls_prob[:, j] > 0.05)[0]
        scores_j = cls_prob[inds, j]
        boxes_j = pred_boxes[inds, j * 4:(j + 1) * 4]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(
            np.float32, copy=False
        )
        depths_j = depths[inds, j]
        keep = nms(dets_j, 0.5)
        nms_dets = dets_j[keep, :]
        nms_depths = depths_j[keep]
        cls_boxes[j] = nms_dets
        cls_depths[j] = nms_depths

    DETECTIONS_PER_IM = 100
    if DETECTIONS_PER_IM > 0:  # cfg.TEST.DETECTIONS_PER_IM
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, num_classes)]
        )
    if len(image_scores) > DETECTIONS_PER_IM:
        image_thresh = np.sort(image_scores)[-DETECTIONS_PER_IM]
        for j in range(1, num_classes):
            keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
            cls_boxes[j] = cls_boxes[j][keep, :]
            cls_depths[j] = cls_depths[j][keep]

    if cls_depths is not None:
        depth_list = [b for b in cls_depths if len(b) > 0]
        if len(depth_list) > 0:
            depths = np.concatenate(depth_list)
        else:
            depths = None
    
    return depths