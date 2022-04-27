import datetime
# import pytz
import threading
import numpy as np
import pandas as pd

from Logging import log_print
from File import save_pickle, load_pickle

from CommunicationHL2 import Communication, CommunicationLUTCameraProjection

# UTC_SHIFT = 2 # in hours
# UTC_SHIFT = 1 # in hours

HOLOLENS_METER_TO_MILIMETER_SCALE = 1000
# HOLOLENS_METER_TO_MILIMETER_SCALE = 1

QR_CODE_NAMES = ["qr_code_front", "qr_code_left", "qr_code_right", "qr_code_top"]
RESEARCH_MODE_CAMERA_NAMES = ["vl_front_left_cam", "vl_front_right_cam", "ahat_depth_cam", "lt_depth_cam"]
CAMERA_NAMES = ["pv_cam"] + RESEARCH_MODE_CAMERA_NAMES
ACQUISITIONS = ["optical", "qr_code_position"] + CAMERA_NAMES
ACQUISITIONS_HOLOLENS = ["qr_code_position"] + CAMERA_NAMES

MAX_AHAT_DEPTH = 1200 # only for visualization purpose
MAX_AHAT_AB = 1200 # only for visualization purpose

MAX_LT_DEPTH = 1200 # only for visualization purpose
#MAX_LT_AB = 3000 # only for visualization purpose
MAX_LT_AB = 11000 # only for visualization purpose

LUT_PROJECTION_X = 0
LUT_PROJECTION_Y = 1
LUT_PROJECTION_U = 2
LUT_PROJECTION_V = 3
LUT_PROJECTION_MIN_X = 4
LUT_PROJECTION_MAX_X = 5
LUT_PROJECTION_MIN_Y = 6
LUT_PROJECTION_MAX_Y = 7

class DataAcquisition:
    VERSION = 2

    def __init__(self):
        self.acquisitions = {}
        self.acquisitions ["version"] = DataAcquisition.VERSION

        self.acquisitions["magnetic_seed"] = pd.DataFrame([], columns = ['time', 'distance', 'angle', 'azimut', 'x', 'y', 'z'])
        self.acquisitions["magnetic_seed"] = self.acquisitions["magnetic_seed"].set_index('time')

        cols = ['time']
        cols += ['frame 1', 'unknown 1', 'status 1', 'qw 1', 'qx 1', 'qy 1', 'qz 1', 'tx 1', 'ty 1', 'tz 1', 'err 1']
        cols += ['nb markers 1']
        cols += ['status 1.1', 'tx 1.1', 'ty 1.1', 'tz 1.1']
        cols += ['status 1.2', 'tx 1.2', 'ty 1.2', 'tz 1.2']
        cols += ['status 1.3', 'tx 1.3', 'ty 1.3', 'tz 1.3']
        cols += ['status 1.4', 'tx 1.4', 'ty 1.4', 'tz 1.4']
        cols += ['frame 2', 'unknown 2', 'status 2', 'qw 2', 'qx 2', 'qy 2', 'qz 2', 'tx 2', 'ty 2', 'tz 2', 'err 2']
        cols += ['nb markers 2']
        cols += ['status 2.1', 'tx 2.1', 'ty 2.1', 'tz 2.1']
        cols += ['status 2.2', 'tx 2.2', 'ty 2.2', 'tz 2.2']
        cols += ['status 2.3', 'tx 2.3', 'ty 2.3', 'tz 2.3']
        cols += ['status 2.4', 'tx 2.4', 'ty 2.4', 'tz 2.4']
        cols += ['frame 3', 'unknown 3', 'status 3', 'qw 3', 'qx 3', 'qy 3', 'qz 3', 'tx 3', 'ty 3', 'tz 3', 'err 3']
        cols += ['nb markers 3']
        cols += ['status 3.1', 'tx 3.1', 'ty 3.1', 'tz 3.1']
        cols += ['status 3.2', 'tx 3.2', 'ty 3.2', 'tz 3.2']
        cols += ['status 3.3', 'tx 3.3', 'ty 3.3', 'tz 3.3']
        cols += ['status 3.4', 'tx 3.4', 'ty 3.4', 'tz 3.4']
        self.acquisitions["optical"] = pd.DataFrame([], columns = cols)
        self.acquisitions["optical"] = self.acquisitions["optical"].set_index('time')

        self.acquisitions["qr_code_position"] = pd.DataFrame([], columns = ['time'
            , 'q1_m11', 'q1_m12', 'q1_m13', 'q1_m14'
            , 'q1_m21', 'q1_m22', 'q1_m23', 'q1_m24'
            , 'q1_m31', 'q1_m32', 'q1_m33', 'q1_m34'
            , 'q1_m41', 'q1_m42', 'q1_m43', 'q1_m44'

            , 'q2_m11', 'q2_m12', 'q2_m13', 'q2_m14'
            , 'q2_m21', 'q2_m22', 'q2_m23', 'q2_m24'
            , 'q2_m31', 'q2_m32', 'q2_m33', 'q2_m34'
            , 'q2_m41', 'q2_m42', 'q2_m43', 'q2_m44'

            , 'q3_m11', 'q3_m12', 'q3_m13', 'q3_m14'
            , 'q3_m21', 'q3_m22', 'q3_m23', 'q3_m24'
            , 'q3_m31', 'q3_m32', 'q3_m33', 'q3_m34'
            , 'q3_m41', 'q3_m42', 'q3_m43', 'q3_m44'

            , 'q4_m11', 'q4_m12', 'q4_m13', 'q4_m14'
            , 'q4_m21', 'q4_m22', 'q4_m23', 'q4_m24'
            , 'q4_m31', 'q4_m32', 'q4_m33', 'q4_m34'
            , 'q4_m41', 'q4_m42', 'q4_m43', 'q4_m44'])
        self.acquisitions["qr_code_position"] = self.acquisitions["qr_code_position"].set_index('time')

        columns = ['time'
            , 'cam_to_w_translation_x', 'cam_to_w_translation_y', 'cam_to_w_translation_z'
            , 'cam_to_w_rotation_x', 'cam_to_w_rotation_y', 'cam_to_w_rotation_z', 'cam_to_w_rotation_w'
            , 'focal_length_x', 'focal_length_y', 'center_coordinate_x', 'center_coordinate_y'
            , 'radial_distortion_x', 'radial_distortion_y', 'tangential_distortion_x', 'tangential_distortion_y', 'radial_distortion_z'
            , 'm11', 'm12', 'm13', 'm14'
            , 'm21', 'm22', 'm23', 'm24'
            , 'm31', 'm32', 'm33', 'm34'
            , 'm41', 'm42', 'm43', 'm44'
            , 'downsampling_factor']
        self.acquisitions["pv_cam"] = pd.DataFrame([], columns = columns)
        self.acquisitions["pv_cam"] = self.acquisitions["pv_cam"].set_index('time')
        self.acquisitions["pv_cam_frames"] = []

        self.acquisitions["vl_front_left_cam"] = pd.DataFrame([], columns = columns)
        self.acquisitions["vl_front_left_cam"] = self.acquisitions["vl_front_left_cam"].set_index('time')
        self.acquisitions["vl_front_left_cam_frames"] = []
        self.acquisitions["vl_front_left_cam_lut_projection"] = [None, None, None, None, 0, 0, 0, 0]

        self.acquisitions["vl_front_right_cam"] = pd.DataFrame([], columns = columns)
        self.acquisitions["vl_front_right_cam"] = self.acquisitions["vl_front_right_cam"].set_index('time')
        self.acquisitions["vl_front_right_cam_frames"] = []
        self.acquisitions["vl_front_right_cam_lut_projection"] = [None, None, None, None, 0, 0, 0, 0]

        self.acquisitions["ahat_depth_cam"] = pd.DataFrame([], columns = columns)
        self.acquisitions["ahat_depth_cam"] = self.acquisitions["ahat_depth_cam"].set_index('time')
        self.acquisitions["ahat_depth_cam_frames"] = []
        self.acquisitions["ahat_depth_cam_ab_frames"] = []
        self.acquisitions["ahat_depth_cam_lut_projection"] = [None, None, None, None, 0, 0, 0, 0]

        self.acquisitions["lt_depth_cam"] = pd.DataFrame([], columns = columns)
        self.acquisitions["lt_depth_cam"] = self.acquisitions["lt_depth_cam"].set_index('time')
        self.acquisitions["lt_depth_cam_frames"] = []
        self.acquisitions["lt_depth_cam_ab_frames"] = []
        self.acquisitions["lt_depth_cam_lut_projection"] = [None, None, None, None, 0, 0, 0, 0]

        self.recording = False

        self.general_mutex = threading.Lock()

    def start_recording(self):
        self.recording = True

    def add_communication(self, communication):
        if self.recording or communication.mandatory_to_receive:
            self.general_mutex.acquire()

            if communication.action == Communication.ACTION_SEND_LUT_CAMERA_PROJECTION:
                log_print("Communication.ACTION_SEND_LUT_CAMERA_PROJECTION")
                self.acquisitions["vl_front_left_cam_lut_projection"] = [
                    communication.lut_camera_projection_x[CommunicationLUTCameraProjection.RM_SENSOR_VL_FRONT_LEFT]
                    , communication.lut_camera_projection_y[CommunicationLUTCameraProjection.RM_SENSOR_VL_FRONT_LEFT]
                    , communication.lut_camera_projection_u[CommunicationLUTCameraProjection.RM_SENSOR_VL_FRONT_LEFT]
                    , communication.lut_camera_projection_v[CommunicationLUTCameraProjection.RM_SENSOR_VL_FRONT_LEFT]
                    , communication.camera_space_min_x[CommunicationLUTCameraProjection.RM_SENSOR_VL_FRONT_LEFT]
                    , communication.camera_space_max_x[CommunicationLUTCameraProjection.RM_SENSOR_VL_FRONT_LEFT]
                    , communication.camera_space_min_y[CommunicationLUTCameraProjection.RM_SENSOR_VL_FRONT_LEFT]
                    , communication.camera_space_max_y[CommunicationLUTCameraProjection.RM_SENSOR_VL_FRONT_LEFT]]
                self.acquisitions["vl_front_right_cam_lut_projection"] = [
                    communication.lut_camera_projection_x[CommunicationLUTCameraProjection.RM_SENSOR_VL_FRONT_RIGHT]
                    , communication.lut_camera_projection_y[CommunicationLUTCameraProjection.RM_SENSOR_VL_FRONT_RIGHT]
                    , communication.lut_camera_projection_u[CommunicationLUTCameraProjection.RM_SENSOR_VL_FRONT_RIGHT]
                    , communication.lut_camera_projection_v[CommunicationLUTCameraProjection.RM_SENSOR_VL_FRONT_RIGHT]
                    , communication.camera_space_min_x[CommunicationLUTCameraProjection.RM_SENSOR_VL_FRONT_RIGHT]
                    , communication.camera_space_max_x[CommunicationLUTCameraProjection.RM_SENSOR_VL_FRONT_RIGHT]
                    , communication.camera_space_min_y[CommunicationLUTCameraProjection.RM_SENSOR_VL_FRONT_RIGHT]
                    , communication.camera_space_max_y[CommunicationLUTCameraProjection.RM_SENSOR_VL_FRONT_RIGHT]]
                self.acquisitions["ahat_depth_cam_lut_projection"] = [
                    communication.lut_camera_projection_x[CommunicationLUTCameraProjection.RM_SENSOR_AHAT_DEPTH]
                    , communication.lut_camera_projection_y[CommunicationLUTCameraProjection.RM_SENSOR_AHAT_DEPTH]
                    , communication.lut_camera_projection_u[CommunicationLUTCameraProjection.RM_SENSOR_AHAT_DEPTH]
                    , communication.lut_camera_projection_v[CommunicationLUTCameraProjection.RM_SENSOR_AHAT_DEPTH]
                    , communication.camera_space_min_x[CommunicationLUTCameraProjection.RM_SENSOR_AHAT_DEPTH]
                    , communication.camera_space_max_x[CommunicationLUTCameraProjection.RM_SENSOR_AHAT_DEPTH]
                    , communication.camera_space_min_y[CommunicationLUTCameraProjection.RM_SENSOR_AHAT_DEPTH]
                    , communication.camera_space_max_y[CommunicationLUTCameraProjection.RM_SENSOR_AHAT_DEPTH]]
                self.acquisitions["lt_depth_cam_lut_projection"] = [
                    communication.lut_camera_projection_x[CommunicationLUTCameraProjection.RM_SENSOR_LT_DEPTH]
                    , communication.lut_camera_projection_y[CommunicationLUTCameraProjection.RM_SENSOR_LT_DEPTH]
                    , communication.lut_camera_projection_u[CommunicationLUTCameraProjection.RM_SENSOR_LT_DEPTH]
                    , communication.lut_camera_projection_v[CommunicationLUTCameraProjection.RM_SENSOR_LT_DEPTH]
                    , communication.camera_space_min_x[CommunicationLUTCameraProjection.RM_SENSOR_LT_DEPTH]
                    , communication.camera_space_max_x[CommunicationLUTCameraProjection.RM_SENSOR_LT_DEPTH]
                    , communication.camera_space_min_y[CommunicationLUTCameraProjection.RM_SENSOR_LT_DEPTH]
                    , communication.camera_space_max_y[CommunicationLUTCameraProjection.RM_SENSOR_LT_DEPTH]]

            else:
                if communication.action == Communication.ACTION_SEND_IMAGE:
                    df = self.acquisitions["pv_cam"]
                    frames = self.acquisitions["pv_cam_frames"]
                elif communication.action == Communication.ACTION_SEND_FRONT_LEFT_IMAGE:
                    df = self.acquisitions["vl_front_left_cam"]
                    frames = self.acquisitions["vl_front_left_cam_frames"]
                elif communication.action == Communication.ACTION_SEND_FRONT_RIGHT_IMAGE:
                    df = self.acquisitions["vl_front_right_cam"]
                    frames = self.acquisitions["vl_front_right_cam_frames"]
                elif communication.action == Communication.ACTION_SEND_AHAT_DEPTH_IMAGE:
                    df = self.acquisitions["ahat_depth_cam"]
                    frames = self.acquisitions["ahat_depth_cam_frames"]
                    ab_frames = self.acquisitions["ahat_depth_cam_ab_frames"]
                elif communication.action == Communication.ACTION_SEND_LT_DEPTH_IMAGE:
                    df = self.acquisitions["lt_depth_cam"]
                    frames = self.acquisitions["lt_depth_cam_frames"]
                    ab_frames = self.acquisitions["lt_depth_cam_ab_frames"]
                elif communication.action == Communication.ACTION_SEND_QR_CODE_POSITION:
                    df = self.acquisitions["qr_code_position"]

                # communication.timestamp = communication.timestamp - datetime.timedelta(hours=UTC_SHIFT)

                # if communication.timestamp in df.index:
                if False:
                    log_print(f"error add_communication {communication.timestamp} already in dataframe")
                else:
                    if communication.action == Communication.ACTION_SEND_QR_CODE_POSITION:
                        qr_codes_params = [[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.]]
                        if communication.qr_code_ids is not None:
                            for i in range(len(communication.qr_code_ids)):
                                communication.qr_code_transforms[i][3] = communication.qr_code_transforms[i][3]*HOLOLENS_METER_TO_MILIMETER_SCALE
                                communication.qr_code_transforms[i][7] = communication.qr_code_transforms[i][7]*HOLOLENS_METER_TO_MILIMETER_SCALE
                                communication.qr_code_transforms[i][11] = communication.qr_code_transforms[i][11]*HOLOLENS_METER_TO_MILIMETER_SCALE

                                qr_codes_params[communication.qr_code_ids[i]] = communication.qr_code_transforms[i]
                        df.loc[communication.timestamp] = tuple(qr_codes_params[0] + qr_codes_params[1] + qr_codes_params[2] + qr_codes_params[3])

                    else:
                        df.loc[communication.timestamp] = tuple([communication.translation_rig_to_world[0]*HOLOLENS_METER_TO_MILIMETER_SCALE, communication.translation_rig_to_world[1]*HOLOLENS_METER_TO_MILIMETER_SCALE, communication.translation_rig_to_world[2]*HOLOLENS_METER_TO_MILIMETER_SCALE
                , communication.rotation_rig_to_world[0], communication.rotation_rig_to_world[1], communication.rotation_rig_to_world[2], communication.rotation_rig_to_world[3]
                , communication.focal_length_x, communication.focal_length_y, communication.center_coordinate_x, communication.center_coordinate_y
                , communication.distortion_coefficients[0], communication.distortion_coefficients[1], communication.distortion_coefficients[2], communication.distortion_coefficients[3], communication.distortion_coefficients[4]
                , communication.rig_to_camera[0,0], communication.rig_to_camera[0,1], communication.rig_to_camera[0,2], communication.rig_to_camera[0,3]*HOLOLENS_METER_TO_MILIMETER_SCALE
                , communication.rig_to_camera[1,0], communication.rig_to_camera[1,1], communication.rig_to_camera[1,2], communication.rig_to_camera[1,3]*HOLOLENS_METER_TO_MILIMETER_SCALE
                , communication.rig_to_camera[2,0], communication.rig_to_camera[2,1], communication.rig_to_camera[2,2], communication.rig_to_camera[2,3]*HOLOLENS_METER_TO_MILIMETER_SCALE
                , communication.rig_to_camera[3,0], communication.rig_to_camera[3,1], communication.rig_to_camera[3,2], communication.rig_to_camera[3,3]
                , communication.downsampling_factor
                ])

                        if communication.action in [Communication.ACTION_SEND_IMAGE, Communication.ACTION_SEND_FRONT_LEFT_IMAGE, Communication.ACTION_SEND_FRONT_RIGHT_IMAGE]:
                            frames.append(communication.image)
                        elif communication.action in [Communication.ACTION_SEND_AHAT_DEPTH_IMAGE, Communication.ACTION_SEND_LT_DEPTH_IMAGE]:
                            if communication.depth_buffer is not None:
                                frames.append(communication.depth_buffer)
                            if communication.active_brightness_buffer is not None:
                                ab_frames.append(communication.active_brightness_buffer)

            self.general_mutex.release()

    def save_data(self, filename):
        self.general_mutex.acquire()
        save_pickle(self.acquisitions, filename)
        self.general_mutex.release()

    def load_data(self, filename, prepare_data=True, check_version=True):
        self.general_mutex.acquire()

        self.acquisitions = load_pickle(filename)

        if check_version and ("version" not in self.acquisitions or self.acquisitions["version"] != DataAcquisition.VERSION):
            print(f"version {self.acquisitions['version'] if 'version' in self.acquisitions else '0'} of the file is old, update the file first")
            assert False

        # debug, remove some acquisition
        # self.acquisitions['magnetic_seed'] = self.acquisitions['magnetic_seed'][0:0]
        # self.acquisitions['optical'] = self.acquisitions['optical'][0:0]
        # self.acquisitions['qr_code_position'] = self.acquisitions['qr_code_position'][0:0]
        # self.acquisitions['pv_cam'] = self.acquisitions['pv_cam'][0:0]
        # self.acquisitions['vl_front_left_cam'] = self.acquisitions['vl_front_left_cam'][0:0]
        # self.acquisitions['vl_front_right_cam'] = self.acquisitions['vl_front_right_cam'][0:0]
        # self.acquisitions['ahat_depth_cam'] = self.acquisitions['ahat_depth_cam'][0:0]
        # self.acquisitions['lt_depth_cam'] = self.acquisitions['lt_depth_cam'][0:0]

        print(f"self.acquisitions['magnetic_seed'].index.size {self.acquisitions['magnetic_seed'].index.size}")
        print(f"self.acquisitions['optical'].index.size {self.acquisitions['optical'].index.size}")
        print(f"self.acquisitions['qr_code_position'].index.size {self.acquisitions['qr_code_position'].index.size}")

        print(f"self.acquisitions['pv_cam'].index.size {self.acquisitions['pv_cam'].index.size}")
        print(f"len(self.acquisitions['pv_cam_frames']) {len(self.acquisitions['pv_cam_frames'])}")
        if len(self.acquisitions['pv_cam_frames']) != 0:
            print(f"self.acquisitions['pv_cam_frames'][0].shape {self.acquisitions['pv_cam_frames'][0].shape}")

        print(f"self.acquisitions['vl_front_left_cam'].index.size {self.acquisitions['vl_front_left_cam'].index.size}")
        print(f"len(self.acquisitions['vl_front_left_cam_frames']) {len(self.acquisitions['vl_front_left_cam_frames'])}")
        if len(self.acquisitions['vl_front_left_cam_frames']) != 0:
            print(f"self.acquisitions['vl_front_left_cam_frames'][0].shape {self.acquisitions['vl_front_left_cam_frames'][0].shape}")
        if self.acquisitions['vl_front_left_cam_lut_projection'][LUT_PROJECTION_X] is not None:
            print(f"self.acquisitions['vl_front_left_cam_lut_projection'][LUT_PROJECTION_X].shape {self.acquisitions['vl_front_left_cam_lut_projection'][LUT_PROJECTION_X].shape}")

        print(f"self.acquisitions['vl_front_right_cam'].index.size {self.acquisitions['vl_front_right_cam'].index.size}")
        print(f"len(self.acquisitions['vl_front_right_cam_frames']) {len(self.acquisitions['vl_front_right_cam_frames'])}")
        if len(self.acquisitions['vl_front_right_cam_frames']) != 0:
            print(f"self.acquisitions['vl_front_right_cam_frames'][0].shape {self.acquisitions['vl_front_right_cam_frames'][0].shape}")
        if self.acquisitions['vl_front_right_cam_lut_projection'][LUT_PROJECTION_X] is not None:
            print(f"self.acquisitions['vl_front_right_cam_lut_projection'][LUT_PROJECTION_X].shape {self.acquisitions['vl_front_right_cam_lut_projection'][LUT_PROJECTION_X].shape}")

        print(f"self.acquisitions['ahat_depth_cam'].index.size {self.acquisitions['ahat_depth_cam'].index.size}")
        print(f"len(self.acquisitions['ahat_depth_cam_frames']) {len(self.acquisitions['ahat_depth_cam_frames'])}")
        if len(self.acquisitions['ahat_depth_cam_frames']) != 0:
            print(f"self.acquisitions['ahat_depth_cam_frames'][0].shape {self.acquisitions['ahat_depth_cam_frames'][0].shape}")
        print(f"len(self.acquisitions['ahat_depth_cam_ab_frames']) {len(self.acquisitions['ahat_depth_cam_ab_frames'])}")
        if len(self.acquisitions['ahat_depth_cam_ab_frames']) != 0:
            print(f"self.acquisitions['ahat_depth_cam_ab_frames'][0].shape {self.acquisitions['ahat_depth_cam_ab_frames'][0].shape}")
        if self.acquisitions['ahat_depth_cam_lut_projection'][LUT_PROJECTION_X] is not None:
            print(f"self.acquisitions['ahat_depth_cam_lut_projection'][LUT_PROJECTION_X].shape {self.acquisitions['ahat_depth_cam_lut_projection'][LUT_PROJECTION_X].shape}")

        print(f"self.acquisitions['lt_depth_cam'].index.size {self.acquisitions['lt_depth_cam'].index.size}")
        print(f"len(self.acquisitions['lt_depth_cam_frames']) {len(self.acquisitions['lt_depth_cam_frames'])}")
        if len(self.acquisitions['lt_depth_cam_frames']) != 0:
            print(f"self.acquisitions['lt_depth_cam_frames'][0].shape {self.acquisitions['lt_depth_cam_frames'][0].shape}")
        print(f"len(self.acquisitions['lt_depth_cam_ab_frames']) {len(self.acquisitions['lt_depth_cam_ab_frames'])}")
        if len(self.acquisitions['lt_depth_cam_ab_frames']) != 0:
            print(f"self.acquisitions['lt_depth_cam_ab_frames'][0].shape {self.acquisitions['lt_depth_cam_ab_frames'][0].shape}")
        if self.acquisitions['lt_depth_cam_lut_projection'][LUT_PROJECTION_X] is not None:
            print(f"self.acquisitions['lt_depth_cam_lut_projection'][LUT_PROJECTION_X].shape {self.acquisitions['lt_depth_cam_lut_projection'][LUT_PROJECTION_X].shape}")

        if prepare_data:
            for acquisition in ACQUISITIONS + ['magnetic_seed']:
                # if self.acquisitions[acquisition].index.size != 0:
                    # self.acquisitions[acquisition].index = self.acquisitions[acquisition].index + pd.Timedelta(hours=UTC_SHIFT)
                if self.acquisitions[acquisition].index.size != 0 and self.acquisitions[acquisition].index.tzinfo is None:
                    self.acquisitions[acquisition].index = self.acquisitions[acquisition].index.tz_localize('CET')

            if self.acquisitions["qr_code_position"].index.size != 0:
                # for now, remove qr code positions when qr code 1 is not visible
                # if True:
                if False:
                    indexes = self.acquisitions["qr_code_position"][self.acquisitions["qr_code_position"]['q1_m44'] == 0].index
                    self.acquisitions["qr_code_position"] = self.acquisitions["qr_code_position"].drop(indexes)

            if 'frame 1' in self.acquisitions["optical"].columns:
                self.acquisitions["optical"] = self.acquisitions["optical"].drop([
                     'frame 1','unknown 1','nb markers 1','status 1.1','status 1.2','status 1.3','status 1.4'
                    ,'frame 2','unknown 2','nb markers 2','status 2.1','status 2.2','status 2.3','status 2.4'
                    ,'frame 3','unknown 3','nb markers 3','status 3.1','status 3.2','status 3.3','status 3.4'

                    , 'err 1'
                    , 'tx 1.1', 'ty 1.1', 'tz 1.1', 'tx 1.2', 'ty 1.2', 'tz 1.2', 'tx 1.3', 'ty 1.3', 'tz 1.3', 'tx 1.4', 'ty 1.4', 'tz 1.4'
                    , 'err 2'
                    , 'tx 2.1', 'ty 2.1', 'tz 2.1', 'tx 2.2', 'ty 2.2', 'tz 2.2', 'tx 2.3', 'ty 2.3', 'tz 2.3', 'tx 2.4', 'ty 2.4', 'tz 2.4'
                    , 'err 3'
                    , 'tx 3.1', 'ty 3.1', 'tz 3.1', 'tx 3.2', 'ty 3.2', 'tz 3.2', 'tx 3.3', 'ty 3.3', 'tz 3.3', 'tx 3.4', 'ty 3.4', 'tz 3.4' ],axis=1)
            self.acquisitions["probe"] = None
            self.acquisitions["pointer"] = None
            self.acquisitions["optical_seed"] = None

            self.acquisitions["probe"] = self.acquisitions["optical"].drop(self.acquisitions["optical"][self.acquisitions["optical"]['status 1'] != 'Enabled'].index)
            self.acquisitions["probe"] = self.acquisitions["probe"].drop(['status 1', 'status 2', 'status 3'
                , 'qx 2', 'qy 2', 'qz 2', 'qw 2', 'tx 2', 'ty 2', 'tz 2'
                , 'qx 3', 'qy 3', 'qz 3', 'qw 3', 'tx 3', 'ty 3', 'tz 3'],axis=1)
            print(f"self.acquisitions['probe'].index.size {self.acquisitions['probe'].index.size}")

            self.acquisitions["pointer"] = self.acquisitions["optical"].drop(self.acquisitions["optical"][self.acquisitions["optical"]['status 2'] != 'Enabled'].index)
            self.acquisitions["pointer"] = self.acquisitions["pointer"].drop(['status 1', 'status 2', 'status 3'
                , 'qx 1', 'qy 1', 'qz 1', 'qw 1', 'tx 1', 'ty 1', 'tz 1'
                , 'qx 3', 'qy 3', 'qz 3', 'qw 3', 'tx 3', 'ty 3', 'tz 3'],axis=1)
            print(f"self.acquisitions['pointer'].index.size {self.acquisitions['pointer'].index.size}")

            self.acquisitions["optical_seed"] = self.acquisitions["optical"].drop(self.acquisitions["optical"][self.acquisitions["optical"]['status 3'] != 'Enabled'].index)
            self.acquisitions["optical_seed"] = self.acquisitions["optical_seed"].drop(['status 1', 'status 2', 'status 3'
                , 'qx 2', 'qy 2', 'qz 2', 'qw 2', 'tx 2', 'ty 2', 'tz 2'
                , 'qx 1', 'qy 1', 'qz 1', 'qw 1', 'tx 1', 'ty 1', 'tz 1'],axis=1)
            print(f"self.acquisitions['optical_seed'].index.size {self.acquisitions['optical_seed'].index.size}")

        self.general_mutex.release()
