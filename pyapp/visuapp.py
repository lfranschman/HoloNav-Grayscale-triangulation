import threading
import copy
import time
import numpy as np
import pandas as pd

from python.common.File import load_pickle, is_file_exist
from python.common.UtilMaths import vec3_to_vec4, identity_matrix44, translation_matrix44, rotation_euler_matrix44, mul_mat44_vec4
from python.common.Logging import log_print

from view3d import View3D
from calibration_helpers import get_mat_m_to_o_series, get_mat_q_to_w, get_mat_m_to_o, get_mat_w_to_o, get_pointer_offset_filename, get_mat_divots_filename, pointer_pivot_calibration, register_divots
from divot_calibration import register_divots_front_qr_code, register_divots_left_qr_code, register_divots_right_qr_code, register_divots_top_qr_code, register_divots_seed_holder
from config import config
from DataAcquisition import DataAcquisition, ACQUISITIONS, ACQUISITIONS_HOLOLENS, RESEARCH_MODE_CAMERA_NAMES

ALL_VISIBILITY = ACQUISITIONS_HOLOLENS + ["ahat_ab", "lt_ab"] + ["probe", "pointer", "optical_seed"]

NDI_STYLUS_TOOL_TIP_OFFSET = (-18.43299216, 1.01690107, -157.39878748)
# NDI_STYLUS_TOOL_TIP_OFFSET = (-(9.55 + (8.77 - 0.64)), 0, -(142.9 + (156.7 - 135) - 5.22), 1) # guess with the documentation

# PROBE_TIP_OPTICAL_DEFAULT = np.array((-38.7280596, -47.26084413, -179.83994926)) # old
PROBE_TIP_OPTICAL_DEFAULT = np.array((-85, 255, -36)) # random but close values to reality
# PROBE_TIP_OPTICAL_DEFAULT = np.array((0, 0, 0))
# SEED_O_DEFAULT = np.array((94.79942413, 301.52460434, -1947.86163424))
SEED_O_DEFAULT = np.array((0, 0, 0))

MAT_QF_TO_M_DEFAULT = np.matmul(translation_matrix44((-40,50,-50)), rotation_euler_matrix44((180,0,0),degrees=True))
MAT_QF_TO_QL_DEFAULT = identity_matrix44()
MAT_QF_TO_QR_DEFAULT = identity_matrix44()
MAT_QF_TO_QT_DEFAULT = identity_matrix44()
MAT_S_TO_M2_DEFAULT = identity_matrix44()

PROBE_TIP_QR_CODE_DEFAULT = np.array((-4, -224, -17)) # random but close values to reality
# PROBE_TIP_QR_CODE_DEFAULT = np.array((0, 0, 0))
# SEED_W_DEFAULT = np.array((70, 340, 270))
SEED_W_DEFAULT = np.array((0, 0, 0))

ROUTINE_SLEEP_TIME = 0.05 # in s

class CalibrationParameters:
    def __init__(self, init_seed, init_tip, tip_unknown, end_callback):
        self.init_seed = init_seed
        self.init_tip = init_tip
        self.tip_unknown = tip_unknown
        self.end_callback = end_callback

class App:
    ACTION_CALIBRATION_OPTICAL_POINTER = 0
    ACTION_CALIBRATION_OPTICAL_FRONT_QR_CODE = 1
    ACTION_CALIBRATION_OPTICAL_LEFT_QR_CODE = 2
    ACTION_CALIBRATION_OPTICAL_RIGHT_QR_CODE = 3
    ACTION_CALIBRATION_OPTICAL_TOP_QR_CODE = 4
    ACTION_CALIBRATION_OPTICAL_SEED = 5
    ACTION_OPTICAL_CALIBRATION = 6
    ACTION_QR_CODE_CALIBRATION = 7

    def __init__(self):
        self.actions = []
        self.thread = None

        self.view3d = View3D(self)

        self.mutex_update = threading.Lock()
        self.has_to_update = True
        self.visibility = dict.fromkeys(ALL_VISIBILITY, True)
        self.visibility.update(dict.fromkeys(RESEARCH_MODE_CAMERA_NAMES + ["ahat_ab", "lt_ab"], False))

        self.data = DataAcquisition()
        self.data.load_data(config.get_filename())
        self.acquisitions = self.data.acquisitions

        self.df = {}
        self.slider_value = {}

        if not self.acquisitions["optical"].empty:
            self.slider_value["optical"] = 0
            self.set_temporal_shift_optical(config.temporal_shift_optical)

            self.optical_calibration_starting_time = config.optical_calibration_starting_time[config.sub_config]
            self.optical_calibration_ending_time = config.optical_calibration_ending_time[config.sub_config]
            if self.optical_calibration_starting_time == -1:
                self.optical_calibration_starting_time = self.slider_value["optical"]
                self.optical_calibration_ending_time = self.acquisitions["optical"].index.size - 1

            self.offset_tip_pointer = NDI_STYLUS_TOOL_TIP_OFFSET
            if is_file_exist(get_pointer_offset_filename(config)):
                self.offset_tip_pointer = load_pickle(get_pointer_offset_filename(config))
                log_print(f"offset_tip_pointer {self.offset_tip_pointer}")
            # self.pos_t_m = [0,0,0]
            self.pos_t_m = PROBE_TIP_OPTICAL_DEFAULT

            self.mat_qf_to_m = MAT_QF_TO_M_DEFAULT # qr code to optical marker (8700339)
            if is_file_exist(get_mat_divots_filename(config, "mat_qf_to_m")):
                self.mat_qf_to_m = load_pickle(get_mat_divots_filename(config, "mat_qf_to_m"))
                log_print(f"mat_qf_to_m {self.mat_qf_to_m}")
            self.mat_qf_to_ql = MAT_QF_TO_QL_DEFAULT
            if is_file_exist(get_mat_divots_filename(config, "mat_qf_to_ql")):
                self.mat_qf_to_ql = load_pickle(get_mat_divots_filename(config, "mat_qf_to_ql"))
                log_print(f"mat_qf_to_ql {self.mat_qf_to_ql}")
            self.mat_qf_to_qr = MAT_QF_TO_QR_DEFAULT
            if is_file_exist(get_mat_divots_filename(config, "mat_qf_to_qr")):
                self.mat_qf_to_qr = load_pickle(get_mat_divots_filename(config, "mat_qf_to_qr"))
                log_print(f"mat_qf_to_qr {self.mat_qf_to_qr}")
            self.mat_qf_to_qt = MAT_QF_TO_QT_DEFAULT
            if is_file_exist(get_mat_divots_filename(config, "mat_qf_to_qt")):
                self.mat_qf_to_qt = load_pickle(get_mat_divots_filename(config, "mat_qf_to_qt"))
                log_print(f"mat_qf_to_qt {self.mat_qf_to_qt}")
            self.mat_s_to_m2 = MAT_S_TO_M2_DEFAULT # seed to optical marker 2 (8700449)
            if is_file_exist(get_mat_divots_filename(config, "mat_s_to_m2")):
                self.mat_s_to_m2 = load_pickle(get_mat_divots_filename(config, "mat_s_to_m2"))
                log_print(f"mat_s_to_m2 {self.mat_s_to_m2}")

            # self.pos_s_o = [0,0,0]
            self.pos_s_o = SEED_O_DEFAULT

            self.registration_divots_pts = None
            self.registration_cloud_pts = None

        if self.available_hololens_acquisitions():
            for acquisition in ACQUISITIONS_HOLOLENS:
                if not self.acquisitions[acquisition].empty:
                    self.slider_value[acquisition] = 0

            self.set_temporal_shift_hololens(config.temporal_shift_hololens)

            self.mat_w_to_o = identity_matrix44()

        if not self.acquisitions["qr_code_position"].empty:
            self.qr_code_optical_calibration_starting_time = config.qr_code_optical_calibration_starting_time
            self.qr_code_calibration_starting_time = config.qr_code_calibration_starting_time[config.sub_config]
            self.qr_code_calibration_ending_time = config.qr_code_calibration_ending_time[config.sub_config]
            if self.qr_code_calibration_starting_time == -1:
                self.qr_code_calibration_starting_time = self.slider_value["qr_code_position"]
                self.qr_code_calibration_ending_time = self.acquisitions["qr_code_position"].index.size - 1

            # self.pos_t_q = [0,0,0]
            self.pos_t_q = PROBE_TIP_QR_CODE_DEFAULT

            # self.pos_s_w = [0,0,0]
            self.pos_s_w = SEED_W_DEFAULT

            if not self.acquisitions["optical"].empty:
                qr_code_index = self.qr_code_optical_calibration_starting_time
                timestamp = self.df["qr_code_position"].index[qr_code_index]
                qr_code_probe_series = self.df["qr_code_position"].loc[timestamp]
                if qr_code_probe_series['q1_m44'] != 0: # front qr code was detected for this timestamp
                    # todo make an interpolation between the two close positions
                    optical_index = self.df_probe.index.get_loc(timestamp, method='nearest')
                    optical_timestamp = self.df_probe.index[optical_index]
                    optical_probe_series = self.df_probe.loc[optical_timestamp]
                    print(f"compute mat_w_to_o, timestamp difference between optical/qr code {(timestamp - optical_timestamp).total_seconds()} s")

                    self.mat_w_to_o, _ = get_mat_w_to_o(qr_code_probe_series, optical_probe_series, self.mat_qf_to_m)

    def available_hololens_acquisitions(self):
        for acquisition in ACQUISITIONS_HOLOLENS:
            if not self.acquisitions[acquisition].empty:
                return True
        return False

    def set_slider_value(self, acquisition, value):
        self.mutex_update.acquire()
        self.slider_value[acquisition] = value

        for other_acquisition in ACQUISITIONS:
            if other_acquisition != acquisition:
                # if not self.acquisitions[other_acquisition].empty:
                if other_acquisition in self.df:
                    timestamp = self.df[acquisition].index[self.slider_value[acquisition]]
                    self.slider_value[other_acquisition] = self.df[other_acquisition].index.get_loc(timestamp, method='nearest')

        self.mutex_update.release()

    def set_temporal_shift_optical(self, temporal_shift_optical):
        self.mutex_update.acquire()

        self.temporal_shift_optical = temporal_shift_optical

        self.df["optical"] = self.acquisitions["optical"].copy()
        self.df["optical"].index = self.df["optical"].index + pd.Timedelta(seconds=temporal_shift_optical)

        self.df_pointer = self.acquisitions["pointer"].copy()
        self.df_pointer.index = self.df_pointer.index + pd.Timedelta(seconds=temporal_shift_optical)

        self.df_probe = self.acquisitions["probe"].copy()
        self.df_probe.index = self.df_probe.index + pd.Timedelta(seconds=temporal_shift_optical)

        self.df_optical_seed = self.acquisitions["optical_seed"].copy()
        self.df_optical_seed.index = self.df_optical_seed.index + pd.Timedelta(seconds=temporal_shift_optical)

        self.mutex_update.release()

        self.set_slider_value("optical", self.slider_value["optical"])

    def set_temporal_shift_hololens(self, temporal_shift_hololens):
        self.mutex_update.acquire()

        self.temporal_shift_hololens = temporal_shift_hololens

        for acquisition in ACQUISITIONS_HOLOLENS:
            if not self.acquisitions[acquisition].empty:
                last_acquisition = acquisition
                self.df[acquisition] = self.acquisitions[acquisition].copy()
                self.df[acquisition].index = self.df[acquisition].index + pd.Timedelta(seconds=self.temporal_shift_hololens)

        self.mutex_update.release()

        self.set_slider_value(last_acquisition, self.slider_value[last_acquisition])

        return last_acquisition

    def run(self):
        self.view3d.run()

        self.thread = threading.Thread(target=self.routine)
        self.thread.daemon = True
        self.thread.start()

    def stop(self):
        # should send stop to self.thread

        self.view3d.stop()

    def routine(self):
        while True:
            time.sleep(ROUTINE_SLEEP_TIME)

            self.mutex_update.acquire()
            actions_tmp = copy.copy(self.actions)
            if not self.acquisitions["optical"].empty:
                mat_qf_to_m = copy.copy(self.mat_qf_to_m)
            self.actions = []
            self.mutex_update.release()

            for action in actions_tmp:
                if isinstance(action, list):
                    if action[0] == self.ACTION_OPTICAL_CALIBRATION:
                        self.optical_calibration(action[1])
                    elif action[0] == self.ACTION_QR_CODE_CALIBRATION:
                        self.qr_code_calibration(action[1])

                elif action == self.ACTION_CALIBRATION_OPTICAL_POINTER:
                    self.pointer_pivot_calibration()
                elif action == self.ACTION_CALIBRATION_OPTICAL_FRONT_QR_CODE:
                    self.register_divots(register_divots_front_qr_code, "mat_qf_to_m", None, True, config.order_divot_index, config.to_delete_gt_divot_index)
                elif action == self.ACTION_CALIBRATION_OPTICAL_LEFT_QR_CODE:
                    self.register_divots(register_divots_left_qr_code, "mat_qf_to_ql", mat_qf_to_m, False, config.order_divot_index, config.to_delete_gt_divot_index)
                elif action == self.ACTION_CALIBRATION_OPTICAL_RIGHT_QR_CODE:
                    self.register_divots(register_divots_right_qr_code, "mat_qf_to_qr", mat_qf_to_m, False, config.order_divot_index, config.to_delete_gt_divot_index)
                elif action == self.ACTION_CALIBRATION_OPTICAL_TOP_QR_CODE:
                    self.register_divots(register_divots_top_qr_code, "mat_qf_to_qt", mat_qf_to_m, False, config.order_divot_index, config.to_delete_gt_divot_index)
                elif action == self.ACTION_CALIBRATION_OPTICAL_SEED:
                    self.register_divots(register_divots_seed_holder, "mat_s_to_m2", None, True, config.order_divot_index, config.to_delete_gt_divot_index)

    def pointer_pivot_calibration(self):
        offset_tip_pointer = pointer_pivot_calibration(self.acquisitions["pointer"], config, average_with_previous_calibration=False)
        self.mutex_update.acquire()
        self.offset_tip_pointer = offset_tip_pointer
        self.has_to_update = True
        self.mutex_update.release()

    def register_divots(self, register_divots_fct, variable, mat_qf_to_m=None, inverse=True, order_divot_index=None, to_delete_gt_divot_index=None):
        mat, divots_pts, cloud_pts = register_divots(self.acquisitions["optical"], config, self.offset_tip_pointer, register_divots_fct, variable, mat_qf_to_m, inverse, order_divot_index, to_delete_gt_divot_index)

        self.mutex_update.acquire()
        self.registration_divots_pts = divots_pts
        self.registration_cloud_pts = cloud_pts
        self.__dict__[variable] = mat
        self.has_to_update = True
        self.mutex_update.release()

    def optical_calibration(self, calibration_parameters):
        calibration_starting_datetime = (self.df["optical"].index[self.optical_calibration_starting_time]) # - datetime.timedelta(hours=2))
        calibration_ending_datetime = (self.df["optical"].index[self.optical_calibration_ending_time]) # - datetime.timedelta(hours=2))
        log_print(f"start {calibration_starting_datetime} end {calibration_ending_datetime}")

        pos_s_o, pos_t_m, residuals, residuals_timestamp = solve_equations(self.acquisitions["magnetic_seed"], self.df_probe, calibration_starting_datetime, calibration_ending_datetime, get_mat=get_mat_m_to_o, init_seed=calibration_parameters.init_seed, init_tip=calibration_parameters.init_tip, tip_unknown=calibration_parameters.tip_unknown)

        self.mutex_update.acquire()
        self.pos_t_m = pos_t_m
        self.pos_s_o = pos_s_o
        self.has_to_update = True
        self.mutex_update.release()

        calibration_parameters.end_callback(pos_t_m, pos_s_o, residuals, residuals_timestamp)

        # compute distance error
        log_print(f"self.pos_s_o {self.pos_s_o}")
        # mat_m2_to_o = get_mat_m_to_o_series(self.df["optical"].loc[self.df["optical"].index[536]], optical_marker_id=3) # 2021_12_02
        # mat_m2_to_o = get_mat_m_to_o_series(self.df["optical"].loc[self.df["optical"].index[1171]], optical_marker_id=3) # 2022_01_28
        mat_m2_to_o = get_mat_m_to_o_series(self.df["optical"].loc[self.df["optical"].index[self.optical_calibration_starting_time]], optical_marker_id=3)
        pos_gt_s_s = np.array((0,0,0,1))
        pos_gt_s_m2 = mul_mat44_vec4(self.mat_s_to_m2, pos_gt_s_s)
        pos_gt_s_o = mul_mat44_vec4(mat_m2_to_o, pos_gt_s_m2)
        log_print(f"pos_gt_s_o {pos_gt_s_o}")
        distance_error = np.linalg.norm(self.pos_s_o[:3] - pos_gt_s_o[:3])
        log_print(f"distance_error {distance_error}")

    def qr_code_calibration(self, calibration_parameters):
        timestamp = self.df["qr_code_position"].index[self.qr_code_calibration_starting_time]
        calibration_starting_datetime = timestamp
        calibration_ending_datetime = self.df["qr_code_position"].index[self.qr_code_calibration_ending_time]
        log_print(f"start {calibration_starting_datetime} end {calibration_ending_datetime}")

        pos_s_w, pos_t_q, residuals, residuals_timestamp = solve_equations(self.acquisitions["magnetic_seed"], self.df["qr_code_position"], calibration_starting_datetime, calibration_ending_datetime, get_mat=get_mat_q_to_w, init_seed=calibration_parameters.init_seed, init_tip=calibration_parameters.init_tip, tip_unknown=calibration_parameters.tip_unknown)

        self.mutex_update.acquire()
        self.pos_t_q = pos_t_q
        self.pos_s_w = pos_s_w
        self.has_to_update = True
        self.mutex_update.release()

        calibration_parameters.end_callback(pos_t_q, pos_s_w, residuals, residuals_timestamp)

        # compute distance error
        log_print(f"self.pos_s_w {self.pos_s_w}")

        # todo make an interpolation between the two close positions
        optical_index = self.df_optical_seed.index.get_loc(timestamp, method='nearest')
        optical_timestamp = self.df_optical_seed.index[optical_index]
        optical_probe_series = self.df_optical_seed.loc[optical_timestamp]
        print(f"qr_code_calibration timestamp difference between optical/qr code {(timestamp - optical_timestamp).total_seconds()} s")

        mat_m2_to_o = get_mat_m_to_o_series(optical_probe_series) #, optical_marker_id=3)
        pos_gt_s_s = np.array((0,0,0,1))
        pos_gt_s_m2 = mul_mat44_vec4(self.mat_s_to_m2, pos_gt_s_s)
        pos_gt_s_o = mul_mat44_vec4(mat_m2_to_o, pos_gt_s_m2)
        log_print(f"pos_gt_s_o {pos_gt_s_o}")
        pos_s_o = mul_mat44_vec4(self.mat_w_to_o, vec3_to_vec4(pos_s_w))
        log_print(f"pos_s_o {pos_s_o}")
        distance_error = np.linalg.norm(pos_s_o[:3] - pos_gt_s_o[:3])
        log_print(f"distance_error {distance_error}")
