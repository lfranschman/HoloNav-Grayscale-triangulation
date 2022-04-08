import numpy as np
import pandas as pd

from File import load_pickle
from QRCodeDetection import QRCodeDetection
from config import config
from DataAcquisition import DataAcquisition, ACQUISITIONS_HOLOLENS
from calibration_helpers import get_mat_c_to_w_series, get_mat_w_to_o, get_mat_q_to_w_series, get_mat_m_to_o_series, get_mat_divots_filename

def create_qr_code_positions():
    data = DataAcquisition()
    data.load_data(config.get_filename("optical_sphere"))

    data.acquisitions["qr_code_position"] = pd.DataFrame([], columns = ['time'
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
    data.acquisitions["qr_code_position"] = data.acquisitions["qr_code_position"].set_index('time')

    qr_code_detection = QRCodeDetection()


    #################################################################################################################
    # pv_cam
    # if True:
    if False:
        for i in range(len(data.acquisitions["pv_cam_frames"])):
            timestamp = data.acquisitions["pv_cam"].index[i]
            serie = data.acquisitions["pv_cam"].loc[timestamp]

            frame = data.acquisitions['pv_cam_frames'][i][...,[2,1,0]] # shape (h,w,rgb)
            extrinsic = get_mat_c_to_w_series(serie)
            intrinsic = np.array([[serie['focal_length_x'], 0                      , serie['center_coordinate_x']] \
                                , [0                      , serie['focal_length_y'], serie['center_coordinate_y']] \
                                , [0                      , 0                      ,1                           ]])
            distortion_coefficients = np.array([serie['radial_distortion_x'], serie['radial_distortion_y'], serie['tangential_distortion_x'], serie['tangential_distortion_y'], serie['radial_distortion_z']])

            mat_q_to_w_list = qr_code_detection.detect(frame, extrinsic, intrinsic, distortion_coefficients)

            data.acquisitions["qr_code_position"].loc[timestamp] = tuple(mat_q_to_w_list.flatten())


    #################################################################################################################
    # front left and right vl_cam triangulation
    else:
    # elif True:
    # elif False:
        for i in range(len(data.acquisitions["vl_front_left_cam_frames"])):
            timestamp_left = data.acquisitions["vl_front_left_cam"].index[i]
            serie_left = data.acquisitions["vl_front_left_cam"].loc[timestamp_left]
            frame_left = data.acquisitions['vl_front_left_cam_frames'][i] # shape (h,w)
            extrinsic_left = get_mat_c_to_w_series(serie_left)
            lut_projection_left = data.acquisitions["vl_front_left_cam_lut_projection"] # shape (2, height*2 + 1, width*2 + 1)

            index_right = data.acquisitions["vl_front_right_cam"].index.get_loc(timestamp_left, method='nearest')
            timestamp_right = data.acquisitions["vl_front_right_cam"].index[index_right]
            serie_right = data.acquisitions["vl_front_right_cam"].loc[timestamp_right]
            frame_right = data.acquisitions['vl_front_right_cam_frames'][index_right] # shape (h,w)
            extrinsic_right = get_mat_c_to_w_series(serie_right)
            lut_projection_right = data.acquisitions["vl_front_right_cam_lut_projection"] # shape (2, height*2 + 1, width*2 + 1)

            mat_q_to_w_list = np.zeros((QRCodeDetection.NB_QR_CODE,4,4), dtype=np.float64)
            # if time difference is less than 100ms
            if abs((timestamp_left - timestamp_right).total_seconds()) < 0.1:
                mat_q_to_w_list = qr_code_detection.detect_triangulation(frame_left, extrinsic_left, lut_projection_left, frame_right, extrinsic_right, lut_projection_right)

                data.acquisitions["qr_code_position"].loc[timestamp_left] = tuple(mat_q_to_w_list.flatten())


    #################################################################################################################
    # save to visualize qr code position (warning slow)
    # data.save_data(config.get_filename("optical_sphere_vl"))


    #################################################################################################################
    # check distances qr code center between optical and qr code detection
    for acquisition in ACQUISITIONS_HOLOLENS:
        if not data.acquisitions[acquisition].empty:
            data.acquisitions[acquisition].index = data.acquisitions[acquisition].index + pd.Timedelta(seconds=config.temporal_shift_hololens)

    mat_qf_to_m = load_pickle(get_mat_divots_filename(config, "mat_qf_to_m"))

    # compute mat_w_to_o: optical/qr code reference
    timestamp = data.acquisitions["qr_code_position"].index[config.qr_code_optical_calibration_starting_time]
    qr_code_serie = data.acquisitions["qr_code_position"].loc[timestamp]
    optical_index = data.acquisitions["probe"].index.get_loc(timestamp, method='nearest')
    optical_timestamp = data.acquisitions["probe"].index[optical_index]
    optical_serie = data.acquisitions["probe"].loc[optical_timestamp]
    mat_w_to_o, _ = get_mat_w_to_o(qr_code_serie, optical_serie, mat_qf_to_m)

    # print(f"mat_w_to_o {mat_w_to_o}")
    # test with mat_w_to_o done with pv_cam
    # mat_w_to_o = np.array( \
    # [[ 4.52996123e-02, -9.98107891e-01, -4.15519444e-02, -6.51342191e+02] \
    # ,[ 2.69222696e-01,  5.22537551e-02, -9.61658118e-01, -5.52302696e+02] \
    # ,[ 9.62010799e-01,  3.23761017e-02,  2.71080742e-01, -1.79286068e+03] \
    # ,[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  1.00000000e+00]] \
    # )

    for i in range(len(data.acquisitions["qr_code_position"])):
        timestamp = data.acquisitions["qr_code_position"].index[i]
        qr_code_serie = data.acquisitions["qr_code_position"].loc[timestamp]

        if qr_code_serie['q1_m44'] == 0: # front qr code not detected for this timestamp
            continue

        optical_index = data.acquisitions["probe"].index.get_loc(timestamp, method='nearest')
        optical_timestamp = data.acquisitions["probe"].index[optical_index]
        optical_serie = data.acquisitions["probe"].loc[optical_timestamp]

        # if optical acquisition time is more than 50ms to the qr code position
        if abs((timestamp - optical_timestamp).total_seconds()) > 0.05:
            continue

        mat_q_to_w = get_mat_q_to_w_series(qr_code_serie)
        mat_m_to_o = get_mat_m_to_o_series(optical_serie)

        pos_optical_q = np.array([0,0,0,1])
        pos_optical_m = np.matmul(mat_qf_to_m, pos_optical_q)
        pos_optical_o = np.matmul(mat_m_to_o, pos_optical_m)

        pos_qr_code_q = np.array([0,0,0,1])
        pos_qr_code_w = np.matmul(mat_q_to_w, pos_qr_code_q)
        pos_qr_code_o = np.matmul(mat_w_to_o, pos_qr_code_w)

        distance_optical_qr_code = np.linalg.norm(pos_optical_o[:3] - pos_qr_code_o[:3])
        print(f"distance_optical_qr_code {distance_optical_qr_code} mm at {timestamp} index {i}")

if  __name__ == '__main__':
    print("main")
    create_qr_code_positions()
