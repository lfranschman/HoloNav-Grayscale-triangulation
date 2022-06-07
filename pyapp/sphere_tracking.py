import time
import math
import numpy as np
import pandas as pd
import skimage as ski
import skimage.io
import scipy

from  python.common.File import load_pickle, save_pickle
from  python.common.UtilImage import draw_disk
from config import config
from DataAcquisition import DataAcquisition, ACQUISITIONS_HOLOLENS
from calibration_helpers import get_mat_c_to_w_series, get_mat_w_to_o, get_mat_m_to_o_series, get_lut_pixel_image, get_mat_divots_filename

def find_optical_spheres():
    data = DataAcquisition()
    newData = {}
    data.load_data(config.get_filename("optical_sphere"))

    newData["true_pos"] = pd.DataFrame([], columns=['index',
                                                                           'sphere1', 'sphere2',
                                                                           'sphere3', 'sphere4'])
    newData["true_pos"] = \
        newData["true_pos"].set_index("index")
    for acquisition in ACQUISITIONS_HOLOLENS:
        if not data.acquisitions[acquisition].empty:
            data.acquisitions[acquisition].index = data.acquisitions[acquisition].index + pd.Timedelta(seconds=config.temporal_shift_hololens)

    mat_qf_to_m = load_pickle(get_mat_divots_filename(config, "mat_qf_to_m"))

    # compute mat_w_to_o: optical/qr code reference
    # from 1 frame
    # if True:
    if False:
        timestamp = data.acquisitions["qr_code_position"].index[config.qr_code_optical_calibration_starting_time]
        qr_code_serie = data.acquisitions["qr_code_position"].loc[timestamp]
        optical_index = data.acquisitions["probe"].index.get_loc(timestamp, method='nearest')
        optical_timestamp = data.acquisitions["probe"].index[optical_index]
        optical_serie = data.acquisitions["probe"].loc[optical_timestamp]
        mat_w_to_o, _ = get_mat_w_to_o(qr_code_serie, optical_serie, mat_qf_to_m)
        mat_o_to_w = np.linalg.inv(mat_w_to_o)
    # average from all frames
    else:
        mat_w_to_o_list = []
        for qr_code_index in range(81,840):
            timestamp = data.acquisitions["qr_code_position"].index[qr_code_index]
            qr_code_serie = data.acquisitions["qr_code_position"].loc[timestamp]
            if qr_code_serie['q1_m44'] == 0: # front qr code not detected for this timestamp
                continue

            optical_index = data.acquisitions["probe"].index.get_loc(timestamp, method='nearest')
            optical_timestamp = data.acquisitions["probe"].index[optical_index]
            # we keep only if time difference is less than 20ms
            if abs((timestamp - optical_timestamp).total_seconds()) > 0.02:
                continue

            optical_serie = data.acquisitions["probe"].loc[optical_timestamp]
            mat_w_to_o, _ = get_mat_w_to_o(qr_code_serie, optical_serie, mat_qf_to_m)
            mat_w_to_o_list.append(mat_w_to_o)

        mat_w_to_o_list = np.array(mat_w_to_o_list) # shape (_, 4, 4)
        translation_mean = mat_w_to_o_list[:,:3,3] # shape (_, 3)
        translation_mean = np.mean(translation_mean, axis=0) # shape (3,)
        rotation = scipy.spatial.transform.Rotation.from_matrix(mat_w_to_o_list[:,:3,:3]).mean().as_matrix() # shape (3,3)

        mat_w_to_o = np.identity(4)
        mat_w_to_o[:3,:3] = rotation
        mat_w_to_o[:3,3] = translation_mean
        # print(mat_w_to_o)

        mat_o_to_w = np.linalg.inv(mat_w_to_o)

    for camera in ["vl_front_left_cam"]:
        print(f"camera {camera}")

        lut_projection = data.acquisitions[camera + "_lut_projection"]

        # nb_images_remaining = 40 # just want to do it for the nb_images_remaining first images
        # nb_images_remaining = 10000 # just want to do it for the nb_images_remaining first images
        # for frame_id in range(len(data.acquisitions[camera + "_frames"])):
        for frame_id in range(100, len(data.acquisitions[camera])):
            print(f"frame_id {frame_id}")
            timestamp = data.acquisitions[camera].index[frame_id]
            serie = data.acquisitions[camera].loc[timestamp]
            extrinsic = get_mat_c_to_w_series(serie)
            mat_w_to_c = np.linalg.inv(extrinsic)
            print(f"timestamp {timestamp}")
            optical_index = data.acquisitions["probe"].index.get_loc(timestamp, method='nearest')
            optical_timestamp = data.acquisitions["probe"].index[optical_index]
            # print(f"optical_index {optical_index} timestamp {timestamp} optical_timestamp {optical_timestamp}")

            # we keep only if time difference is less than 20ms
            if abs((timestamp - optical_timestamp).total_seconds()) > 0.025:
                continue # skip this frame

            optical_serie = data.acquisitions["probe"].loc[optical_timestamp]
            mat_m_to_o = get_mat_m_to_o_series(optical_serie)

            # compute mat_w_to_o each time with the qr code in the current frame
            # if True:
            if False:
                qr_code_index = data.acquisitions["qr_code_position"].index.get_loc(timestamp, method='nearest')
                qr_code_timestamp = data.acquisitions["qr_code_position"].index[qr_code_index]
                if abs((timestamp - qr_code_timestamp).total_seconds()) > 0.1:
                    continue # skip this frame

                qr_code_serie = data.acquisitions["qr_code_position"].loc[qr_code_timestamp]
                if qr_code_serie['q1_m44'] == 0: # front qr code not detected for this timestamp
                    continue # skip this frame

                mat_w_to_o, _ = get_mat_w_to_o(qr_code_serie, optical_serie, mat_qf_to_m)
                mat_o_to_w = np.linalg.inv(mat_w_to_o)

            # mat_m_to_c = np.matmul(mat_w_to_c, np.matmul(mat_o_to_w, mat_m_to_o))

            frame = np.copy(data.acquisitions[camera + '_frames'][frame_id]) # shape (h,w)
            # NDI optical_markers_8700339, check Polaris_Spectra_Tool_Kit_Guide.pdf for the position of the spheres
            sphere_positions = [[0,0,0,1], [0, 28.59, 41.02, 1], [0, 0, 88, 1], [0, -44.32, 40.45, 1]]
            draw_sphere = False
            points = []
            for i, sphere_pos in enumerate(sphere_positions):
                pos_sphere1_c = np.matmul(mat_o_to_w, np.matmul(mat_m_to_o, sphere_pos))
                # t1 = time.time()
                # coord = get_lut_pixel_image(lut_projection, pos_sphere1_c[0], pos_sphere1_c[1], pos_sphere1_c[2])
                # print(f"get lut {time.time() - t1} s")
                if pos_sphere1_c is not None and not math.isnan(pos_sphere1_c[0]) and not math.isnan(pos_sphere1_c[1]) and not math.isnan(pos_sphere1_c[2]):
                    # print(frame_id, ":")
                    print(f"center of the sphere {i}: {pos_sphere1_c}")
                    # draw_disk(frame, coord[0], coord[1], 255, size=1)
                    # draw_sphere = True
                    points.append(pos_sphere1_c)
            dataEntry = [points[0], points[1], points[2], points[3]]
            newData["true_pos"].loc[frame_id] = dataEntry
    save_pickle(newData, config.get_filename("world_positions"))
        # return points
            # if draw_sphere:
            #     ski.io.imsave(f"generated/{camera}_{frame_id:04.0f}.png", frame)

            # nb_images_remaining -= 1
            # if nb_images_remaining == 0:
            #     break

if  __name__ == '__main__':
    print("main")
    find_optical_spheres()
