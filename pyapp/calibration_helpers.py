import time
import math
import copy
import numpy as np
import scipy
import scipy.optimize # fsolve, root, least_squares

from Logging import log_print
from UtilMaths import get_mat_camera_to_world, get_mat_quaternion, get_inverse_mat_quaternion, mul_mat44_vec4, normalize_vec4, mul_mat44_vec4_list, vec3_list_to_vec4_list, point_in_triangle, barycentric, vec3_to_vec4, polar_to_cartesian
from File import load_pickle, save_pickle

from divot_calibration import pivot_calibration

def get_mat_c_to_w_series(series):
    translation_rig_to_world = [series['cam_to_w_translation_x'], series['cam_to_w_translation_y'], series['cam_to_w_translation_z']]
    translation_rig_to_world.append(1.)
    rotation_rig_to_world = [series['cam_to_w_rotation_x'], series['cam_to_w_rotation_y'], series['cam_to_w_rotation_z'], series['cam_to_w_rotation_w']]
    rig_to_camera = np.array([[series['m11'], series['m12'], series['m13'], series['m14']]
        , [series['m21'], series['m22'], series['m23'], series['m24']]
        , [series['m31'], series['m32'], series['m33'], series['m34']]
        , [series['m41'], series['m42'], series['m43'], series['m44']]])
    return get_mat_camera_to_world(translation_rig_to_world, rotation_rig_to_world, rig_to_camera)

def get_mat_c_to_w(df, timestamp):
    return get_mat_c_to_w_series(df.loc[timestamp])

 # TODO average qr code, qr_code_id = -1 ?
def get_mat_q_to_w_series(series, qr_code_id = 0):
    qr_code_id += 1
    mat_q_to_w = np.array([[series[f"q{qr_code_id}_m11"], series[f"q{qr_code_id}_m12"], series[f"q{qr_code_id}_m13"], series[f"q{qr_code_id}_m14"]]
    , [series[f"q{qr_code_id}_m21"], series[f"q{qr_code_id}_m22"], series[f"q{qr_code_id}_m23"], series[f"q{qr_code_id}_m24"]]
    , [series[f"q{qr_code_id}_m31"], series[f"q{qr_code_id}_m32"], series[f"q{qr_code_id}_m33"], series[f"q{qr_code_id}_m34"]]
    , [series[f"q{qr_code_id}_m41"], series[f"q{qr_code_id}_m42"], series[f"q{qr_code_id}_m43"], series[f"q{qr_code_id}_m44"]]])
    return mat_q_to_w

def get_mat_q_to_w(df_qr_code, timestamp, qr_code_id = 0):
    # translation_vec = list(df_qr_code.loc[timestamp][:3])
    return get_mat_q_to_w_series(df_qr_code.loc[timestamp], qr_code_id)

# lut shape (2, height*2 + 1, width*2 + 1)
# output shape (3,)
# we take the value in the middle of the pixel (x + 0.5, y + 0.5)
def get_lut_projection_pixel(lut, x, y):
    # width = (lut[0].shape[1] - 1)//2
    # index = (y*2 + 1)*lut[0].shape[1] + (x*2 + 1) # we take the value in the middle of the pixel (x + 0.5, y + 0.5)
    # return [lut[0][index], lut[1][index], 1]
    return [lut[0][int(y*2 + 1), int(x*2 + 1)], lut[1][int(y*2 + 1), int(x*2 + 1)], 1]

# brute force to find the position in the 2D camera image coordinate space (TODO improve this, too slow) -> use intrinsic + distortion coefficients instead
# lut shape (2, height*2 + 1, width*2 + 1)
# (x,y,z) in camera space
def get_lut_pixel_image(lut, x, y, z):
    if z <= 0:
        return None

    # find projection line when z = 1, line parametric equation -> origin + t.([x,y,z] - origin)
    t = 1/z
    x = t*x
    y = t*y
    # print(f"t {t} x {x} y {y}")

    # find the 3 closest points
    closest = {0:[(-1,-1),(-1,-1),99999999999], 1:[(-1,-1),(-1,-1),99999999999], 2:[(-1,-1),(-1,-1),99999999999]}
    CLOSEST_CAMERA_SPACE = 0
    CLOSEST_IMAGE_SPACE = 1
    CLOSEST_DISTANCE = 2
    for j in range(lut[0].shape[0]):
        for i in range(lut[0].shape[1]):
            dist = (x - lut[0][j,i])**2 + (y - lut[1][j,i])**2
            for k in range(3):
                if dist < closest[k][CLOSEST_DISTANCE]:
                    if k <= 1:
                        closest[2] = copy.copy(closest[1])
                    if k == 0:
                        closest[1] = copy.copy(closest[0])
                    closest[k] = [(lut[0][j,i], lut[1][j,i]), (i/2, j/2), dist]
                    break

        # print(f"closest {closest}")
        # if point_in_triangle((x,y), closest[0][CLOSEST_CAMERA_SPACE], closest[1][CLOSEST_CAMERA_SPACE], closest[2][CLOSEST_CAMERA_SPACE]):
        b = barycentric(np.array((x,y)), np.array((closest[0][CLOSEST_CAMERA_SPACE])), np.array((closest[1][CLOSEST_CAMERA_SPACE])), np.array((closest[2][CLOSEST_CAMERA_SPACE])))
        if b is None:
            return None
        image_x = b[0]*closest[0][CLOSEST_IMAGE_SPACE][0] + b[1]*closest[1][CLOSEST_IMAGE_SPACE][0] + b[2]*closest[2][CLOSEST_IMAGE_SPACE][0]
        image_y = b[0]*closest[0][CLOSEST_IMAGE_SPACE][1] + b[1]*closest[1][CLOSEST_IMAGE_SPACE][1] + b[2]*closest[2][CLOSEST_IMAGE_SPACE][1]
        return (image_x, image_y)

    return None

# lut_x/lut_y shape (_, _)
# output shape (_*_, 3)
def get_normalized_lut_projection(lut_x, lut_y):
    lut_x = lut_x.flatten() # shape (_*_)
    # print(f"lut_x.shape {lut_x.shape}")
    lut_y = lut_y.flatten() # shape (_*_)
    lut_z = np.ones(lut_x.shape)

    inv_norm = 1/np.sqrt(lut_x*lut_x + lut_y*lut_y + 1*1) # z = 1
    inv_norm = np.repeat(inv_norm[:, np.newaxis], 3, axis=1) # shape (_*_, 3)
    #print(inv_norm.shape)

    lut = np.stack((lut_x, lut_y, lut_z),axis=1) # shape (_*_, 3)
    lut = lut*inv_norm
    return lut

# get only the lut for the middle of the pixels (0.5,0.5), (0.5,1.5), ...
# lut shape (2, height*2 + 1, width*2 + 1)
# output shape (height*width, 3)
def get_lut_projection_pixel_mapping(lut):
    lut_x = lut[0][1:-1,1:-1]
    lut_x = lut_x[0::2, 0::2] # shape (height, width)
    # print(f"lut_x.shape {lut_x.shape} lut_x.dtype {lut_x.dtype}")
    lut_y = lut[1][1:-1,1:-1]
    lut_y = lut_y[0::2, 0::2] # shape (height, width)
    return get_normalized_lut_projection(lut_x, lut_y) # shape (height*width, 3)

# get only the lut for the top-left corner of the pixels (0.,0.), (0.,1.*skip_factor), ...
# lut shape (2, height*2 + 1, width*2 + 1)
# output shape ((height/skip_factor + 1)*(width/skip_factor + 1), 3)
def get_lut_projection_camera_mapping(lut, skip_factor=1):
    lut_x = lut[0][0::2*skip_factor, 0::2*skip_factor] # shape (height/skip_factor + 1, width/skip_factor + 1)
    lut_y = lut[1][0::2*skip_factor, 0::2*skip_factor] # shape (height/skip_factor + 1, width/skip_factor + 1)

    # return get_normalized_lut_projection(lut_x, lut_y) # shape ((height/skip_factor + 1)*(width/skip_factor + 1), 3)

    lut_x = lut_x.flatten() # shape (_*_)
    lut_y = lut_y.flatten() # shape (_*_)
    lut_z = np.ones(lut_x.shape)
    return np.stack((lut_x, lut_y, lut_z),axis=1) # shape ((height/skip_factor + 1)*(width/skip_factor + 1), 3)

# depth_image shape (height, width) dtype=np.uint16
# lut_projection shape (height*width, 3)
# output (height*width, 3)
def get_3d_points_in_camera_space(depth_image, lut_projection, clamp_max):
    depth_image[depth_image > clamp_max] = 0
    depth = np.tile(depth_image.flatten().reshape((-1, 1)), (1, 3)) # (height*width, 3)
    # print(f"depth.shape {depth.shape}")

    points = depth*lut_projection # shape (height*width, 3)

    remove_ids = np.where(np.sum(points, axis=1) < 1e-6)[0] # remove invalid points
    points = np.delete(points, remove_ids, axis=0)
    return points, remove_ids

# frame shape (height, width, 3) dtype=np.uint8 min=0 max=255
# depth_image shape (height, width) dtype=np.uint16
# lut_projection shape (height*width, 3)
# output shape (height*width - removed points, 3), (height*width - removed points, 3)
def get_3d_points_in_world_space(frame, depth_image, lut_projection, clamp_max, mat_c_to_w):
    frame = frame.astype(np.float64)/255.
    rgb_points = frame.reshape((-1, 3)) # shape (height*width, 3)

    # print(f"lut_projection.shape {lut_projection.shape}")
    points, remove_ids = get_3d_points_in_camera_space(depth_image, lut_projection, clamp_max) # shape (height*width - removed points, 3)
    # print(points.shape)
    homogeneous_points = vec3_list_to_vec4_list(points) # shape (height*width - removed points, 4)

    # mat_c_to_w = get_mat_c_to_w_series(series)
    homogeneous_points = mul_mat44_vec4_list(mat_c_to_w, homogeneous_points) # shape (height*width - removed points, 4)
    # print(homogeneous_points.shape)

    rgb_points = np.delete(rgb_points, remove_ids, axis=0)

    return homogeneous_points[:,:3], rgb_points

# lut_projection shape ((height/skip_factor + 1)*(width/skip_factor + 1), 3)
# output shape ((height/skip_factor + 1)*(width/skip_factor + 1), 3)
def get_camera_pixels_in_world_space(lut_projection, scale, mat_c_to_w):
    points = scale*lut_projection # shape ((height/skip_factor + 1)*(width/skip_factor + 1), 3)
    homogeneous_points = vec3_list_to_vec4_list(points) # shape ((height/skip_factor + 1)*(width/skip_factor + 1), 4)
    homogeneous_points = mul_mat44_vec4_list(mat_c_to_w, homogeneous_points) # shape ((height/skip_factor + 1)*(width/skip_factor + 1), 4)
    return homogeneous_points[:,:3] # shape ((height/skip_factor + 1)*(width/skip_factor + 1), 3)

def get_mat_w_to_o(qr_code_probe_series, optical_probe_series, mat_qf_to_m):
    if optical_probe_series is not None:
        # print("probe")
        mat_m_to_o = get_mat_m_to_o_series(optical_probe_series)
    else:
        mat_m_to_o = np.identity(4)

    mat_qf_to_w = get_mat_q_to_w_series(qr_code_probe_series, qr_code_id=0) # front
    mat_w_to_qf = np.linalg.inv(mat_qf_to_w)
    mat_qf_to_o = np.matmul(mat_m_to_o, mat_qf_to_m)
    mat_w_to_o = np.matmul(mat_qf_to_o, mat_w_to_qf)

    return mat_w_to_o, mat_qf_to_o

# markers is pandas.core.series.Series
def get_df_translation_rotation(markers, optical_marker_id=None):
    translation_vec = []
    if optical_marker_id is None:
        if 'tx 1' in markers:
            translation_vec.append(float(markers['tx 1']))
            translation_vec.append(float(markers['ty 1']))
            translation_vec.append(float(markers['tz 1']))
        elif 'tx 2' in markers:
            translation_vec.append(float(markers['tx 2']))
            translation_vec.append(float(markers['ty 2']))
            translation_vec.append(float(markers['tz 2']))
        else:
            translation_vec.append(float(markers['tx 3']))
            translation_vec.append(float(markers['ty 3']))
            translation_vec.append(float(markers['tz 3']))
    else:
        translation_vec.append(float(markers[f"tx {optical_marker_id}"]))
        translation_vec.append(float(markers[f"ty {optical_marker_id}"]))
        translation_vec.append(float(markers[f"tz {optical_marker_id}"]))
    translation_vec.append(1.)
    # print(translation_vec)

    rotation_vec = []
    if optical_marker_id is None:
        if 'qx 1' in markers:
            rotation_vec.append(float(markers['qx 1']))
            rotation_vec.append(float(markers['qy 1']))
            rotation_vec.append(float(markers['qz 1']))
            rotation_vec.append(float(markers['qw 1']))
        elif 'qx 2' in markers:
            rotation_vec.append(float(markers['qx 2']))
            rotation_vec.append(float(markers['qy 2']))
            rotation_vec.append(float(markers['qz 2']))
            rotation_vec.append(float(markers['qw 2']))
        else:
            rotation_vec.append(float(markers['qx 3']))
            rotation_vec.append(float(markers['qy 3']))
            rotation_vec.append(float(markers['qz 3']))
            rotation_vec.append(float(markers['qw 3']))
    else:
        rotation_vec.append(float(markers[f"qx {optical_marker_id}"]))
        rotation_vec.append(float(markers[f"qy {optical_marker_id}"]))
        rotation_vec.append(float(markers[f"qz {optical_marker_id}"]))
        rotation_vec.append(float(markers[f"qw {optical_marker_id}"]))
    # print(rotation_vec)
    return translation_vec, rotation_vec

def get_mat_m_to_o_series(markers, optical_marker_id=None):
    translation_vec, rotation_vec = get_df_translation_rotation(markers, optical_marker_id)
    return get_mat_quaternion(translation_vec, rotation_vec)

def get_mat_o_to_m_series(markers, optical_marker_id=None):
    translation_vec, rotation_vec = get_df_translation_rotation(markers, optical_marker_id)
    return get_inverse_mat_quaternion(translation_vec, rotation_vec)

def get_mat_m_to_o(df_markers, timestamp):
    # if 'Tx' in df_markers:
        # translation_vec.append(float(df_markers['Tx'][timestamp]))
    return get_mat_m_to_o_series(df_markers.loc[timestamp])

def get_pointer_offset_filename(config):
    return config.folder + "pointer_offset.pickle"

def get_mat_divots_filename(config, variable):
    return config.folder + variable + ".pickle"

def pointer_pivot_calibration(df, config, average_with_previous_calibration):
    offset_tip_pointer = pivot_calibration(df, get_mat_m_to_o_series)

    if average_with_previous_calibration:
        offset_tip_pointer2 = load_pickle(get_pointer_offset_filename(config))
        offset_tip_pointer = (offset_tip_pointer + offset_tip_pointer2)/2

    save_pickle(offset_tip_pointer, get_pointer_offset_filename(config))

    log_print(f"offset_tip_pointer {offset_tip_pointer}")

    return offset_tip_pointer

def register_divots(df, config, offset_tip_pointer, register_divots_fct, variable, mat_qf_to_m=None, inverse=True, order_divot_index=None, to_delete_gt_divot_index=None):
    log_print("register_divots")
    # dpg.lock_mutex()
    divots_pts, cloud_pts, mat_marker_to_reference = register_divots_fct(df, vec3_to_vec4(offset_tip_pointer), get_mat_m_to_o_series, get_mat_o_to_m_series, order_divot_index, to_delete_gt_divot_index)
    # dpg.unlock_mutex()

    mat = mat_marker_to_reference
    if inverse:
        mat = np.linalg.inv(mat) # mat_reference_to_marker
    elif mat_qf_to_m is not None:
        mat = np.matmul(mat, mat_qf_to_m) # mat_qf_to_reference

    save_pickle(mat, get_mat_divots_filename(config, variable))
    print(f"{variable} {mat}")

    return mat, divots_pts, cloud_pts
