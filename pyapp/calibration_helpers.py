import time
import numpy as np
import scipy
import scipy.optimize # fsolve, root, least_squares

from Logging import log_print
from UtilMaths import get_mat_camera_to_world, get_mat_quaternion, get_inverse_mat_quaternion, mul_mat44_vec4_list, vec3_list_to_vec4_list

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
def get_lut_projection_pixel(lut, x, y):
    # width = (lut[0].shape[1] - 1)//2
    # index = (y*2 + 1)*lut[0].shape[1] + (x*2 + 1) # we take the value in the middle of the pixel (x + 0.5, y + 0.5)
    # return [lut[0][index], lut[1][index], 1]
    return [lut[0][int(y*2 + 1), int(x*2 + 1)], lut[1][int(y*2 + 1), int(x*2 + 1)], 1]

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

    # mat_c_to_w = np.matmul(mat_c_to_w, rotation_euler_matrix44((0,0,180), degrees=True))

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

# vars type np.float64
# x_array shape(_,5)
# y_array shape(_,5)
# z_array shape(_,5)
# d2_array shape(_,)
def seed_tip_equations(x_array,y_array,z_array,d2_array, vars):
    sx, sy, sz, tx, ty, tz = vars # seed position in W CS (sx, sy, sz) and tip position in Q CS (tx, ty, tz)

    x_array = np.matmul(x_array,[sx, tx, ty, tz, 1]) # shape(_,)
    y_array = np.matmul(y_array,[sy, tx, ty, tz, 1]) # shape(_,)
    z_array = np.matmul(z_array,[sz, tx, ty, tz, 1]) # shape(_,)

    return x_array*x_array + y_array*y_array + z_array*z_array - d2_array # shape(_,)

# vars type np.float64
# x_array shape(_,)
# y_array shape(_,)
# z_array shape(_,)
# d2_array shape(_,)
def seed_equations(x_array,y_array,z_array,d2_array, vars):
    sx, sy, sz = vars # seed position in W CS (sx, sy, sz)

    x_array = sx + x_array # shape(_,)
    y_array = sy + y_array # shape(_,)
    z_array = sz + z_array # shape(_,)

    return x_array*x_array + y_array*y_array + z_array*z_array - d2_array # shape(_,)

def solve_equations(df_magnetic_seed, df_probe_position, calibration_starting_time, calibration_ending_time, get_mat=get_mat_m_to_o, init_seed=(0,0,0), init_tip=(0,0,0), tip_unknown=True):
    starting_time = time.time()

    probe_starting_index = df_probe_position.index.get_loc(calibration_starting_time, method='bfill') # ffill, next or equal
    probe_ending_index = df_probe_position.index.get_loc(calibration_ending_time, method='ffill') # bfill, previous or equal

    x_array = []
    y_array = []
    z_array = []
    d2_array = []
    timestamps = []
    for i in range(probe_starting_index, probe_ending_index + 1):
        timestamp = df_probe_position.index[i]

        previous_index = df_magnetic_seed.index.get_loc(timestamp, method='ffill')
        previous_timestamp = df_magnetic_seed.index[previous_index]
        next_index = df_magnetic_seed.index.get_loc(timestamp, method='bfill')
        next_timestamp = df_magnetic_seed.index[next_index]
        # closest_index = df_magnetic_seed.index.get_loc(timestamp, method='nearest')
        # closest_timestamp = df_magnetic_seed.index[closest_index]

        if previous_timestamp < timestamp:
            previous_diff_time = timestamp - previous_timestamp # timedelta
        else:
            previous_diff_time = previous_timestamp - timestamp # timedelta

        if next_timestamp < timestamp:
            next_diff_time = timestamp - next_timestamp # timedelta
        else:
            next_diff_time = next_timestamp - timestamp # timedelta

        if previous_index == next_index or (previous_diff_time.microseconds < 25000 and next_diff_time.microseconds < 25000): # magnetic seed tracking frequency should be around ~40-50Hz -> every min ~25000 us, we don't want to have measurements too old/early
            timestamps.append(timestamp)

            m = get_mat(df_probe_position, timestamp)

            if previous_index == next_index: # timestamp perfect match:
                d = df_magnetic_seed.loc[previous_timestamp][0] # milimeter
            else:
                coef = (timestamp - previous_timestamp)/(next_timestamp - previous_timestamp)
                d = (1-coef)*df_magnetic_seed.loc[previous_timestamp][0] + coef*df_magnetic_seed.loc[next_timestamp][0] # milimeter

            # equation = (sx - tx*m[0][0] - ty*m[0][1] - tz*m[0][2] - m[0][3])**2 \
            #      + (sy - tx*m[1][0] - ty*m[1][1] - tz*m[1][2] - m[1][3])**2 \
            #      + (sz - tx*m[2][0] - ty*m[2][1] - tz*m[2][2] - m[2][3])**2 \
            #      - d**2

            if tip_unknown:
                x_array.append([1, -m[0][0], -m[0][1], -m[0][2], -m[0][3]])
                y_array.append([1, -m[1][0], -m[1][1], -m[1][2], -m[1][3]])
                z_array.append([1, -m[2][0], -m[2][1], -m[2][2], -m[2][3]])
            else:
                tx, ty, tz = init_tip
                x_array.append(-tx*m[0][0] - ty*m[0][1] - tz*m[0][2] - m[0][3])
                y_array.append(-tx*m[1][0] - ty*m[1][1] - tz*m[1][2] - m[1][3])
                z_array.append(-tx*m[2][0] - ty*m[2][1] - tz*m[2][2] - m[2][3])

            d2_array.append(d**2)

    x_array = np.array(x_array) # x_array shape(_,5) or x_array shape(_,)
    y_array = np.array(y_array) # y_array shape(_,5) or x_array shape(_,)
    z_array = np.array(z_array) # z_array shape(_,5) or x_array shape(_,)
    d2_array = np.array(d2_array) # d2_array shape(_,)

    elasped_time = time.time() - starting_time
    log_print(f"elasped_time {elasped_time:.1f} s, probe_start_id {probe_starting_index} probe_end_id {probe_ending_index} {probe_ending_index - probe_starting_index + 1} x_array.shape {x_array.shape} y_array.shape {y_array.shape} z_array.shape {z_array.shape} d2_array.shape {d2_array.shape}")

    starting_time = time.time()
    if tip_unknown:
        ret = scipy.optimize.least_squares(lambda vars:seed_tip_equations(x_array,y_array,z_array,d2_array, vars), list(init_seed) + list(init_tip)) #,method='lm',ftol=1e-15,xtol=1e-15,gtol=1e-15)
    else:
        ret = scipy.optimize.least_squares(lambda vars:seed_equations(x_array,y_array,z_array,d2_array, vars), init_seed) #,ftol=1e-15,xtol=1e-15,gtol=1e-15)
    elasped_time = time.time() - starting_time
    log_print(f"least_squares elasped_time {elasped_time:.1f} s")
    log_print(f"seed pos {ret['x'][:3]}")
    if tip_unknown:
        log_print(f"probe tip {ret['x'][3:]}")

    print(ret)

    # check/debug test
    sx, sy, sz = ret['x'][:3] # seed position in W CS (sx, sy, sz)
    if tip_unknown:
        tx, ty, tz = ret['x'][3:] # tip position in Q CS (tx, ty, tz)
        x_array = np.matmul(x_array,[sx, tx, ty, tz, 1]) # shape(_,)
        y_array = np.matmul(y_array,[sy, tx, ty, tz, 1]) # shape(_,)
        z_array = np.matmul(z_array,[sz, tx, ty, tz, 1]) # shape(_,)
    else:
        x_array = sx + x_array # shape(_,)
        y_array = sy + y_array # shape(_,)
        z_array = sz + z_array # shape(_,)
    residuals2 = x_array*x_array + y_array*y_array + z_array*z_array - d2_array # shape(_,)
    print(f"check sum substraction residuals {np.sum(np.array(ret['fun']) - residuals2)} (should be 0.0)")
    # print(repr(residuals2))
    residuals_distances = np.sqrt(x_array*x_array + y_array*y_array + z_array*z_array) - np.sqrt(d2_array) # shape(_,)
    # print(repr(residuals_distances))
    residuals_distances = np.abs(residuals_distances)
    log_print(f"absolute residuals mean {np.mean(residuals_distances):.2f} median {np.median(residuals_distances):.2f} min {np.min(residuals_distances):.2f} max {np.max(residuals_distances):.2f}")
    log_print(f"percentile 25th {np.percentile(residuals_distances, 25):.2f} 75th {np.percentile(residuals_distances, 75):.2f} 95th {np.percentile(residuals_distances, 95):.2f} 99th {np.percentile(residuals_distances, 99):.2f}")

    if tip_unknown:
        return ret['x'][:3], ret['x'][3:], residuals_distances, timestamps
    return ret['x'][:3], list(init_tip), residuals_distances, timestamps
