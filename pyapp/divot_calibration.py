import datetime
import numpy as np
from scipy.signal import find_peaks

import pivot # https://github.com/UCL/scikit-surgerycalibration

# from File import save_pickle, load_pickle # debug
from UtilMaths import mul_mat44_vec4_list, vec3_list_to_vec4_list, mul_mat44_vec4, translation_matrix44, rotation_euler_matrix44, identity_matrix44, vec3_to_vec4, point_based_registration, iterative_closest_point #, print_matrix44
from Logging import log_print
from calibration_helpers import get_mat_o_to_m_series, get_mat_m_to_o_series

SLIDING_WINDOW = '4s' # in s
MIN_WINDOW_SIZE = 20 # in nb of measurement, 20 is minimum ~1s (optical tracking is 20Hz)
MIN_DISTANCE_BETWEEN_PEAK = 20*5 # in nb of measurement, 20*4 is minimum ~4s
MAX_SQUARED_DISTANCE_MEAN = 0.5*0.5 # in mm^2

# DIVOT_DEPTH = 0.9 # Real one, but we have to take care of the pointer tip shape/size/calibration
DIVOT_DEPTH = 0.3 # in mm # get max_error_dist between 0.15 and 0.25mm for qr code and 0.51-0.67mm for seed holder
# DIVOT_DEPTH = -0.425 # in mm # seems to give the best registration results so far

CLOUD_INSTANCE = 0
CLOUD_TRANSFORMATION = 1

class Cloud:
    def __init__(self):
        self.pts = []

    def get_points(self):
        return np.array(self.pts)

    # def update_transform(self):

    # def match(self, cloud):

class CloudGroup(Cloud):
    def __init__(self):
        super().__init__()
        self.clouds = {}

    def get_points(self):
        pts = []
        for _, cloud in self.clouds.items():
            transformed_pts = []
            for pt in cloud[CLOUD_INSTANCE].pts:
                transformed_pts.append(mul_mat44_vec4(cloud[CLOUD_TRANSFORMATION], vec3_to_vec4(pt))[:3])
            pts.extend(transformed_pts)
        return np.array(pts)

    # def update_transform(self):
        # for key, cloud in self.clouds.items():
            # cloud.update_transform()

    # def match(self, cloud):

class QRCodeCloud(Cloud):
    def __init__(self, nb_rows, nb_columns, pixel_size):
        super().__init__()

        for j in range(nb_rows):
            for i in range(nb_columns):
                x = -(nb_columns - 1)*pixel_size/2 + i*pixel_size
                y = (nb_rows - 1)*pixel_size/2 - j*pixel_size
                z = -DIVOT_DEPTH
                self.pts.append((x,y,z))

class CubeQRCodeCloud(CloudGroup):
    def __init__(self, nb_rows, nb_columns, pixel_size):
        super().__init__()

        white_border = 10 # in mm
        qr_code_thickness = 6 # in mm

        self.clouds["front"] = (QRCodeCloud(nb_rows, nb_columns, pixel_size), identity_matrix44())

        rotation = rotation_euler_matrix44((0,90,0), degrees=True)
        translation = translation_matrix44((-(nb_columns - 1)*pixel_size/2 - white_border, 0, -(nb_columns - 1)*pixel_size/2 - white_border - qr_code_thickness, 1))
        mat_q_to_origin = np.matmul(translation, rotation)
        self.clouds["left"] = (QRCodeCloud(nb_rows, nb_columns, pixel_size), mat_q_to_origin)

        rotation = rotation_euler_matrix44((0,-90,0), degrees=True)
        translation = translation_matrix44(((nb_columns - 1)*pixel_size/2 + white_border, 0, -(nb_columns - 1)*pixel_size/2 - white_border - qr_code_thickness, 1))
        mat_q_to_origin = np.matmul(translation, rotation)
        self.clouds["right"] = (QRCodeCloud(nb_rows, nb_columns, pixel_size), mat_q_to_origin)

        rotation = rotation_euler_matrix44((-90,0,0), degrees=True)
        translation = translation_matrix44((0, (nb_columns - 1)*pixel_size/2 + white_border + qr_code_thickness, -(nb_columns - 1)*pixel_size/2 - white_border - qr_code_thickness, 1))
        mat_q_to_origin = np.matmul(translation, rotation)
        self.clouds["top"] = (QRCodeCloud(nb_rows, nb_columns, pixel_size), mat_q_to_origin)

class SeedHolderCloud(Cloud):
    def __init__(self):
        super().__init__()

        seed_length = 5.25 # in mm, diameter 1.68 mm but 1.72 mm in the middle

        # x axis -> longitudinal seed, direction from the seed to the hole's exit
        # y axis -> width (30 mm) seed holder
        # z axis -> height (10 mm) seed holder, direction from the seed to the 5x3 divots plane
        z = 2 - 5
        border_x = 6 - seed_length/2
        for i in range(5):
            x = -4*10 -3 + border_x + i*10
            for j in range(3):
                self.pts.append((x, 10 - j*10, 2 - DIVOT_DEPTH))

            self.pts.append((x,  15 - DIVOT_DEPTH, z))
            self.pts.append((x, -15 + DIVOT_DEPTH, z))

        self.pts.append((border_x - DIVOT_DEPTH,  10, z))
        self.pts.append((border_x - DIVOT_DEPTH,   0, z))
        self.pts.append((border_x - DIVOT_DEPTH, -10, z))

def pivot_calibration(df_pointer):
    # df_pointer = df_pointer[df_pointer['status 2'] == 'Enabled']
    log_print(f"df_pointer.index.size {df_pointer.index.size}")

    positions = []
    for _, row in df_pointer.iterrows():
        #print(index)
        #print(type(row))
        positions.append(get_mat_m_to_o_series(row, '2'))
    positions = np.array(positions)

    offset_tip_pointer, _ , residual_error = pivot.pivot_calibration(positions)
    log_print(f"residual_error {residual_error}")
    return offset_tip_pointer

def find_divots(df_optical, offset_tip_pointer, optical_marker_id):
    df = df_optical[(df_optical['status 2'] == 'Enabled') & (df_optical[f"status {optical_marker_id}"] == 'Enabled')].copy()
    print(f"df.index.size {df.index.size}")

    # compute pointer position in optical markers coordinate system
    start_time = datetime.datetime.now()
    for i in range(df.index.size):
    #for i in range(1):
        serie = df.loc[df.index[i]]

        mat_m_to_o = get_mat_m_to_o_series(serie, optical_marker_id=2)
        pos_pointer_in_o = mul_mat44_vec4(mat_m_to_o, offset_tip_pointer)
        df.loc[df.index[i],"tx 2"] = pos_pointer_in_o[0]
        df.loc[df.index[i],"ty 2"] = pos_pointer_in_o[1]
        df.loc[df.index[i],"tz 2"] = pos_pointer_in_o[2]

        # mat_m_to_o = get_mat_m_to_o_series(serie, optical_marker_id=optical_marker_id)
        #print_matrix44(mat_m_to_o)
        # mat_o_to_m = np.linalg.inv(mat_m_to_o)
        #print_matrix44(mat_o_to_m)
        mat_o_to_m = get_mat_o_to_m_series(serie, optical_marker_id=optical_marker_id)
        #print_matrix44(mat_o_to_m)

        # pos_pointer_in_o = (serie["tx 2"], serie["ty 2"], serie["tz 2"], 1)
        pos_pointer_in_m = mul_mat44_vec4(mat_o_to_m, pos_pointer_in_o)
        df.loc[df.index[i],"tx 2"] = pos_pointer_in_m[0]
        df.loc[df.index[i],"ty 2"] = pos_pointer_in_m[1]
        df.loc[df.index[i],"tz 2"] = pos_pointer_in_m[2]
        # print(pos_pointer_in_m)

    print(df.loc[df.index[10],"tx 2"])

    df_shifted = df.shift(1) # shift forward (first entry will be NaN)
    squared_distance = (df['tx 2'] - df_shifted['tx 2'])**2 \
        + (df['ty 2'] - df_shifted['ty 2'])**2 \
        + (df['tz 2'] - df_shifted['tz 2'])**2

    squared_distance = squared_distance.rolling(SLIDING_WINDOW, min_periods=MIN_WINDOW_SIZE).mean() # first entries will be NaN (until SLIDING_WINDOW seconds)

    peaks, _ = find_peaks(-squared_distance, distance=MIN_DISTANCE_BETWEEN_PEAK)
    peaks = [peak for peak in peaks if squared_distance[peak] < MAX_SQUARED_DISTANCE_MEAN]
    log_print(f"peaks found: {len(peaks)}")

    x_mean = df['tx 2'].rolling(SLIDING_WINDOW, min_periods=MIN_WINDOW_SIZE).mean()
    x_mean = x_mean.loc[squared_distance.index[peaks]]
    # print(x_mean)
    y_mean = df['ty 2'].rolling(SLIDING_WINDOW, min_periods=MIN_WINDOW_SIZE).mean()
    y_mean = y_mean.loc[squared_distance.index[peaks]]
    z_mean = df['tz 2'].rolling(SLIDING_WINDOW, min_periods=MIN_WINDOW_SIZE).mean()
    z_mean = z_mean.loc[squared_distance.index[peaks]]

    # print(type(x_mean[0]))
    # print(type(x_mean.to_numpy()[0]))
    divots = np.stack((x_mean.to_numpy(), y_mean.to_numpy(), z_mean.to_numpy())).transpose()
    # divots = np.stack((x_mean, y_mean, z_mean)).transpose()

    log_print(f"{datetime.datetime.now() - start_time}")

    return divots

# order_divot_index.shape (_,)
# to_delete_gt_divot_index.shape (_,)
def register_divots(df_optical, offset_tip_pointer, optical_marker_id, cloud, order_divot_index=None, to_delete_gt_divot_index=None):
    # log_print("register_divots")
    # divots = load_pickle("C:/Users/dibule/Desktop/divots_cube.pickle")
    divots = find_divots(df_optical, offset_tip_pointer, optical_marker_id=optical_marker_id) # shape (n,3)
    # save_pickle(divots, "C:/Users/dibule/Desktop/divots_cube.pickle")
    print(f"divots.shape {divots.shape}")

    gt_divots = cloud.get_points() # shape (m,3)
    print(f"gt_divots.shape {gt_divots.shape}")

    divots_modified = divots
    gt_divots_modified = gt_divots
    # change order points if needed and remove some in the ground truth (case by case, manual step)
    if order_divot_index is not None:
        divots_modified = divots[order_divot_index]
    if to_delete_gt_divot_index is not None:
        gt_divots_modified = np.delete(gt_divots, to_delete_gt_divot_index, axis=0)
    # print(divots_modified)
    # print(gt_divots_modified)

    mat_marker_to_reference, rmse, mean, max_dist = point_based_registration(divots_modified, gt_divots_modified)
    log_print(f"pbr rmse {rmse} mean {mean} max_dist {max_dist}")
    # mat_marker_to_reference = np.identity(4)

    mat_marker_to_reference, mean, max_dist = iterative_closest_point(divots, gt_divots, 100, mat_marker_to_reference)
    log_print(f"icp mean {mean} max_dist {max_dist}")

    divots = mul_mat44_vec4_list(mat_marker_to_reference, vec3_list_to_vec4_list(divots))
    divots = divots[:,:3]

    # return divots, gt_divots, mat_marker_to_reference
    return divots, gt_divots_modified, mat_marker_to_reference

def register_divots_front_qr_code(df_optical, offset_tip_pointer, order_divot_index=None, to_delete_gt_divot_index=None):
    cloud = QRCodeCloud(7,7,10)
    return register_divots(df_optical, offset_tip_pointer, 1, cloud, order_divot_index, to_delete_gt_divot_index)

def register_divots_left_qr_code(df_optical, offset_tip_pointer, order_divot_index=None, to_delete_gt_divot_index=None):
    cloud = QRCodeCloud(7,7,10)
    return register_divots(df_optical, offset_tip_pointer, 1, cloud, order_divot_index, to_delete_gt_divot_index)

def register_divots_right_qr_code(df_optical, offset_tip_pointer, order_divot_index=None, to_delete_gt_divot_index=None):
    cloud = QRCodeCloud(7,7,10)
    return register_divots(df_optical, offset_tip_pointer, 1, cloud, order_divot_index, to_delete_gt_divot_index)

def register_divots_top_qr_code(df_optical, offset_tip_pointer, order_divot_index=None, to_delete_gt_divot_index=None):
    cloud = QRCodeCloud(7,7,10)
    return register_divots(df_optical, offset_tip_pointer, 1, cloud, order_divot_index, to_delete_gt_divot_index)

# def register_divots_cube_qr_code(df_optical, offset_tip_pointer, order_divot_index=None, to_delete_gt_divot_index=None):
    # cloud = CubeQRCodeCloud(7,7,10)
    # return register_divots(df_optical, offset_tip_pointer, 1, cloud, order_divot_index, to_delete_gt_divot_index)

def register_divots_seed_holder(df_optical, offset_tip_pointer, order_divot_index=None, to_delete_gt_divot_index=None):
    cloud = SeedHolderCloud()
    return register_divots(df_optical, offset_tip_pointer, 3, cloud, order_divot_index, to_delete_gt_divot_index)
