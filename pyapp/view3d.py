import time
import copy
import threading
import numpy as np

import open3d as o3d

from python.common.UtilMaths import vec3_to_vec4, identity_matrix44, translation_matrix44, mul_mat44_vec4
from python.common.UtilImage import convert_gray_to_hsv_mapping

from python.common.UtilOpen3d import create_coordinate_system, create_qr_code, create_probe, create_optical_pointer_8700340,  create_optical_markers_8700449, create_optical_markers_8700339, create_optical_tracker, create_sphere, create_camera, create_projected_lines, compute_projected_points, compute_camera_points, create_camera_frustum, MeshGroup, create_spheres, OPTICAL_SPHERE_RADIUS, create_seed_holder, RESET_BOUNDING_BOX, SCALE_UNIT, get_out_position, get_invisible_transformation, create_camera_frustum_from_lut_projection, INVISIBLE_SCALE
from calibration_helpers import get_mat_c_to_w_series, get_mat_m_to_o_series, get_mat_q_to_w_series, get_mat_w_to_o, get_lut_projection_pixel_mapping, get_3d_points_in_world_space, get_lut_projection_camera_mapping, get_camera_pixels_in_world_space
from DataAcquisition import QR_CODE_NAMES, CAMERA_NAMES, ACQUISITIONS_HOLOLENS, HOLOLENS_METER_TO_MILIMETER_SCALE, MAX_AHAT_AB, MAX_LT_AB, MAX_AHAT_DEPTH, MAX_LT_DEPTH, RESEARCH_MODE_CAMERA_NAMES

# MARKER_LENGTH = 0.06 # in m # size printed QR code 6cm
MARKER_LENGTH = 60*SCALE_UNIT # in mm # size printed QR code 6cm
# MARKER_LENGTH = 6*5*0.0016 # in m # lego QR code, lego unit 1.6mm, square base is 5 lego unit, 6 squares
# MARKER_LENGTH = 6*5*1.6 # in mm # lego QR code, lego unit 1.6mm, square base is 5 lego unit, 6 squares

DRAW_CS_LENGTH = 50*SCALE_UNIT # 100 # in mm

DIVOT_SPHERE_RADIUS = 3*SCALE_UNIT # in mm

# LUT_SKIP_FACTOR = {'vl_front_left_cam':2, 'vl_front_right_cam':2, 'ahat_depth_cam':2, 'lt_depth_cam':2}
# LUT_SKIP_FACTOR = {'vl_front_left_cam':160, 'vl_front_right_cam':160, 'ahat_depth_cam':64, 'lt_depth_cam':32}
LUT_SKIP_FACTOR = {'vl_front_left_cam':40, 'vl_front_right_cam':40, 'ahat_depth_cam':16, 'lt_depth_cam':8}

WINDOW_3D_WIDTH = 840
WINDOW_3D_HEIGHT = 900

VISU_SLEEP_TIME = 0.05 # in s

class View3D:
    def __init__(self, app):
        self.app = app
        self.visualization = None
        self.visualization_thread = None

        self.objects = {}

    def run(self):
        self.visualization_thread = threading.Thread(target=self.visualization_routine)
        self.visualization_thread.daemon = True
        self.visualization_thread.start()

    def stop(self):
        self.visualization.destroy_window()
        # should send stop to self.visualization_thread

    def visualization_routine(self):
        self.visualization = o3d.visualization.Visualizer()
        self.visualization.create_window(width=WINDOW_3D_WIDTH,height=WINDOW_3D_HEIGHT,left=842,top=60)

        self.create_frame()

        view_control = self.visualization.get_view_control()
        if not self.app.acquisitions["optical"].empty:
            # view_control.set_front([0,0,1])
            view_control.set_lookat([0,0,-1900])
            view_control.set_up([-1,0,0])
            view_control.set_constant_z_near(1.)
            view_control.set_constant_z_far(4000.)
        else:
            if False: # convert_to_pinhole_camera_parameters seems bugged, wait for fix..., don't forget to put back RESET_BOUNDING_BOX = False
            # if True:
                # param = {"class_name" : "PinholeCameraParameters"
                #         ,"extrinsic" : [-1.0,0.0,0.0,0.0
                #                         ,0.0,1.0,0.0,0.0
                #                         ,0.0,0.0,1.0,0.0
                #                         ,0.0,0.0,-1000,1.0],
                #         "intrinsic" : {"height" : WINDOW_3D_HEIGHT
                #                     ,"intrinsic_matrix" : [779.4228634059948,0.0,0.0
                #                                           ,0.0,779.4228634059948,0.0
                #                                           ,419.5,449.5,1.0]
                #                     ,"width" : WINDOW_3D_WIDTH}
                #     ,"version_major" : 1
                #     ,"version_minor" : 0}
                param = view_control.convert_to_pinhole_camera_parameters()
                param =  o3d.camera.PinholeCameraParameters(param)
                print(param.extrinsic)
                print(param.extrinsic.shape)
                print(param.extrinsic.dtype)
                print(param.intrinsic)
                print(param.intrinsic.intrinsic_matrix)
                # param = copy.copy(param)
                # 180 rotation around y, to obtain typical camera (+x,-y,+z)
                # param.extrinsic = np.array([[-1.0,0.0,0.0,0.0]
                                            # ,[0.0,1.0,0.0,0.0]
                                            # ,[0.0,0.0,-1.0,0.0]
                                            # ,[0.0,0.0,0,1.0]])
                param.intrinsic.set_intrinsics(WINDOW_3D_WIDTH, WINDOW_3D_HEIGHT, 600, 600, 419.5, 449.5)
                # param.intrinsic.intrinsic_matrix = np.array([[600,0,419.5]
                                                            # ,[0,600,449.5]
                                                            # ,[0,0,1.]])
                view_control.convert_from_pinhole_camera_parameters(param, allow_arbitrary=True)
                param = view_control.convert_to_pinhole_camera_parameters()
                print(param.extrinsic)
                print(param.extrinsic.shape)
                print(param.extrinsic.dtype)
                print(param.intrinsic)
                print(param.intrinsic.intrinsic_matrix)
            else:
                view_control.set_lookat([0,0,-1])
                view_control.set_up([0,1,0])
                # view_control.camera_local_translate(0,0,-1000)

        while True:
            time.sleep(VISU_SLEEP_TIME)

            self.app.mutex_update.acquire()
            has_to_update_tmp = copy.copy(self.app.has_to_update)
            self.app.has_to_update = False
            self.app.mutex_update.release()

            if has_to_update_tmp:
                self.update_frame()

            self.visualization.poll_events()
            self.visualization.update_renderer()

    # objects is a dict
    def add_objects(self, objects):
        self.objects = {**self.objects, **objects}

    def add_geometry(self, reset_bounding_box=RESET_BOUNDING_BOX):
        for _, item in self.objects.items():
            item.add_geometry(self.visualization, reset_bounding_box)

    def clear_objects(self):
        self.objects = {}
        self.visualization.clear_geometry()

    def create_frame(self):
        # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=np.array([0.0, 0.0, 0.0]))
        # self.visualization.add_geometry(mesh)

        # HACK create dummy box area for the camera initialization (didn't find better way for now)
        pts = np.array([[-1,-1,-1],[1,-1,-1],[1,1,-1],[-1,1,-1] \
                        ,[-1,-1,1],[1,-1,1],[1,1,1],[-1,1,1]],dtype=np.float64)
        pts *= HOLOLENS_METER_TO_MILIMETER_SCALE/2
        vertices = o3d.utility.Vector3dVector(pts)
        lines = o3d.utility.Vector2iVector(np.array(((0,1),(0,3),(0,4),(1,2),(1,5),(2,3),(2,6),(3,7),(4,5),(4,7),(5,6),(6,7))))
        line_set = o3d.geometry.LineSet(vertices, lines)
        line_set.colors = o3d.utility.Vector3dVector(np.tile(((1,0,1),), (12,1)))
        self.visualization.add_geometry(line_set)

        cs = create_coordinate_system(identity_matrix44(),length=DRAW_CS_LENGTH)
        self.add_objects({"cs_origin":cs})

        if not self.app.acquisitions["optical"].empty:
            cs = create_optical_tracker()
            self.add_objects({"optical_tracker":cs})

            cs = create_optical_pointer_8700340(get_invisible_transformation(), self.app.offset_tip_pointer, length=DRAW_CS_LENGTH)
            self.add_objects({"pointer":cs})

            cs = create_optical_markers_8700449(get_invisible_transformation(),length=DRAW_CS_LENGTH)
            self.add_objects({"optical_markers_seed":cs})

            cs = create_optical_markers_8700339(get_invisible_transformation(),length=DRAW_CS_LENGTH)
            self.add_objects({"optical_markers":cs})

            cs = create_seed_holder(get_invisible_transformation(),length=DRAW_CS_LENGTH/2)
            self.add_objects({"optical_seed_holder":cs})

            cs = create_probe(translation_matrix44(vec3_to_vec4(self.app.pos_t_m)), tip_color = (0,1,0), probe_color = (0.97,0.89,0.71), rotation=(-90,0,0), translation=(0,-220,0)) #PROBE_TIP_OPTICAL_DEFAULT)
            self.add_objects({"optical_probe":cs})

            for qr_code_name in QR_CODE_NAMES:
                cs = create_qr_code(get_invisible_transformation(), length=MARKER_LENGTH/2, qr_code_length=MARKER_LENGTH, add_coordinate_system=False)
                self.add_objects({"optical_" + qr_code_name:cs})

            cs = MeshGroup()
            cs.add_mesh(create_sphere([0,0,0], [0,0,1], OPTICAL_SPHERE_RADIUS/2))
            # cs.transform(translation_matrix44(vec3_to_vec4(self.app.pos_s_o)), internal=False, revert=False) # I think it should be revert=True here
            self.add_objects({"optical_seed":cs})

            self.add_objects({"registration_divots_pts":MeshGroup()})
            self.add_objects({"registration_cloud_pts":MeshGroup()})

        if not self.app.acquisitions["qr_code_position"].empty:
            for qr_code_name in QR_CODE_NAMES:
                cs = create_qr_code(get_invisible_transformation(), length=MARKER_LENGTH/2, qr_code_length=MARKER_LENGTH)
                self.add_objects({qr_code_name:cs})

            cs = create_projected_lines(get_invisible_transformation(), get_invisible_transformation(), get_invisible_transformation(), get_invisible_transformation(), get_invisible_transformation(), MARKER_LENGTH, get_out_position(), get_out_position(), get_out_position(), get_out_position())
            self.add_objects({"projected_lines":cs})

            if not self.app.acquisitions["magnetic_seed"].empty:
                # cs = create_probe(translation_matrix44(vec3_to_vec4(self.app.pos_t_q)), (1,0.5,0), (0.72, 0.47, 0.34), rotation=(90,0,0), translation=(0,220,0))
                cs = create_probe(get_invisible_transformation(), (1,0.5,0), (0.72, 0.47, 0.34), rotation=(90,0,0), translation=(0,220,0))
                self.add_objects({"qr_probe":cs})

                cs = MeshGroup()
                cs.add_mesh(create_sphere([0,0,0], [0.5,0,1], OPTICAL_SPHERE_RADIUS/2))
                # cs.transform(translation_matrix44(vec3_to_vec4(self.app.pos_s_w)), internal=False, revert=False)
                # cs.transform(get_invisible_transformation(), internal=False, revert=False) # I think it should be revert=True here
                self.add_objects({"qr_seed":cs})

        for camera_name in CAMERA_NAMES:
            if not self.app.acquisitions[camera_name].empty:
                cs = create_camera(DRAW_CS_LENGTH, identity_matrix44()) # we put dummy value as update_frame() will do the proper job
                self.add_objects({camera_name:cs})

                if camera_name in RESEARCH_MODE_CAMERA_NAMES:
                # if False:
                    if camera_name in ["ahat_depth_cam", "lt_depth_cam"]:
                        frame = self.app.acquisitions[camera_name + "_ab_frames"][0]
                    else:
                        frame = self.app.acquisitions[camera_name + "_frames"][0]

                    downsampling_factor = self.app.acquisitions[camera_name].loc[self.app.acquisitions[camera_name].index[0]]['downsampling_factor']
                    cs = create_camera_frustum_from_lut_projection(frame.shape[1]*downsampling_factor, frame.shape[0]*downsampling_factor, LUT_SKIP_FACTOR[camera_name])
                else:
                    cs = create_camera_frustum(100, 100, 100, 100, 100, 100, identity_matrix44(), HOLOLENS_METER_TO_MILIMETER_SCALE) # we put dummy value as update_frame() will do the proper job
                self.add_objects({camera_name + "_frustum":cs})

                if camera_name in ["ahat_depth_cam", "lt_depth_cam"] and len(self.app.acquisitions[camera_name + "_frames"]) != 0:
                    cs = MeshGroup()
                    cs.add_mesh(o3d.geometry.PointCloud())
                    self.add_objects({camera_name + "_point_cloud":cs})

        self.add_geometry()

    def update_frame(self):

        pos_s_o = get_out_position()
        pos_gt_s_o = get_out_position()

        self.app.mutex_update.acquire()

        visibility = copy.copy(self.app.visibility)

        if not self.app.acquisitions["optical"].empty:
            optical_slider_value_tmp = copy.copy(self.app.slider_value["optical"])
            timestamp = self.app.df["optical"].index[optical_slider_value_tmp]
            # print(f"value {optical_slider_value_tmp} timestamp {timestamp}")

            pointer_series = None
            if timestamp in self.app.df_pointer.index:
                pointer_series = self.app.df_pointer.loc[timestamp].copy()

            optical_probe_series = None
            if timestamp in self.app.df_probe.index:
                optical_probe_series = self.app.df_probe.loc[timestamp].copy()

            optical_seed_series = None
            if timestamp in self.app.df_optical_seed.index:
                optical_seed_series = self.app.df_optical_seed.loc[timestamp].copy()

            pos_t_m = copy.copy(self.app.pos_t_m)

            mat_qf_to_m = copy.copy(self.app.mat_qf_to_m)
            mat_qf_to_ql = copy.copy(self.app.mat_qf_to_ql)
            mat_qf_to_qr = copy.copy(self.app.mat_qf_to_qr)
            mat_qf_to_qt = copy.copy(self.app.mat_qf_to_qt)
            mat_s_to_m2 = copy.copy(self.app.mat_s_to_m2)

            pos_s_o = copy.copy(self.app.pos_s_o)

            registration_divots_pts = copy.copy(self.app.registration_divots_pts)
            registration_cloud_pts = copy.copy(self.app.registration_cloud_pts)

        slider_value_tmp = {}
        series = {}
        if self.app.available_hololens_acquisitions():
            mat_w_to_o = copy.copy(self.app.mat_w_to_o)

            for acquisition in ACQUISITIONS_HOLOLENS:
                if not self.app.acquisitions[acquisition].empty:
                    slider_value_tmp[acquisition] = copy.copy(self.app.slider_value[acquisition])
                    timestamp = self.app.df[acquisition].index[slider_value_tmp[acquisition]]
                    series[acquisition] = self.app.df[acquisition].loc[timestamp].copy()

        if not self.app.acquisitions["qr_code_position"].empty:
            if not self.app.acquisitions["magnetic_seed"].empty:
                pos_t_q = copy.copy(self.app.pos_t_q)
                pos_s_w = copy.copy(self.app.pos_s_w)
            else:
                pos_t_q = get_out_position()
                pos_s_w = get_out_position()

        self.app.mutex_update.release()

        if not self.app.acquisitions["optical"].empty:
            if pointer_series is not None and visibility["pointer"]:
                # print("pointer")
                mat_m_to_o = get_mat_m_to_o_series(pointer_series)
                # print(mat_m_to_o)
            else:
                mat_m_to_o = get_invisible_transformation()
            self.objects["pointer"].transform(mat_m_to_o, internal=False, revert=True)
            self.objects["pointer"].update_geometry(self.visualization)

            if optical_seed_series is not None and visibility["optical_seed"]:
                # print("seed")
                mat_m_to_o = get_mat_m_to_o_series(optical_seed_series)
                # print(mat_m_to_o)
            else:
                mat_m_to_o = get_invisible_transformation()
            self.objects["optical_markers_seed"].transform(mat_m_to_o, internal=False, revert=True)
            self.objects["optical_markers_seed"].update_geometry(self.visualization)

            mat_s_to_o = np.matmul(mat_m_to_o, mat_s_to_m2)
            self.objects["optical_seed_holder"].transform(mat_s_to_o, internal=False, revert=True)
            self.objects["optical_seed_holder"].update_geometry(self.visualization)

            if mat_m_to_o[0][0] != INVISIBLE_SCALE:
                pos_gt_s_s = np.array((0,0,0,1))
                pos_gt_s_o = mul_mat44_vec4(mat_s_to_o, pos_gt_s_s)

            if optical_probe_series is not None and visibility["probe"]:
                # print("probe")
                mat_m_to_o = get_mat_m_to_o_series(optical_probe_series)
            else:
                mat_m_to_o = get_invisible_transformation()
            self.objects["optical_markers"].transform(mat_m_to_o, internal=False, revert=True)
            self.objects["optical_markers"].update_geometry(self.visualization)

            self.objects["optical_probe"].transform(translation_matrix44(vec3_to_vec4(pos_t_m)), internal=False, revert=True)
            self.objects["optical_probe"].transform(mat_m_to_o, internal=False, revert=False)
            self.objects["optical_probe"].update_geometry(self.visualization)

            mat_qf_to_o = np.matmul(mat_m_to_o, mat_qf_to_m)
            self.objects["optical_qr_code_front"].transform(mat_qf_to_o, internal=False, revert=True)
            self.objects["optical_qr_code_front"].update_geometry(self.visualization)
            mat_ql_to_qf = np.linalg.inv(mat_qf_to_ql)
            mat_ql_to_o = np.matmul(mat_qf_to_o, mat_ql_to_qf)
            self.objects["optical_qr_code_left"].transform(mat_ql_to_o, internal=False, revert=True)
            self.objects["optical_qr_code_left"].update_geometry(self.visualization)
            mat_qr_to_qf = np.linalg.inv(mat_qf_to_qr)
            mat_qr_to_o = np.matmul(mat_qf_to_o, mat_qr_to_qf)
            self.objects["optical_qr_code_right"].transform(mat_qr_to_o, internal=False, revert=True)
            self.objects["optical_qr_code_right"].update_geometry(self.visualization)
            mat_qt_to_qf = np.linalg.inv(mat_qf_to_qt)
            mat_qt_to_o = np.matmul(mat_qf_to_o, mat_qt_to_qf)
            self.objects["optical_qr_code_top"].transform(mat_qt_to_o, internal=False, revert=True)
            self.objects["optical_qr_code_top"].update_geometry(self.visualization)

            self.objects["optical_seed"].transform(translation_matrix44(vec3_to_vec4(pos_s_o)), internal=False, revert=True)
            self.objects["optical_seed"].update_geometry(self.visualization)

            if registration_divots_pts is not None and len(self.objects["registration_divots_pts"].meshes) == 0:
                # print(f"registration_divots_pts.shape {registration_divots_pts.shape}")
                self.objects["registration_divots_pts"] = create_spheres(registration_divots_pts*20, [0,1,1], DIVOT_SPHERE_RADIUS*10)
                self.objects["registration_divots_pts"].add_geometry(self.visualization)
                self.objects["registration_cloud_pts"] = create_spheres(registration_cloud_pts*20, [0.33,1,1], DIVOT_SPHERE_RADIUS*10)
                self.objects["registration_cloud_pts"].add_geometry(self.visualization)

        mat_c_to_w = {}
        first_acquisition = None
        for acquisition in CAMERA_NAMES:
            if not self.app.acquisitions[acquisition].empty:
                if visibility[acquisition] \
                or (acquisition == "ahat_depth_cam" and visibility["ahat_ab"]) \
                or (acquisition == "lt_depth_cam" and visibility["lt_ab"]):
                    if first_acquisition is None:
                        first_acquisition = acquisition
                    mat_c_to_w[acquisition] = get_mat_c_to_w_series(series[acquisition])
                    mat_c_to_w[acquisition] = np.matmul(mat_w_to_o, mat_c_to_w[acquisition]) # hololens world is now in the optical world
                else:
                    mat_c_to_w[acquisition] = get_invisible_transformation()

        if not self.app.acquisitions["qr_code_position"].empty:
            if series["qr_code_position"]["q1_m44"] != 0. and visibility["qr_code_position"]: # qr code is visible?
                if False:
                    mat_w_to_o, mat_qf_to_w = get_mat_w_to_o(series["qr_code_position"], optical_probe_series, mat_qf_to_m) # hololens world is now in the optical world so mat_qf_to_w is equal to mat_qf_to_o
                else:
                    mat_qf_to_w = get_mat_q_to_w_series(series["qr_code_position"], qr_code_id=0) # front
                    mat_qf_to_w = np.matmul(mat_w_to_o, mat_qf_to_w) # hololens world is now in the optical world
            else:
                mat_qf_to_w = get_invisible_transformation()

            if series["qr_code_position"]["q2_m44"] != 0. and visibility["qr_code_position"]: # qr code is visible?
                mat_ql_to_w = get_mat_q_to_w_series(series["qr_code_position"], qr_code_id=1) # left
                # if not np.allclose(mat_ql_to_w, np.identity(4)):
                mat_ql_to_w = np.matmul(mat_w_to_o, mat_ql_to_w) # hololens world is now in the optical world
            else:
                mat_ql_to_w = get_invisible_transformation()

            if series["qr_code_position"]["q3_m44"] != 0. and visibility["qr_code_position"]: # qr code is visible?
                mat_qr_to_w = get_mat_q_to_w_series(series["qr_code_position"], qr_code_id=2) # right
                mat_qr_to_w = np.matmul(mat_w_to_o, mat_qr_to_w) # hololens world is now in the optical world
            else:
                mat_qr_to_w = get_invisible_transformation()

            if series["qr_code_position"]["q4_m44"] != 0. and visibility["qr_code_position"]: # qr code is visible?
                mat_qt_to_w = get_mat_q_to_w_series(series["qr_code_position"], qr_code_id=3) # top
                mat_qt_to_w = np.matmul(mat_w_to_o, mat_qt_to_w) # hololens world is now in the optical world
            else:
                mat_qt_to_w = get_invisible_transformation()

            if not (np.array(pos_s_w) == np.array([0,0,0])).all(): # if not default position
                pos_s_w = mul_mat44_vec4(mat_w_to_o, vec3_to_vec4(pos_s_w))[:3] # hololens world is now in the optical world

            self.objects["qr_code_front"].transform(mat_qf_to_w, internal=False, revert=True)
            self.objects["qr_code_front"].update_geometry(self.visualization)
            self.objects["qr_code_left"].transform(mat_ql_to_w, internal=False, revert=True)
            self.objects["qr_code_left"].update_geometry(self.visualization)
            self.objects["qr_code_right"].transform(mat_qr_to_w, internal=False, revert=True)
            self.objects["qr_code_right"].update_geometry(self.visualization)
            self.objects["qr_code_top"].transform(mat_qt_to_w, internal=False, revert=True)
            self.objects["qr_code_top"].update_geometry(self.visualization)

            if first_acquisition is None:
                mat_c_to_w_tmp = get_invisible_transformation()
            else:
                mat_c_to_w_tmp = mat_c_to_w[first_acquisition]
            self.objects["projected_lines"].meshes[0].points = o3d.utility.Vector3dVector(compute_projected_points(mat_qf_to_w, mat_ql_to_w, mat_qr_to_w, mat_qt_to_w, mat_c_to_w_tmp, MARKER_LENGTH, pos_t_q, pos_s_w, pos_s_o, pos_gt_s_o[:3]))
            self.objects["projected_lines"].update_geometry(self.visualization)

            if not self.app.acquisitions["magnetic_seed"].empty:
                self.objects["qr_probe"].transform(translation_matrix44(vec3_to_vec4(pos_t_q)), internal=False, revert=True)
                self.objects["qr_probe"].transform(mat_qf_to_w, internal=False, revert=False)
                self.objects["qr_probe"].update_geometry(self.visualization)

                # if (np.array(pos_s_w) == np.array([0,0,0])).all(): # if default position then invisible
                    # self.objects["qr_seed"].transform(get_invisible_transformation(), internal=False, revert=True)
                # else:
                self.objects["qr_seed"].transform(translation_matrix44(vec3_to_vec4(pos_s_w)), internal=False, revert=True)
                self.objects["qr_seed"].update_geometry(self.visualization)

        for acquisition in CAMERA_NAMES:
            if not self.app.acquisitions[acquisition].empty:
                self.objects[acquisition].transform(mat_c_to_w[acquisition], internal=False, revert=True)
                self.objects[acquisition].update_geometry(self.visualization)

                if acquisition in ["ahat_depth_cam", "lt_depth_cam"]:
                    mat_c_to_w_tmp = mat_c_to_w[acquisition]
                    if not visibility[acquisition]:
                        mat_c_to_w_tmp = get_invisible_transformation()

                    if acquisition == "ahat_depth_cam":
                        max_value = MAX_AHAT_AB
                        max_depth_value = MAX_AHAT_DEPTH
                    else:
                        max_value = MAX_LT_AB
                        max_depth_value = MAX_LT_DEPTH

                    frame = self.app.acquisitions[acquisition + "_ab_frames"][slider_value_tmp[acquisition]]
                    frame = convert_gray_to_hsv_mapping(frame, max_value) # shape (h,w,3) min 0 max 255 dtype=uint8

                    if len(self.app.acquisitions[acquisition + "_frames"]) != 0:
                        depth_frame = self.app.acquisitions[acquisition + "_frames"][slider_value_tmp[acquisition]] # shape (height, width)
                        lut_projection = get_lut_projection_pixel_mapping(self.app.acquisitions[acquisition + "_lut_projection"]) # shape (height*width, 3)
                        points, rgb_points = get_3d_points_in_world_space(frame, depth_frame, lut_projection, max_depth_value, mat_c_to_w_tmp) # shape (height*width - removed points, 3), (height*width - removed points, 3)
                        # print(f"points.shape {points.shape} points.dtype {points.dtype} rgb_points.shape {rgb_points.shape} rgb_points.dtype {rgb_points.dtype}")

                        self.objects[acquisition + "_point_cloud"].meshes[0].points = o3d.utility.Vector3dVector(points)
                        self.objects[acquisition + "_point_cloud"].meshes[0].colors = o3d.utility.Vector3dVector(rgb_points)
                        self.objects[acquisition + "_point_cloud"].update_geometry(self.visualization)

                else:
                    frame = self.app.acquisitions[acquisition + "_frames"][slider_value_tmp[acquisition]]
                    if len(frame.shape) == 2: # i.e. nb_channel == 1
                        frame = np.repeat(frame[:, :, np.newaxis], 3, axis=2) # shape (h,w, rgb)
                    else: # shape (h,w,bgr) or shape (h,w,bgra)
                        frame = frame[...,[2,1,0]] # shape (h,w,rgb)

                frame = np.asarray(frame, dtype=None, order='C') # open3d accept only c-style buffer for the texture
                # print(f"{type(frame)} {frame.shape} {frame.dtype} {np.min(frame)} {np.max(frame)}")

                mat_c_to_w_tmp = mat_c_to_w[acquisition]
                if (acquisition == "ahat_depth_cam" and not visibility["ahat_ab"]) \
                or (acquisition == "lt_depth_cam" and not visibility["lt_ab"]):
                    mat_c_to_w_tmp = get_invisible_transformation()

                if acquisition in RESEARCH_MODE_CAMERA_NAMES:
                # if False:
                    lut_projection = get_lut_projection_camera_mapping(self.app.acquisitions[acquisition + "_lut_projection"], skip_factor=LUT_SKIP_FACTOR[acquisition]) # shape ((height/LUT_SKIP_FACTOR + 1)*(width/LUT_SKIP_FACTOR + 1), 3)
                    # print(f"lut_projection.shape {lut_projection.shape}")
                    pts = get_camera_pixels_in_world_space(lut_projection, HOLOLENS_METER_TO_MILIMETER_SCALE, mat_c_to_w_tmp) # shape ((height/LUT_SKIP_FACTOR + 1)*(width/LUT_SKIP_FACTOR + 1), 3)
                    # print(f"pts.shape {pts.shape} pts.dtype {pts.dtype}")
                    # print(pts)
                    vec3_center = mul_mat44_vec4(mat_c_to_w_tmp, np.array([0,0,0,1]))[:3]
                    vec3_top_left = pts[0]
                    vec3_top_right = pts[int(frame.shape[1]*series[acquisition]['downsampling_factor']/LUT_SKIP_FACTOR[acquisition] + 1 - 1),:]
                    vec3_bottom_right = pts[-1]
                    vec3_bottom_left = pts[int(-(frame.shape[1]*series[acquisition]['downsampling_factor']/LUT_SKIP_FACTOR[acquisition] + 1)),:]
                    # print(f"vec3_center {vec3_center} vec3_top_left {vec3_top_left} vec3_top_right {vec3_top_right} vec3_bottom_right {vec3_bottom_right} vec3_bottom_left {vec3_bottom_left}")

                    self.objects[acquisition + "_frustum"].meshes[0].vertices = o3d.utility.Vector3dVector(pts)
                    # self.objects[acquisition + "_frustum"].meshes[0].vertices = o3d.utility.Vector3dVector(np.array([vec3_top_left, vec3_top_right, vec3_bottom_right, vec3_bottom_left])) # debug
                    self.objects[acquisition + "_frustum"].meshes[1].points = o3d.utility.Vector3dVector(np.array([vec3_center, vec3_top_left, vec3_top_right, vec3_bottom_right, vec3_bottom_left]))

                else:
                # elif False:
                    # FRUSTUM_DISTANCE = {"pv_cam":1000, "vl_front_left_cam":1500, "vl_front_right_cam":2000, "ahat_depth_cam":2500, "lt_depth_cam":3000}
                    pts = compute_camera_points(series[acquisition]['focal_length_x'], series[acquisition]['focal_length_y'], series[acquisition]['center_coordinate_x'], series[acquisition]['center_coordinate_y'], frame.shape[1]*series[acquisition]['downsampling_factor'], frame.shape[0]*series[acquisition]['downsampling_factor'], mat_c_to_w_tmp, HOLOLENS_METER_TO_MILIMETER_SCALE) # FRUSTUM_DISTANCE[acquisition]) # shape (5,3)
                    self.objects[acquisition + "_frustum"].meshes[0].vertices = o3d.utility.Vector3dVector(pts[1:])
                    self.objects[acquisition + "_frustum"].meshes[1].points = o3d.utility.Vector3dVector(pts)

                self.objects[acquisition + "_frustum"].meshes[0].textures = [o3d.geometry.Image(frame)]
                self.objects[acquisition + "_frustum"].update_geometry(self.visualization)
