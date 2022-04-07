import numpy as np
import cv2
import cv2.aruco as aruco

from UtilMaths import get_mat_mrp, vec3_to_vec4, intersection_lines, point_based_registration
from calibration_helpers import get_lut_projection_pixel

class QRCodeDetection:
    MARKER_LENGTH = 60 # in mm # size printed QR code 6cm
    NB_QR_CODE = 4

    def __init__(self):
        self.aruco_dict = aruco.custom_dictionary(10, 4)
        mirror_dict = np.array([[172, 67, 25, 108, 194, 53, 54, 152] \
            , [95, 31, 245, 213, 248, 250, 171, 175] \
            , [64, 207, 17, 179, 243, 2, 205, 136] \
            , [132, 150, 33, 90, 105, 33, 90, 132] \
            , [12, 249, 50, 103, 159, 48, 230, 76] \
            , [149, 79, 209, 121, 242, 169, 158, 139] \
            , [46, 162, 15, 70, 69, 116, 98, 240] \
            , [177, 224, 202, 42, 7, 141, 84, 83] \
            , [211, 190, 231, 155, 125, 203, 217, 231] \
            , [235, 118, 111, 188, 110, 215, 61, 246]], dtype=np.uint8) # shape (10, 4*2) -> 4 rotations (90 degrees ccw rotation) and 2 uint8 is enough to store 4x4 qr code (16 bits)
        self.aruco_dict.bytesList = mirror_dict

        self.aruco_param = aruco.DetectorParameters_create()

    # extrinsic.shape (4,4)
    # intrinsic.shape (3,3)
    # distortion_coefficients (5,)
    # return mat_q_to_w_list (NB_QR_CODE,4,4) # 4 qr codes
    def detect(self, img, extrinsic, intrinsic, distortion_coefficients):
        mat_q_to_w_list = np.zeros((self.NB_QR_CODE,4,4), dtype=np.float64)

        bboxs, ids, rejected = aruco.detectMarkers(img, self.aruco_dict, parameters = self.aruco_param)

        if ids is not None:
            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(bboxs, self.MARKER_LENGTH, intrinsic, distortion_coefficients)

            for i in range(self.NB_QR_CODE):
                if i in ids:
                    # print(f"tvecs {tvecs} rvecs {rvecs}")
                    translation_vec = vec3_to_vec4(tvecs[np.where(ids == i)[0][0]][0])
                    rotation_vec = rvecs[np.where(ids == i)[0][0]][0]
                    mat_q_to_c = get_mat_mrp(translation_vec, rotation_vec)
                    mat_q_to_w_list[i,:,:] = np.matmul(extrinsic, mat_q_to_c)

        return mat_q_to_w_list

    # extrinsic.shape (4,4)
    # return mat_q_to_w_list (NB_QR_CODE,4,4) # 4 qr codes
    def detect_triangulation(self, frame_left, extrinsic_left, lut_projection_left, frame_right, extrinsic_right, lut_projection_right):
        mat_q_to_w_list = np.zeros((self.NB_QR_CODE,4,4), dtype=np.float64)

        bboxs_left, ids_left, _ = aruco.detectMarkers(frame_left, self.aruco_dict, parameters = self.aruco_param)
        # print(bboxs_left)
        bboxs_right, ids_right, _ = aruco.detectMarkers(frame_right, self.aruco_dict, parameters = self.aruco_param)
        # print(bboxs_right)

        for i in range(self.NB_QR_CODE):
            if ids_left is not None and i in ids_left and ids_right is not None and i in ids_right:
                id_left = np.where(ids_left == i)[0][0]
                id_right = np.where(ids_right == i)[0][0]

                corners = []
                for corner in range(4):
                    corner_left_c_left = get_lut_projection_pixel(lut_projection_left, bboxs_left[id_left][0][corner][0], bboxs_left[id_left][0][corner][1])
                    corner_left_w = np.matmul(extrinsic_left, vec3_to_vec4(corner_left_c_left))
                    origin_left_w = np.matmul(extrinsic_left, np.array([0,0,0,1]))

                    corner_right_c_right = get_lut_projection_pixel(lut_projection_right, bboxs_right[id_right][0][corner][0], bboxs_right[id_right][0][corner][1])
                    corner_right_w = np.matmul(extrinsic_right, vec3_to_vec4(corner_right_c_right))
                    origin_right_w = np.matmul(extrinsic_right, np.array([0,0,0,1]))

                    corner_w = intersection_lines(origin_left_w, corner_left_w, origin_right_w, corner_right_w)
                    corners.append(corner_w)
                corners = np.array(corners)

                reference = np.array(((-self.MARKER_LENGTH/2,self.MARKER_LENGTH/2,0), (self.MARKER_LENGTH/2,self.MARKER_LENGTH/2,0), (self.MARKER_LENGTH/2,-self.MARKER_LENGTH/2,0), (-self.MARKER_LENGTH/2,-self.MARKER_LENGTH/2,0)))

                mat_corners_to_reference, rmse, mean, max_dist = point_based_registration(corners, reference)
                mat_q_to_w_list[i,:,:] = np.linalg.inv(mat_corners_to_reference)

                # print(corners)
                # print(f"{np.linalg.norm(corners[0] - corners[1])} {np.linalg.norm(corners[0] - corners[3])} {np.linalg.norm(corners[1] - corners[2])} {np.linalg.norm(corners[2] - corners[3])}")
                # print(f"{np.linalg.norm(reference[0] - reference[1])} {np.linalg.norm(reference[0] - reference[3])} {np.linalg.norm(reference[1] - reference[2])} {np.linalg.norm(reference[2] - reference[3])}")
                # print(f"rmse {rmse} mean {mean} max_dist {max_dist}")

        return mat_q_to_w_list
