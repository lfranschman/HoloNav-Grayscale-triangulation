import math
import numpy as np
import scipy
import scipy.spatial
import scipy.optimize # fsolve, root, least_squares
try:
    import open3d as o3d
    G_OPEN_3D_LOADED = True
except ImportError:
    G_OPEN_3D_LOADED = False

try:
    import cv2
    G_OPENCV_LOADED = True
except ImportError:
    G_OPENCV_LOADED = False

BIG_NUMBER = 1000000000

def polar_to_cartesian(phi, theta, r):
    sin_theta = math.sin(theta)
    return r*math.cos(phi)*sin_theta, r*math.sin(phi)*sin_theta, r*math.cos(theta)

# from_pt, to_pt, up_vec shape (3,)
def camera_look_at(from_pt, to_pt, up_vec = (0,-1,0)):
    z_axis = normalize_vec3(to_pt - from_pt)
    x_axis = np.cross(z_axis, up_vec)
    y_axis = -np.cross(x_axis, z_axis)

    mat44 = np.identity(4)
    mat44[:3,0] = x_axis
    mat44[:3,1] = y_axis
    mat44[:3,2] = z_axis
    mat44[:3,3] = from_pt

    return mat44

def vec3_to_vec4(vec3):
    dtype = None
    if isinstance(vec3,np.ndarray):
        dtype = vec3.dtype
    return np.array((vec3[0],vec3[1],vec3[2],1),dtype=dtype)

# shape (n, 3)
# output shape (n,4)
def vec3_list_to_vec4_list(vec3_list):
    return np.pad(vec3_list, ((0,0),(0,1)), mode='constant', constant_values=1)
    # return np.hstack((vec3_list, np.ones((vec3_list.shape[0], 1)))) # check which one is faster

def identity_matrix44():
    # return np.array([[1., 0., 0., 0.] \
           # ,[0., 1., 0., 0.] \
           # ,[0., 0., 1., 0.] \
           # ,[0., 0., 0., 1.]])
    return np.identity(4)

def translation_matrix44(vec4):
    return np.array([
        [1,0,0,vec4[0]],
        [0,1,0,vec4[1]],
        [0,0,1,vec4[2]],
        [0,0,0,1]
    ])

def pose_matrix44(vec3, mat33_rot):
    return np.array([
        [mat33_rot[0][0], mat33_rot[0][1], mat33_rot[0][2], vec3[0]],
        [mat33_rot[1][0], mat33_rot[1][1], mat33_rot[1][2], vec3[1]],
        [mat33_rot[2][0], mat33_rot[2][1], mat33_rot[2][2], vec3[2]],
        [              0,               0,               0,       1]
    ])

def rotation_quaternion_xyzw_matrix44(quat):
    r = scipy.spatial.transform.Rotation.from_quat(quat)
    mat = r.as_matrix()
    return np.array([
        [mat[0][0], mat[0][1], mat[0][2], 0],
        [mat[1][0], mat[1][1], mat[1][2], 0],
        [mat[2][0], mat[2][1], mat[2][2], 0],
        [        0,         0,         0, 1]
    ])

# the returned rotation matrix converts a point in the local reference frame to a point in the global reference frame
def rotation_quaternion_wxyz_matrix44(quat):
    r00 = 2*(quat[0]*quat[0] + quat[1]*quat[1]) - 1
    r01 = 2*(quat[1]*quat[2] - quat[0]*quat[3])
    r02 = 2*(quat[1]*quat[3] + quat[0]*quat[2])

    r10 = 2*(quat[1]*quat[2] + quat[0]*quat[3])
    r11 = 2*(quat[0]*quat[0] + quat[2]*quat[2]) - 1
    r12 = 2*(quat[2]*quat[3] - quat[0]*quat[1])

    r20 = 2*(quat[1]*quat[3] - quat[0]*quat[2])
    r21 = 2*(quat[2]*quat[3] + quat[0]*quat[1])
    r22 = 2*(quat[0]*quat[0] + quat[3]*quat[3]) - 1

    return np.array([[r00, r01, r02, 0],
                     [r10, r11, r12, 0],
                     [r20, r21, r22, 0],
                     [  0,   0,   0, 1]])

# vec3 in degree if degrees is True, radian otherwise
def rotation_euler_matrix44(vec3, degrees):
    r = scipy.spatial.transform.Rotation.from_euler('xyz', vec3, degrees=degrees)
#     r = scipy.spatial.transform.Rotation.from_euler('zyx', vec3, degrees=degrees)
    mat = r.as_matrix()
    return np.array([
        [mat[0][0], mat[0][1], mat[0][2], 0],
        [mat[1][0], mat[1][1], mat[1][2], 0],
        [mat[2][0], mat[2][1], mat[2][2], 0],
        [        0,         0,         0, 1]
    ])

# vec3 in degree
# mrp -> modified rodrigues parameters
def rotation_mrp_matrix44(vec3):
    #r = scipy.spatial.transform.Rotation.from_mrp(vec3) # doesn't give same result as opencv???
    #mat = r.as_matrix()
    mat = cv2.Rodrigues(np.array(vec3))[0]
    return np.array([
        [mat[0][0], mat[0][1], mat[0][2], 0],
        [mat[1][0], mat[1][1], mat[1][2], 0],
        [mat[2][0], mat[2][1], mat[2][2], 0],
        [        0,         0,         0, 1]
    ])

def normalize_vec3(vec3):
    return vec3/np.linalg.norm(vec3)

def normalize_vec4(vec4):
    normalized_vec4 = vec4/np.linalg.norm(vec4[:3])
    normalized_vec4[3] = 1
    return normalized_vec4

def mul_mat44_vec4(mat44, vec4):
    vec4 = np.transpose(np.expand_dims(vec4,axis=0))
    vec4 = np.matmul(mat44, vec4)
    return np.transpose(vec4)[0]

# mat44 shape (4,4)
# vec4_list shape (_, 4)
def mul_mat44_vec4_list(mat44, vec4_list):
    return np.einsum("ij,kj->ik", vec4_list, mat44)

# https://docs.opencv.org/4.5.3/d9/d0c/group__calib3d.html
# p shape (3,)
# distortion_coefficients shape (5,)
# camera_proj (3,3)
def projection_3d_point(p, distortion_coefficients, camera_proj, integer=True):
    x_p = p[0]/p[2]
    y_p = p[1]/p[2]
    r_2 = x_p*x_p + y_p*y_p
    x_dp = x_p*((1 + distortion_coefficients[0]*r_2 + distortion_coefficients[1]*r_2*r_2 + distortion_coefficients[4]*r_2*r_2*r_2)/(1)) + 2*distortion_coefficients[2]*x_p*y_p + distortion_coefficients[3]*(r_2 + 2*x_p*x_p)
    y_dp = y_p*((1 + distortion_coefficients[0]*r_2 + distortion_coefficients[1]*r_2*r_2 + distortion_coefficients[4]*r_2*r_2*r_2)/(1)) + 2*distortion_coefficients[3]*x_p*y_p + distortion_coefficients[2]*(r_2 + 2*y_p*y_p)

    x = camera_proj[0][0]*x_dp + camera_proj[0][2]
    y = camera_proj[1][1]*y_dp + camera_proj[1][2]

    if integer:
        return np.array([int(math.floor(x)),int(math.floor(y))], dtype=np.uint32)
    return np.array([x,y])

# input translation vector (4,), quaternion rotation (4,) and rig_to_camera (4,4)
def get_mat_camera_to_world(translation_rig_to_world, rotation_rig_to_world, rig_to_camera):
    mat_trans_R_to_W = translation_matrix44(translation_rig_to_world)
    mat_rot_R_to_W = rotation_quaternion_xyzw_matrix44(rotation_rig_to_world)
    mat_R_to_W = np.matmul(mat_trans_R_to_W, mat_rot_R_to_W)
    camera_to_rig = np.linalg.inv(rig_to_camera)
    return np.matmul(mat_R_to_W, camera_to_rig)

# input translation vector (4,) and modified Rodriguez rotation (3,)
def get_mat_mrp(translation, rotation):
    mat_translation = translation_matrix44(translation)
    mat_rotation = rotation_mrp_matrix44(rotation)
    return np.matmul(mat_translation, mat_rotation)

# input translation vector (4,) and quaternion rotation (4,)
def get_mat_quaternion(translation, rotation):
    mat_translation = translation_matrix44(translation)
    mat_rotation = rotation_quaternion_xyzw_matrix44(rotation)
    return np.matmul(mat_translation, mat_rotation)

# input translation vector (4,) and quaternion rotation (4,)
def get_inverse_mat_quaternion(translation, rotation):
    mat_translation = translation_matrix44([-translation[0],-translation[1],-translation[2],1])
    mat_rotation = rotation_quaternion_xyzw_matrix44([-rotation[0],-rotation[1],-rotation[2],rotation[3]])
    return np.matmul(mat_rotation, mat_translation)

def squared_distance_vector_2d(vec1, vec2):
    return (vec1[0] - vec2[0])*(vec1[0] - vec2[0]) \
                    + (vec1[1] - vec2[1])*(vec1[1] - vec2[1])

def squared_distance_vector_3d(vec1, vec2):
    return (vec1[0] - vec2[0])*(vec1[0] - vec2[0]) \
            + (vec1[1] - vec2[1])*(vec1[1] - vec2[1]) \
            + (vec1[2] - vec2[2])*(vec1[2] - vec2[2])

def distance_vector_2d(vec1, vec2):
    return math.sqrt(squared_distance_vector_2d(vec1, vec2))

# pt_list1 and pt_list2 doesn't have necessarilly the same number of points
def distance_between_pt_lists(pt_list1, pt_list2, squared_distance_fonction=squared_distance_vector_2d):
    max_distance = 0
    centerline_error = 0
    for mk in pt_list1:
        dist_min = BIG_NUMBER
        for mk2 in pt_list2:
            squared_dist = squared_distance_fonction(mk, mk2)
            if squared_dist < dist_min:
                dist_min = squared_dist

        if dist_min > max_distance:
            max_distance = dist_min
        centerline_error = centerline_error + math.sqrt(dist_min)
    return centerline_error/len(pt_list1), math.sqrt(max_distance)

# set1.shape and set2.shape (n,3)
# based on Arun et al., 1987
# https://stackoverflow.com/questions/66923224/rigid-registration-of-two-point-clouds-with-known-correspondence
def point_based_registration(set1, set2):
    set1 = set1.transpose() # (3, n)
    set2 = set2.transpose() # (3, n)

    # calculate centroids
    set1_c = np.mean(set1, axis = 1).reshape((-1,1)) # shape(3,1)
    set2_c = np.mean(set2, axis = 1).reshape((-1,1)) # shape(3,1)
    # print(set1_c)
    # print(set2_c)

    # subtract centroids
    q1 = set1 - set1_c
    q2 = set2 - set2_c

    h = np.matmul(q1,q2.transpose()) # calculate covariance matrix
    # print(f"h {h}")

    # calculate singular value decomposition (SVD)
    u, _, vt = np.linalg.svd(h) # the SVD of linalg gives you vt
    # print(f"u {u}")
    # print(f"vt {vt}")

    r = np.matmul(vt.transpose(), u.transpose()) # calculate rotation matrix # shape (3,3)
    # print(f"r {r}")

    det = np.linalg.det(r)
    if np.allclose(det, -1.0): # following Arun et al., change the sign of the 3rd column of vt? (or 3rd line, depending on transpose?)
        vt[0][2] = -vt[0][2]
        vt[1][2] = -vt[1][2]
        vt[2][2] = -vt[2][2]
        r = np.matmul(vt.transpose(), u.transpose()) # calculate rotation matrix # shape (3,3)
        # print(f"r {r}")
        det = np.linalg.det(r)

    # print(f"np.linalg.det(r) {det}")
    assert np.allclose(det, 1.0), "Rotation matrix of N-point registration not 1, see paper Arun et al."

    t = set2_c - np.matmul(r, set1_c) # calculate translation matrix # shape (3, 1)

    predictions = t + np.matmul(r, set1)
    rmse = np.sqrt(np.mean((predictions - set2)**2)) # TODO check, not sure it is the right formula
    norm = np.linalg.norm(predictions.transpose() - set2.transpose(), axis=1)
    mean = np.mean(norm)
    max_dist = np.max(norm)

    mat = pose_matrix44(t.reshape((3,)), r)
    return mat, rmse, mean, max_dist

# set1.shape (n,3) and set2.shape (m,3)
def iterative_closest_point(set1, set2, threshold = 0.2, mat_init = None, max_iteration = 2000):
    if mat_init is None:
        mat_init = np.identity(4)

    pc_set1 = o3d.cpu.pybind.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(set1))
    pc_set2 = o3d.cpu.pybind.geometry.PointCloud(o3d.cpu.pybind.utility.Vector3dVector(set2))

    reg_p2p = o3d.pipelines.registration.registration_icp(pc_set1, pc_set2, threshold, mat_init \
            , o3d.pipelines.registration.TransformationEstimationPointToPoint() \
            #, o3d.pipelines.registration.TransformationEstimationPointToPlane() \
            , o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iteration))
    print(reg_p2p)
    # print(reg_p2p.transformation)

    evaluation = o3d.pipelines.registration.evaluate_registration(pc_set1, pc_set2, threshold, mat_init)
    # print(evaluation)

    set1_transformed = mul_mat44_vec4_list(reg_p2p.transformation, vec3_list_to_vec4_list(set1)) # shape (n,4)
    mean, max = distance_between_pt_lists(set1_transformed, set2, squared_distance_vector_3d)
    return reg_p2p.transformation, mean, max

# REF https://stackoverflow.com/questions/44631259/line-line-intersection-in-python-with-numpy
# takes in two lines, the line formed by pt1 and pt2, and the line formed by pt3 and pt4, and finds their intersection or closest point
def intersection_lines(pt1,pt2,pt3,pt4):
    # least squares method
    def errFunc(estimates):
        s, t = estimates
        x = pt1 + s * (pt2 - pt1) - (pt3 + t * (pt4 - pt3))
        return x

    estimates = [1, 1]

    sols = scipy.optimize.least_squares(errFunc, estimates)
    s,t = sols.x

    x1 =  pt1[0] + s * (pt2[0] - pt1[0])
    x2 =  pt3[0] + t * (pt4[0] - pt3[0])
    y1 =  pt1[1] + s * (pt2[1] - pt1[1])
    y2 =  pt3[1] + t * (pt4[1] - pt3[1])
    z1 =  pt1[2] + s * (pt2[2] - pt1[2])
    z2 = pt3[2] + t * (pt4[2] - pt3[2])

    x = (x1 + x2) / 2  # halfway point if they don't match
    y = (y1 + y2) / 2  # halfway point if they don't match
    z = (z1 + z2) / 2  # halfway point if they don't match

    return (x,y,z)

# REF https://gamedev.stackexchange.com/questions/23743/whats-the-most-efficient-way-to-find-barycentric-coordinates
# pt/a/b/c shape (2,)
def barycentric(pt, a, b, c):
    v0 = b - a
    v1 = c - a
    v2 = pt - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    d20 = np.dot(v2, v0)
    d21 = np.dot(v2, v1)
    denom = d00*d11 - d01*d01
    if denom == 0:
        return None
    v = (d11*d20 - d01*d21)/denom
    w = (d00*d21 - d01*d20)/denom
    u = 1. - v - w
    return (u,v,w)

# REF https://stackoverflow.com/questions/2049582/how-to-determine-if-a-point-is-in-a-2d-triangle
# pt/v1/v2/v3 shape (2,)
if False:
    def point_in_triangle(pt, v1, v2, v3):
        # p1/p2/p3 shape (2,)
        def sign(p1, p2, p3):
            return (p1[0] - p3[0])*(p2[1] - p3[1]) - (p2[0] - p3[0])*(p1[1] - p3[1])

        d1 = sign(pt, v1, v2)
        d2 = sign(pt, v2, v3)
        d3 = sign(pt, v3, v1)

        has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
        has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)

        return not (has_neg and has_pos)

# REF https://stackoverflow.com/questions/36387928/loop-for-points-in-2d-triangles
# pt/p1/p2/p3 shape (2,)
def point_in_triangle(pt, p1, p2, p3):
    full = abs(p1[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - p2[1]))
    first = abs(p1[0]*(p2[1] - pt[1]) + p2[0]*(pt[1] - p1[1]) + pt[0]*(p1[1] - p2[1]))
    second = abs(p1[0]*(pt[1] - p3[1]) + pt[0]*(p3[1] - p1[1]) + p3[0]*(p1[1] - pt[1]))
    third = abs(pt[0]*(p2[1] - p3[1]) + p2[0]*(p3[1] - pt[1]) + p3[0]*(pt[1] - p2[1]))
    return abs(first + second + third - full) < .0000000001
