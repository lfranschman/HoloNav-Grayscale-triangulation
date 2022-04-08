import numpy as np
import open3d as o3d

from UtilMaths import vec3_to_vec4, mul_mat44_vec4, rotation_euler_matrix44, identity_matrix44 #, print_matrix44
from UtilImage import hls_to_rgb

# RESET_BOUNDING_BOX = False
RESET_BOUNDING_BOX = True

SCALE_UNIT = 1 # in mm
# SCALE_UNIT = 1/1000 # in m

OPTICAL_SPHERE_RADIUS = 10*SCALE_UNIT # in mm

COEF_PROJECTION = 20 # we want the projection long enough to reach the projected 2d image

# https://github.com/isl-org/Open3D/blob/master/examples/python/visualization/non_blocking_visualization.py
# http://www.open3d.org/docs/release/python_api/open3d.geometry.TriangleMesh.html

INVISIBLE_SCALE = 0.01 # dirty way to hide object

def get_out_position():
    # return (50,0,-50)
    return (0,0,0)

# dirty way to hide object
def get_invisible_transformation():
    # mat = np.zeros((4,4)) # doesn't work as we need to revert (so inverse matrix) transformation with open3d meshes
    mat = np.identity(4)*INVISIBLE_SCALE
    mat[3,3] = 1
    return mat
    # return translation_matrix44([200,0,-50,1])
    # return translation_matrix44([0,0,0,1])

class MeshGroup:
    def __init__(self):
        self.meshes = []
        self.mat_m_to_w = identity_matrix44()

    def add_mesh(self, mesh):
        self.meshes.append(mesh)

    def add_geometry(self, vis, reset_bounding_box=RESET_BOUNDING_BOX):
        for mesh in self.meshes:
            vis.add_geometry(mesh, reset_bounding_box)

    def update_geometry(self, vis):
        for mesh in self.meshes:
            vis.update_geometry(mesh)

    def transform(self, mat44, internal=False, revert=False):
        if not internal:
            if revert:
                mat_w_to_m = np.linalg.inv(self.mat_m_to_w)
                mat44 = np.matmul(mat44, mat_w_to_m)

            self.mat_m_to_w = np.matmul(mat44, self.mat_m_to_w)

        for mesh in self.meshes:
            mesh.transform(mat44)

    # never tested
    def translate(self, vec3):
        for mesh in self.meshes:
            mesh.translate(vec3)

def create_sphere(vec3, color3, radius):
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
    sphere.paint_uniform_color(color3)
    sphere.compute_vertex_normals()
    sphere.translate(vec3)
    return sphere

# hls_color3 in Hue Lightness Saturation format, list
def create_spheres(vec3_list, hls_color3, radius):
    cs = MeshGroup()
    # if True:
    if False:
        delta = 1/len(vec3_list)
    else:
        delta = 0.9/len(vec3_list)
        hls_color3[1] = 0.9
    hls_color3 = list(hls_color3)
    for vec3 in vec3_list:
        cs.add_mesh(create_sphere(vec3, hls_to_rgb(hls_color3), radius))
        # color = np.max(((0,0,0), np.array(color) - delta), axis=0)
        hls_color3[1] = max(0, hls_color3[1] - delta)
    return cs

# length, center_radius in mm
def create_coordinate_system(mat44, length = 10, color_origin = (1,0,0)):
    # vec4_center = mul_mat44_vec4(mat44, np.array([0,0,0,1]))
    # vec4_x_axe = mul_mat44_vec4(mat44, np.array([1,0,0,1])) - vec4_center
    # print(vec4_x_axe)
    # vec4_y_axe = mul_mat44_vec4(mat44, np.array([0,1,0,1])) - vec4_center
    # print(vec4_y_axe)
    # vec4_z_axe = mul_mat44_vec4(mat44, np.array([0,0,1,1])) - vec4_center
    # print(vec4_z_axe)

    cylinder_radius = 0.05*length
    center_radius = 0.2*length

    # length = 0.98 # debug, less than 1.0 to see the tip of the arrow, size similar to o3d.geometry.TriangleMesh.create_coordinate_frame
    # cylinder_radius = 0.035 # debug, size similar to o3d.geometry.TriangleMesh.create_coordinate_frame
    # center_radius = 0.08

    cs = MeshGroup()
    cs.add_mesh(create_sphere((0,0,0), color_origin, center_radius))

    x_axis = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=length, resolution=20, split=4)
    x_axis.paint_uniform_color([1, 0, 0])
    x_axis.compute_vertex_normals()
    mat_rot = rotation_euler_matrix44([0,-90,0], True)
    x_axis.transform(mat_rot)
    x_axis.translate([length/2,0,0])
    cs.add_mesh(x_axis)

    y_axis = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=length, resolution=20, split=4)
    y_axis.paint_uniform_color([0, 1, 0])
    y_axis.compute_vertex_normals()
    mat_rot = rotation_euler_matrix44([-90,0,0], True)
    # print_matrix44(mat_rot)
    y_axis.transform(mat_rot)
    y_axis.translate([0,length/2,0])
    cs.add_mesh(y_axis)

    z_axis = o3d.geometry.TriangleMesh.create_cylinder(radius=cylinder_radius, height=length, resolution=20, split=4)
    z_axis.paint_uniform_color([0, 0, 1])
    z_axis.compute_vertex_normals()
    z_axis.translate([0,0,length/2])
    cs.add_mesh(z_axis)

    cs.transform(mat44)
    return cs

# ndi polaris vega VT
# check x,y,z axis in Polaris Vega API Guide.pdf, page 137/144
def create_optical_tracker():
    cs = MeshGroup()

    vertices = o3d.utility.Vector3dVector(np.array([[0,0,0],[-1312/2,-1566/2,-2400],[1312/2,-1566/2,-2400],[-1312/2,1566/2,-2400],[1312/2,1566/2,-2400]]))
    triangles = o3d.utility.Vector3iVector(np.array([[0,1,3],[0,1,2],[0,3,4],[0,2,4],[1,2,4],[4,3,1]],dtype=np.int32))
    pyramid = o3d.geometry.TriangleMesh(vertices, triangles)
    pyramid = o3d.geometry.LineSet.create_from_triangle_mesh(pyramid)
    cs.add_mesh(pyramid)

    floor = o3d.geometry.TriangleMesh.create_box(10,1566,2500)
    floor.translate([1312/2 + 500,-1566/2,-2500]) # +500 just to have more space for visibility
    cs.add_mesh(floor)

    return cs

def create_camera(cs_length, extrinsic_mat):
    cs = create_coordinate_system(extrinsic_mat, length=cs_length, color_origin=[1,1,1])
    return cs

# output shape (5,3)
def compute_camera_points(focal_length_x, focal_length_y, center_coordinate_x, center_coordinate_y, image_width, image_height, extrinsic_mat, scale):
    u = 0
    v = 0
    x0 = (u - center_coordinate_x)/focal_length_x
    y0 = (v - center_coordinate_y)/focal_length_y
    u = image_width
    v = image_height
    x1 = (u - center_coordinate_x)/focal_length_x
    y1 = (v - center_coordinate_y)/focal_length_y

    x0 *= scale # in mm
    y0 *= scale # in mm
    x1 *= scale # in mm
    y1 *= scale # in mm

    vec4_center = mul_mat44_vec4(extrinsic_mat, np.array([0,0,0,1]))
    vec4_top_left = mul_mat44_vec4(extrinsic_mat, np.array([x0,y0,scale,1]))
    vec4_top_right = mul_mat44_vec4(extrinsic_mat, np.array([x1,y0,scale,1]))
    vec4_bottom_right = mul_mat44_vec4(extrinsic_mat, np.array([x1,y1,scale,1]))
    vec4_bottom_left = mul_mat44_vec4(extrinsic_mat, np.array([x0,y1,scale,1]))

    return np.array([vec4_center[:3],vec4_top_left[:3],vec4_top_right[:3],vec4_bottom_right[:3],vec4_bottom_left[:3]]) # shape (5,3)

def create_camera_frustum(focal_length_x, focal_length_y, center_coordinate_x, center_coordinate_y, image_width, image_height, extrinsic_mat, scale):
    cs = MeshGroup()
    pts = compute_camera_points(focal_length_x, focal_length_y, center_coordinate_x, center_coordinate_y, image_width, image_height, extrinsic_mat, scale)
    # print(f"pts.shape {pts.shape} dtype {pts.dtype}")
    # print(pts)

    vertices = o3d.utility.Vector3dVector(pts[1:])
    # triangles = o3d.utility.Vector3iVector(np.array([[0,1,2],[2,3,0]],dtype=np.int32)) # texture backward, it seems order vertices matters, normals side doesn't change anything
    triangles = o3d.utility.Vector3iVector(np.array([[2,1,0],[0,3,2]],dtype=np.int32))
    sprite = o3d.geometry.TriangleMesh(vertices, triangles)
    # sprite.triangle_uvs = o3d.utility.Vector2dVector(np.array([[0,0],[0,1],[1,1],[1,1],[1,0],[0,0]]))
    sprite.triangle_uvs = o3d.utility.Vector2dVector(np.array([[1,1],[1,0],[0,0],[0,0],[0,1],[1,1]]))
    sprite.triangle_material_ids = o3d.utility.IntVector(np.array([0,0],dtype=np.int32))
    sprite.triangle_normals = o3d.utility.Vector3dVector(np.array([[0,0,-1],[0,0,-1]]))
    sprite.vertex_normals = o3d.utility.Vector3dVector(np.array([[0,0,-1],[0,0,-1],[0,0,-1],[0,0,-1]]))
    # sprite.compute_vertex_normals()
    # sprite.compute_triangle_normals()
    # print(f"sprite.triangle_normals {np.asarray(sprite.triangle_normals)}")
    # print(f"sprite.vertex_normals {np.asarray(sprite.vertex_normals)}")
    image = np.ones((100, 100, 3), dtype=np.float32) # dummy image
    image[20:50,50:90,:] = 0
    image = o3d.geometry.Image(image)
    sprite.textures = [image]
    cs.add_mesh(sprite)

    vertices = o3d.utility.Vector3dVector(pts)
    # triangles = o3d.utility.Vector3iVector(np.array([[0,4,1],[0,1,2],[0,2,3],[0,4,3],[1,2,3],[3,4,1]],dtype=np.int32))
    # triangles = o3d.utility.Vector3iVector(np.array([[0,4,1],[0,1,2],[0,2,3],[0,4,3]],dtype=np.int32))
    # pyramid = o3d.geometry.TriangleMesh(vertices, triangles)
    # pyramid = o3d.geometry.LineSet.create_from_triangle_mesh(pyramid)
    # pyramid = o3d.geometry.LineSet.create_camera_visualization(image_width, image_height, intrinsic_matrix, identity_matrix44(), scale=scale) # same as previous code but done by open3d
    # pyramid.paint_uniform_color((1,0,1))
    # cs.add_mesh(pyramid)
    lines = o3d.utility.Vector2iVector(np.array(((0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (3,4), (4,1))))
    line_set = o3d.geometry.LineSet(vertices, lines)
    line_set.colors = o3d.utility.Vector3dVector(np.array(((1,0,1),(1,0,1),(1,0,1),(1,0,1),(1,0,1),(1,0,1),(1,0,1),(1,0,1))))
    cs.add_mesh(line_set)

    return cs

def create_camera_frustum_from_lut_projection(width, height, skip_factor):
    print(f"width {width} height {height} skip_factor {skip_factor}")

    cs = MeshGroup()

    # WARNING be careful here for the initialization with pts, if you put random numbers it's not gonna work as it seems the camera set its near/far parameters with the first frame? so values should be close to reality. At least that's what I think right now.

    # pts = np.zeros((int((width/skip_factor + 1)*(height/skip_factor + 1)),3))
    # placeholder points grid
    # nb_pts = int((width/skip_factor + 1)*(height/skip_factor + 1))
    # print(nb_pts)
    # x_values = np.arange(0,width/skip_factor + 1)
    # y_values = np.arange(0,height/skip_factor + 1)
    # xx, yy = np.meshgrid(x_values, y_values)
    # pts = np.stack((xx,yy),axis=2) # shape (width/skip_factor + 1, height/skip_factor + 1, 2)
    # pts = np.reshape(pts,(nb_pts, 2))
    # pts = np.pad(pts, ((0,0),(0,1)), mode='constant', constant_values=1) # shape ((width/skip_factor + 1)*(height/skip_factor + 1), 3)
    # pts = np.array([[    0.   ,  0.   ,  0.] \

    # , [-1000., -1000. , 1000.] \
    # , [    0., -1000.,  1000.] \
    # , [    0.,     0. , 1000.] \
    # , [-1000. ,    0. , 1000.]])
    # pts = np.tile(pts, (4,1))

    pts = np.array([[-1000., -1000. , 1000.]])
    pts = np.tile(pts, (int((width/skip_factor + 1)*(height/skip_factor + 1)),1))
    # print(f"pts.shape {pts.shape} dtype {pts.dtype}")
    # print(pts)

    list_triangles = []
    list_uvs = []
    for j in range(int(height/skip_factor)):
        for i in range(int(width/skip_factor)):
            top_left = int((width/skip_factor + 1)*j + i)
            top_right = int((width/skip_factor + 1)*j + i + 1)
            bottom_left = int((width/skip_factor + 1)*(j + 1) + i)
            bottom_right = int((width/skip_factor + 1)*(j + 1) + i + 1)
            list_triangles.append([bottom_right,top_right,top_left])
            list_triangles.append([top_left,bottom_left,bottom_right])

            uv_top_left = [i/(width/skip_factor), j/(height/skip_factor)]
            uv_top_right = [(i + 1)/(width/skip_factor), j/(height/skip_factor)]
            uv_bottom_left = [i/(width/skip_factor), (j + 1)/(height/skip_factor)]
            uv_bottom_right = [(i + 1)/(width/skip_factor), (j + 1)/(height/skip_factor)]
            list_uvs.extend([uv_bottom_right,uv_top_right,uv_top_left])
            list_uvs.extend([uv_top_left,uv_bottom_left,uv_bottom_right])
    list_triangles = np.array(list_triangles, dtype=np.uint32) # shape (2*(width/skip_factor)*(height/skip_factor), 3)
    # print(f"list_triangles.shape {list_triangles.shape} dtype {list_triangles.dtype}")
    # print(list_triangles)
    list_uvs = np.array(list_uvs) # shape (3*2*(width/skip_factor)*(height/skip_factor), 2)
    # list_uvs = np.array(list_uvs, dtype=np.float32) # shape (3*2*(width/skip_factor)*(height/skip_factor), 2)
    # print(f"list_uvs.shape {list_uvs.shape} dtype {list_uvs.dtype}")
    # print(list_uvs)

    vertices = o3d.utility.Vector3dVector(pts)
    triangles = o3d.utility.Vector3iVector(list_triangles)
    sprite = o3d.geometry.TriangleMesh(vertices, triangles)
    sprite.triangle_uvs = o3d.utility.Vector2dVector(np.asarray(list_uvs.copy(), dtype=None, order='C'))
    sprite.triangle_material_ids = o3d.utility.IntVector(np.zeros((int(2*(width/skip_factor)*(height/skip_factor))),dtype=np.int32))
    sprite.triangle_normals = o3d.utility.Vector3dVector(np.tile(np.array([0,0,-1]), (int(2*(width/skip_factor)*(height/skip_factor)), 1))) # shape (2*(width/skip_factor)*(height/skip_factor), 3)
    # print(np.asarray(sprite.triangle_normals))
    sprite.vertex_normals = o3d.utility.Vector3dVector(np.tile(np.array([0,0,-1]), (pts.shape[0], 1))) # shape (pts.shape[0], 3)
    # print(np.asarray(sprite.vertex_normals))

    image = np.ones((100, 100, 3), dtype=np.float32) # dummy image
    image[20:50,50:90,:] = 0
    image = np.asarray(image, dtype=None, order='C')
    image = o3d.geometry.Image(image)
    sprite.textures = [image]
    cs.add_mesh(sprite)

    vertices = o3d.utility.Vector3dVector(np.asarray(pts[:5].copy(), dtype=None, order='C')) #np.zeros((5,3)))
    lines = o3d.utility.Vector2iVector(np.array(((0,1), (0,2), (0,3), (0,4), (1,2), (2,3), (3,4), (4,1))))
    line_set = o3d.geometry.LineSet(vertices, lines)
    line_set.colors = o3d.utility.Vector3dVector(np.array(((1,0,1),(1,0,1),(1,0,1),(1,0,1),(1,0,1),(1,0,1),(1,0,1),(1,0,1))))
    cs.add_mesh(line_set)

    return cs

def create_qr_code(mat44, length, qr_code_length, add_coordinate_system=True):
    if add_coordinate_system:
    # if True:
        cs = create_coordinate_system(identity_matrix44(), length=length, color_origin=[1,0,1])
    else:
        cs = MeshGroup()

    qr_code = o3d.geometry.TriangleMesh.create_box(qr_code_length, qr_code_length, 10*SCALE_UNIT)
    qr_code.translate([-qr_code_length/2,-qr_code_length/2,-10*SCALE_UNIT])
    qr_code.paint_uniform_color((0.3,0.3,0.3))
    qr_code.compute_vertex_normals()
    cs.add_mesh(qr_code)

    top_left_corner = o3d.geometry.TriangleMesh.create_box(qr_code_length/6, qr_code_length/6, 10*SCALE_UNIT)
    top_left_corner.translate([-qr_code_length/2 - 0.1*SCALE_UNIT,qr_code_length*2/6 - 0.1*SCALE_UNIT,-9*SCALE_UNIT]) # - 0.1 to avoid overlap with qr code
    top_left_corner.paint_uniform_color((0.,0.,0.))
    top_left_corner.compute_vertex_normals()
    cs.add_mesh(top_left_corner)

    cs.transform(mat44)
    return cs

def compute_qr_code_projected_points(mat_q_to_w, vec4_center_cam, qr_code_length):
    vec4_center = mul_mat44_vec4(mat_q_to_w, np.array([0,0,0,1]))
    vec4_x_axe = mul_mat44_vec4(mat_q_to_w, np.array([qr_code_length/2,0,0,1])) #- vec4_center
    # print(vec4_x_axe)
    vec4_y_axe = mul_mat44_vec4(mat_q_to_w, np.array([0,qr_code_length/2,0,1])) #- vec4_center
    # print(vec4_y_axe)
    vec4_z_axe = mul_mat44_vec4(mat_q_to_w, np.array([0,0,qr_code_length/2,1])) #- vec4_center
    # print(vec4_z_axe)

    if mat_q_to_w[0][0] == INVISIBLE_SCALE: # if default position then invisible
    # if True:
        projection_center = vec4_center_cam[:3]
        projection_x_axe = vec4_center_cam[:3]
        projection_y_axe = vec4_center_cam[:3]
        projection_z_axe = vec4_center_cam[:3]
    else:
        projection_center = vec4_center_cam[:3] + COEF_PROJECTION*(vec4_center[:3] - vec4_center_cam[:3])
        # print(f"projection_center {projection_center}")
        projection_x_axe = vec4_center_cam[:3] + COEF_PROJECTION*(vec4_x_axe[:3] - vec4_center_cam[:3])
        projection_y_axe = vec4_center_cam[:3] + COEF_PROJECTION*(vec4_y_axe[:3] - vec4_center_cam[:3])
        projection_z_axe = vec4_center_cam[:3] + COEF_PROJECTION*(vec4_z_axe[:3] - vec4_center_cam[:3])
        # projection_z_axe = COEF_PROJECTION*(vec4_z_axe[:3] - np.array((0,0,0)))

    return (projection_center, projection_x_axe, projection_y_axe, projection_z_axe)

def compute_projected_point(vec4_center_cam, point):
    # print(f"point {type(point)} {point}")
    if (np.array(point) == np.array([0,0,0])).all(): # if default position then invisible
        projection_point = vec4_center_cam[:3]
    else:
        projection_point = vec4_center_cam[:3] + COEF_PROJECTION*(point[:3] - vec4_center_cam[:3])
    return projection_point

def compute_projected_points(mat_qf_to_w, mat_ql_to_w, mat_qr_to_w, mat_qt_to_w, mat_c_to_w, qr_code_length, pos_t_q, pos_s_w, pos_s_o, pos_gt_s_o):
    vec4_center_cam = mul_mat44_vec4(mat_c_to_w, np.array([0,0,0,1]))

    qf_points = compute_qr_code_projected_points(mat_qf_to_w, vec4_center_cam, qr_code_length)
    ql_points = compute_qr_code_projected_points(mat_ql_to_w, vec4_center_cam, qr_code_length)
    qr_points = compute_qr_code_projected_points(mat_qr_to_w, vec4_center_cam, qr_code_length)
    qt_points = compute_qr_code_projected_points(mat_qt_to_w, vec4_center_cam, qr_code_length)

    pos_t_w = mul_mat44_vec4(mat_qf_to_w, vec3_to_vec4(pos_t_q))
    projection_tip = compute_projected_point(vec4_center_cam, pos_t_w[:3])
    projection_seed = compute_projected_point(vec4_center_cam, pos_s_w)
    projection_seed_optical = compute_projected_point(vec4_center_cam, pos_s_o)
    projection_seed_gt_optical = compute_projected_point(vec4_center_cam, pos_gt_s_o)

    return np.array((vec4_center_cam[:3], projection_tip, projection_seed, projection_seed_optical, projection_seed_gt_optical) + qf_points + ql_points + qr_points + qt_points)

def create_projected_lines(mat_qf_to_w, mat_ql_to_w, mat_qr_to_w, mat_qt_to_w, mat_c_to_w, qr_code_length, pos_t_q, pos_s_w, pos_s_o, pos_gt_s_o):
    cs = MeshGroup()
    points = o3d.utility.Vector3dVector(compute_projected_points(mat_qf_to_w, mat_ql_to_w, mat_qr_to_w, mat_qt_to_w, mat_c_to_w, qr_code_length, pos_t_q, pos_s_w, pos_s_o, pos_gt_s_o))
    lines = o3d.utility.Vector2iVector(np.array(((0,1), (0,2), (0,3), (0,4)
                                               , (0,5), (0,6), (0,7), (0,8)
                                               , (0,9), (0,10), (0,11), (0,12)
                                               , (0,13), (0,14), (0,15), (0,16)
                                               , (0,17), (0,18), (0,19), (0,20))))
    line_set = o3d.geometry.LineSet(points, lines)
    line_set.colors = o3d.utility.Vector3dVector(np.array(((1,0.5,0),(0.5,0,1),(0,0,1),(1,1,1)
                                                          ,(1,0,1),(1,0,0),(0,1,0),(0,0,1)
                                                          ,(1,0,1),(1,0,0),(0,1,0),(0,0,1)
                                                          ,(1,0,1),(1,0,0),(0,1,0),(0,0,1)
                                                          ,(1,0,1),(1,0,0),(0,1,0),(0,0,1))))
    cs.add_mesh(line_set)
    return cs

# rotation in degrees, shape (3,)
def create_probe(mat44, tip_color, probe_color, rotation, translation):
    cs = MeshGroup()
    cs.add_mesh(create_sphere([0,0,0], tip_color, OPTICAL_SPHERE_RADIUS/2))

    cone_height=10
    probe = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=10,cone_radius=10,cylinder_height=220 - cone_height,cone_height=cone_height)
    probe.paint_uniform_color(probe_color)
    probe.compute_vertex_normals()
    mat_rot = rotation_euler_matrix44(rotation, degrees=True)
    probe.transform(mat_rot)
    probe.translate(translation)
    cs.add_mesh(probe)

    cs.transform(mat44, internal=False, revert=False)

    return cs

def create_seed_holder(mat44, length):
    cs = create_coordinate_system(identity_matrix44(), length=length, color_origin=[1,1,1])

    seed_length = 5.25 # in mm, diameter 1.68 mm but 1.72 mm in the middle
    # x axis -> longitudinal seed, direction from the seed to the hole's exit
    # y axis -> width (30 mm) seed holder
    # z axis -> height (10 mm) seed holder, direction from the seed to the 5x3 divots plane
    length_holder = 250 # in mm
    width_holder = 30 # in mm
    height_holder = 10 # in mm

    border_x = 6 - seed_length/2

    holder = o3d.geometry.TriangleMesh.create_box(length_holder, 30, height_holder)
    holder.paint_uniform_color([0.8,0.8,0.8])
    holder.compute_vertex_normals()
    holder.translate((-length_holder + border_x, -width_holder + 15, -height_holder + 2))
    cs.add_mesh(holder)

    cs.transform(mat44)
    return cs

# check Polaris_Spectra_Tool_Kit_Guide.pdf for the position of the spheres
def create_optical_pointer_8700340(mat44, offset_tip_pointer, length):
    cs = create_coordinate_system(identity_matrix44(), length=length, color_origin=[1,1,0])
    cs.add_mesh(create_sphere([0, 0, 50], [0.2,0.2,0.2], OPTICAL_SPHERE_RADIUS))
    cs.add_mesh(create_sphere([0, 25, 100], [0.2,0.2,0.2], OPTICAL_SPHERE_RADIUS))
    cs.add_mesh(create_sphere([0, -25, 135], [0.2,0.2,0.2], OPTICAL_SPHERE_RADIUS))

    cone_height = 40
    cylinder_height = 142.9 + (156.7 - 135) - 5.22
    arrow = o3d.geometry.TriangleMesh.create_arrow(cylinder_radius=5,cone_radius=5,cylinder_height=cylinder_height - cone_height,cone_height=cone_height) # 5.22 was measure approximately on MD-8700340.pdf, page 3
    arrow.paint_uniform_color([0.2,0.2,0.2])
    arrow.compute_vertex_normals()
    mat_rot = rotation_euler_matrix44([180,0,0], degrees=True)
    arrow.transform(mat_rot)
    arrow.translate((0, 0, cylinder_height))
    # arrow.translate([-(9.55 + (8.77 - 0.64)), 0, 0]) # 9.55 was measure approximately on MD-8700340.pdf, page 3; 8.77 comes from Polaris_Spectra_Tool_Kit_Guide.pdf, page 3
    arrow.translate(offset_tip_pointer)
    cs.add_mesh(arrow)

    cs.transform(mat44)
    return cs

# check Polaris_Spectra_Tool_Kit_Guide.pdf for the position of the spheres
def create_optical_markers_8700339(mat44, length):
    cs = create_coordinate_system(identity_matrix44(), length=length, color_origin=[0,1,1])
    cs.add_mesh(create_sphere([0, 28.59, 41.02], [0.2,0.2,0.2], OPTICAL_SPHERE_RADIUS))
    cs.add_mesh(create_sphere([0, 0, 88], [0.2,0.2,0.2], OPTICAL_SPHERE_RADIUS))
    cs.add_mesh(create_sphere([0, -44.32, 40.45], [0.2,0.2,0.2], OPTICAL_SPHERE_RADIUS))

    cs.transform(mat44)
    return cs

# check Polaris_Spectra_Tool_Kit_Guide.pdf for the position of the spheres
def create_optical_markers_8700449(mat44, length):
    cs = create_coordinate_system(identity_matrix44(), length=length, color_origin=[1,0,1])
    cs.add_mesh(create_sphere([0, 47.38, 28.99], [0.2,0.2,0.2], OPTICAL_SPHERE_RADIUS))
    cs.add_mesh(create_sphere([0, 0, 89.1], [0.2,0.2,0.2], OPTICAL_SPHERE_RADIUS))
    cs.add_mesh(create_sphere([0, -35.36, 35.36], [0.2,0.2,0.2], OPTICAL_SPHERE_RADIUS))

    cs.transform(mat44)
    return cs
