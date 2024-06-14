import os
#os.environ['PYOPENGL_PLATFORM'] = 'egl'
import math
import trimesh
import pyrender
import numpy as np
from pyrender.constants import RenderFlags


class WeakPerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 scale,
                 translation,
                 znear=pyrender.camera.DEFAULT_Z_NEAR,
                 zfar=None,
                 name=None):
        super(WeakPerspectiveCamera, self).__init__(
            znear=znear,
            zfar=zfar,
            name=name,
        )
        self.scale = scale
        self.translation = translation

    def get_projection_matrix(self, width=None, height=None):
        P = np.eye(4)
        P[0, 0] = self.scale[0]
        P[1, 1] = self.scale[1]
        P[0, 3] = self.translation[0] * self.scale[0]
        P[1, 3] = -self.translation[1] * self.scale[1]
        P[2, 2] = -1
        return P


class PerspectiveCamera(pyrender.Camera):
    def __init__(self,
                 yfov=None,
                 aspectRatio=1):
        super(PerspectiveCamera, self).__init__(
            znear=0.05,
            zfar=100.0
        )
        self.znear = 0.05
        self.zfar = 100.0
        self.yfov = yfov
        self.aspectRatio = aspectRatio

    def get_projection_matrix(self, width=None, height=None):
        znear = 0.05
        zfar = 100.0
        fovX = self.yfov
        tanHalfFovY = math.tan((self.yfov / 2))
        tanHalfFovX = math.tan((fovX / 2))

        top = tanHalfFovY * znear
        bottom = -top
        right = tanHalfFovX * znear
        left = -right

        P = np.zeros((4, 4))

        z_sign = 1.0

        P[0, 0] = 2.0 * znear / (right - left)
        P[1, 1] = 2.0 * znear / (top - bottom)
        P[0, 2] = (right + left) / (right - left)
        P[1, 2] = (top + bottom) / (top - bottom)
        P[3, 2] = -z_sign
        P[2, 2] = z_sign * zfar / (znear-zfar)
        P[2, 3] = 2*(zfar * znear) / (znear-zfar)
        return P





class Renderer:
    def __init__(self, resolution=(224,224), orig_img=False, wireframe=False, faces=None):
        self.resolution = resolution

        self.faces = faces
        self.orig_img = orig_img
        self.wireframe = wireframe
        self.renderer = pyrender.OffscreenRenderer(
            viewport_width=self.resolution[0],
            viewport_height=self.resolution[1],
            point_size=1.0
        )

        # set the scene
        self.scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0], ambient_light=(0.3, 0.3, 0.3))

        # light = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=0.8)
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=0.8)

        light_pose = np.eye(4)
        light_pose[:3, 3] = [0, -1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [0, 1, 1]
        self.scene.add(light, pose=light_pose)

        light_pose[:3, 3] = [1, 1, 2]
        self.scene.add(light, pose=light_pose)

    def render(self, verts, cam=None, angle=None, axis=None, mesh_filename=None, color=[1.0, 1.0, 0.9], rotate=False):
        mesh = trimesh.Trimesh(vertices=verts, faces=self.faces, process=False)

        Rx = trimesh.transformations.rotation_matrix(math.radians(180), [1, 0, 0])
        mesh.apply_transform(Rx)

        if rotate:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(60), [0, 1, 0])
            mesh.apply_transform(rot)

        if mesh_filename is not None:
            mesh.export(mesh_filename)
        angle = 180
        axis = [1, 0, 0]
        if angle and axis:
            R = trimesh.transformations.rotation_matrix(math.radians(angle), axis)
            mesh.apply_transform(R)



        camera = PerspectiveCamera(yfov=0.5, aspectRatio=1)

        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            smooth=True,
            wireframe=True,
            roughnessFactor=1.0,
            emissiveFactor=(0.1, 0.1, 0.1),
            baseColorFactor=(color[0], color[1], color[2], 1.0)
        )

        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        mesh_node = self.scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = np.array([0, 0, 5])
        cam_node = self.scene.add(camera, pose=camera_pose)

        if self.wireframe:
            render_flags = RenderFlags.RGBA | RenderFlags.ALL_WIREFRAME
        else:
            render_flags = RenderFlags.RGBA

        rgb, _ = self.renderer.render(self.scene, flags=render_flags)
        # valid_mask = (rgb[:, :, -1] > 0)[:, :, np.newaxis]
        # #output_img = rgb[:, :, :-1] * valid_mask + (1 - valid_mask) * img
        # output_img = rgb * valid_mask + (1 - valid_mask) * img
        # image = output_img.astype(np.uint8)

        self.scene.remove_node(mesh_node)
        self.scene.remove_node(cam_node)

        return rgb
