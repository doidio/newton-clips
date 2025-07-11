import hashlib
import json
import os
import uuid
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton import Model
from newton.core.model import Vec3, Quat
from trimesh import Trimesh, geometry


class ModelBuilder(newton.ModelBuilder):
    SAVE_DIR: str | os.PathLike | None = None

    def __init__(self, up_vector=(0.0, 1.0, 0.0), gravity=-9.80665):
        super().__init__(up_vector, gravity)

        if ModelBuilder.SAVE_DIR is None:
            raise RuntimeError('Unset ModelBuilder.SAVE_DIR')

        self._save_dir = Path(ModelBuilder.SAVE_DIR)
        self._save_json = self._save_dir / 'model.json'
        self._cache_dir = self._save_dir / '.cache'

        self.save_dict = {
            'Uuid': uuid.uuid4().hex,
            'ShapeMesh': [],
            'SoftMesh': [],
        }

    def save(self):
        os.makedirs(self._save_json.parent, exist_ok=True)
        self._save_json.write_text(json.dumps(self.save_dict, indent=4), 'utf-8')

    def cache(self, hash_data: bytes | str) -> str | bytes | None:
        if isinstance(hash_data, bytes):
            sha1 = hashlib.sha1(hash_data).hexdigest()
            if not (f := (self._cache_dir / sha1)).exists():
                os.makedirs(f.parent, exist_ok=True)
                f.write_bytes(hash_data)
            return sha1
        elif isinstance(hash_data, str):
            if len(hash_data) and (f := (self._cache_dir / hash_data)).exists():
                return f.read_bytes()
            else:
                return bytes()
        return None

    def add_shape_mesh(
            self,
            body: int,
            mesh: newton.Mesh = None,
            pos: Vec3 = wp.vec3(0.0, 0.0, 0.0),
            rot: Quat = wp.quat(0.0, 0.0, 0.0, 1.0),
            scale: Vec3 = wp.vec3(1.0, 1.0, 1.0),
            density: float = newton.ModelBuilder.default_shape_density,
            ke: float = newton.ModelBuilder.default_shape_ke,
            kd: float = newton.ModelBuilder.default_shape_kd,
            kf: float = newton.ModelBuilder.default_shape_kf,
            ka: float = newton.ModelBuilder.default_shape_ka,
            mu: float = newton.ModelBuilder.default_shape_mu,
            restitution: float = newton.ModelBuilder.default_shape_restitution,
            is_solid: bool = True,
            thickness: float = newton.ModelBuilder.default_shape_thickness,
            has_ground_collision: bool = True,
            has_shape_collision: bool = True,
            collision_group: int = -1,
            is_visible: bool = True,
    ):
        super().add_shape_mesh(
            body=body,
            pos=pos,
            rot=rot,
            mesh=mesh,
            scale=scale,
            density=density,
            ke=ke,
            kd=kd,
            kf=kf,
            ka=ka,
            mu=mu,
            restitution=restitution,
            is_solid=is_solid,
            thickness=thickness,
            has_ground_collision=has_ground_collision,
            has_shape_collision=has_shape_collision,
            collision_group=collision_group,
            is_visible=is_visible,
        )

        if self.shape_geo_type[-1] != newton.GEO_MESH:
            raise RuntimeError('Only support GEO_MESH')

        mesh = Trimesh(
            np.array(self.shape_geo_src[-1].vertices).reshape(-1, 3),
            np.array(self.shape_geo_src[-1].indices).reshape(-1, 3),
        )
        mesh.fix_normals()

        vertex_normals = geometry.weighted_vertex_normals(
            vertex_count=len(mesh.vertices),
            faces=mesh.faces,
            face_normals=mesh.face_normals,
            face_angles=mesh.face_angles,
        )

        self.save_dict['ShapeMesh'].append({
            'Body': self.shape_body[-1],
            'Transform': tuple(self.shape_transform[-1]),
            'Scale': tuple(self.shape_geo_scale[-1]),
            'Vertices': self.cache(mesh.vertices.flatten().astype(np.float32).tobytes()),
            'Indices': self.cache(mesh.faces.flatten().astype(np.int32).tobytes()),
            'VertexNormals': self.cache(vertex_normals.flatten().astype(np.float32).tobytes()),
            'VertexUVs': '',
        })

    def update_shape_mesh(self, i: int = -1, VertexUVs: np.ndarray[(-1, 2), np.float32] | None = None):
        if VertexUVs is not None:
            self.save_dict['ShapeMesh'][i].update({
                'VertexUVs': self.cache(np.array(VertexUVs, np.float32).reshape(-1, 2).flatten().tobytes()),
            })

    def add_soft_mesh(
            self,
            vertices: list[Vec3],
            indices: list[int],
            pos: Vec3 = wp.vec3(0.0, 0.0, 0.0),
            rot: Quat = wp.quat(0.0, 0.0, 0.0, 1.0),
            scale: float = 1.0,
            vel: Vec3 = wp.vec3(0.0, 0.0, 0.0),
            density: float = newton.ModelBuilder.default_shape_density,
            k_mu: float = 1.0e3,
            k_lambda: float = 1.0e3,
            k_damp: float = 0.0,
            tri_ke: float = newton.ModelBuilder.default_tri_ke,
            tri_ka: float = newton.ModelBuilder.default_tri_ka,
            tri_kd: float = newton.ModelBuilder.default_tri_kd,
            tri_drag: float = newton.ModelBuilder.default_tri_drag,
            tri_lift: float = newton.ModelBuilder.default_tri_lift,
    ):
        begin = len(self.particle_q)

        super().add_soft_mesh(
            pos=pos,
            rot=rot,
            scale=scale,
            vel=vel,
            vertices=[wp.vec3(_) for _ in vertices],
            indices=indices,
            density=density,
            k_mu=k_mu,
            k_lambda=k_lambda,
            k_damp=k_damp,
            tri_ke=tri_ke,
            tri_ka=tri_ka,
            tri_kd=tri_kd,
            tri_drag=tri_drag,
            tri_lift=tri_lift,
        )

        count = len(self.particle_q) - begin

        if count <= 0:
            raise RuntimeError(f'Invalid begin {begin} count {count}')

        a, b = begin, begin + count

        mesh = Trimesh(
            np.array(self.particle_q).reshape(-1, 3)[a:b],
            np.array(self.tri_indices).reshape(-1, 3)[a:b],
        )
        mesh.fix_normals()

        vertex_normals = geometry.weighted_vertex_normals(
            vertex_count=len(mesh.vertices),
            faces=mesh.faces,
            face_normals=mesh.face_normals,
            face_angles=mesh.face_angles,
        )

        self.save_dict['SoftMesh'].append({
            'Begin': begin,
            'Count': count,
            'Vertices': self.cache(mesh.vertices.flatten().astype(np.float32).tobytes()),
            'Indices': self.cache(mesh.faces.flatten().astype(np.int32).tobytes()),
            'VertexNormals': self.cache(vertex_normals.flatten().astype(np.float32).tobytes()),
            'VertexUVs': '',
        })

    def update_soft_mesh(self, i: int = -1, VertexUVs: np.ndarray[(-1, 2), np.float32] | None = None):
        if VertexUVs is not None:
            self.save_dict['SoftMesh'][i].update({
                'VertexUVs': self.cache(np.array(VertexUVs, np.float32).reshape(-1, 2).flatten().tobytes()),
            })

    def finalize(self, device=None, requires_grad=False) -> Model:
        self.save()
        return super().finalize(device, requires_grad)
