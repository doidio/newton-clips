import hashlib
import json
import os
from pathlib import Path

import newton
import numpy as np
import warp as wp
from newton import Model, Mesh, State
from newton.core.types import Vec3, Quat, Transform, AxisType, Axis
from trimesh import Trimesh, geometry


class ModelBuilder(newton.ModelBuilder):
    SAVE_DIR: str | os.PathLike | None = None
    SCALE: float = 1.0

    def __init__(self, up_vector: AxisType = Axis.Z, gravity: float = -9.81):
        super().__init__(up_vector, gravity)

        if self.SAVE_DIR is None:
            raise RuntimeError('Unset SAVE_DIR')

        self._save_dir = Path(self.SAVE_DIR)
        self._cache_dir = self._save_dir / '.cache'
        self._model_json = self._save_dir / 'model.json'
        self._frame_dir = self._save_dir / 'frames'

        self._model_dict = {
            'Sha1': '',
            'Scale': self.SCALE,
            'ShapeMesh': [],
            'SoftMesh': [],
        }

        self._frames = []

        self.sim_time = 0.0
        self.delta_time = 0.0

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
            xform: Transform | None = None,
            mesh: Mesh | None = None,
            scale: Vec3 | None = None,
            cfg: newton.ModelBuilder.ShapeConfig | None = None,
            key: str | None = None,
    ):
        super().add_shape_mesh(
            body=body,
            mesh=mesh,
            scale=scale,
            cfg=cfg,
            key=key,
        )

        if self.shape_geo_type[-1] != newton.GEO_MESH:
            raise RuntimeError('Only support GEO_MESH')

        mesh = Trimesh(
            np.array(self.shape_geo_src[-1].vertices).reshape(-1, 3),
            np.array(self.shape_geo_src[-1].indices).reshape(-1, 3),
            process=False,
        )
        mesh.fix_normals()

        vertex_normals = geometry.weighted_vertex_normals(
            vertex_count=len(mesh.vertices),
            faces=mesh.faces,
            face_normals=mesh.face_normals,
            face_angles=mesh.face_angles,
        )

        self._model_dict['ShapeMesh'].append({
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
            self._model_dict['ShapeMesh'][i].update({
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
            density: float = 1.0,
            k_mu: float = 1.0e3,
            k_lambda: float = 1.0e3,
            k_damp: float = 0.0,
            tri_ke: float = 100.0,
            tri_ka: float = 100.0,
            tri_kd: float = 10.0,
            tri_drag: float = 0.0,
            tri_lift: float = 0.0,
    ):
        vtx_begin = len(self.particle_q)
        tri_begin = len(self.tri_indices)

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

        vtx_count = len(self.particle_q) - vtx_begin
        tri_count = len(self.tri_indices) - tri_begin

        if vtx_count <= 0:
            raise RuntimeError(f'Invalid begin {vtx_begin} count {vtx_count}')

        mesh = Trimesh(
            np.array(self.particle_q).reshape(-1, 3)[vtx_begin:vtx_begin + vtx_count],
            np.array(self.tri_indices).reshape(-1, 3)[tri_begin:tri_begin + tri_count] - vtx_begin,
            process=False,
        )
        mesh.fix_normals()

        vertex_normals = geometry.weighted_vertex_normals(
            vertex_count=len(mesh.vertices),
            faces=mesh.faces,
            face_normals=mesh.face_normals,
            face_angles=mesh.face_angles,
        )

        self._model_dict['SoftMesh'].append({
            'Begin': vtx_begin,
            'Count': vtx_count,
            'Vertices': self.cache(mesh.vertices.flatten().astype(np.float32).tobytes()),
            'Indices': self.cache(mesh.faces.flatten().astype(np.int32).tobytes()),
            'VertexNormals': self.cache(vertex_normals.flatten().astype(np.float32).tobytes()),
            'VertexUVs': '',
        })

    def update_soft_mesh(self, i: int = -1, VertexUVs: np.ndarray[(-1, 2), np.float32] | None = None):
        if VertexUVs is not None:
            self._model_dict['SoftMesh'][i].update({
                'VertexUVs': self.cache(np.array(VertexUVs, np.float32).reshape(-1, 2).flatten().tobytes()),
            })

    def finalize(self, device=None, requires_grad=False) -> Model:
        json_str = json.dumps(self._model_dict, sort_keys=True, ensure_ascii=False)
        self._model_dict['Sha1'] = hashlib.sha1(json_str.encode('utf-8')).hexdigest()
        os.makedirs(self._model_json.parent, exist_ok=True)
        self._model_json.write_text(json.dumps(self._model_dict, indent=4, ensure_ascii=False), 'utf-8')

        return super().finalize(device, requires_grad)

    @property
    def renderer(self):
        return self

    def begin_frame(self, sim_time: float):
        self.delta_time = sim_time - self.sim_time
        self.sim_time = sim_time

    def render(self, state: State):
        body_q = state.body_q.numpy() if state.body_q is not None else []
        particle_q = state.particle_q.numpy() if state.particle_q is not None else []

        frame = {
            'DeltaTime': self.delta_time,
            'BodyTransform': self.cache(np.array(body_q, np.float32).reshape(-1, 7).flatten().tobytes()),
            'ParticlePosition': self.cache(np.array(particle_q, np.float32).reshape(-1, 3).flatten().tobytes()),
        }

        os.makedirs(self._frame_dir, exist_ok=True)

        frame_json = self._frame_dir / f'{len(self._frames)}.json'
        frame_json.write_text(json.dumps(frame, indent=4, ensure_ascii=False), 'utf-8')

        self._frames.append(frame)

    def end_frame(self):
        """"""


def tri_surface_from_tet(tetrahedrons: np.ndarray) -> np.ndarray:
    # 生成所有四面体的四个面，每个面顶点排序
    faces_indices = np.array([[0, 2, 1], [1, 2, 3], [0, 1, 3], [0, 3, 2]])
    faces = tetrahedrons[:, faces_indices].reshape(-1, 3)
    sorted_faces = np.sort(faces, axis=1)

    # 按字典序排序所有面
    sort_order = np.lexsort((sorted_faces[:, 2], sorted_faces[:, 1], sorted_faces[:, 0]))
    sorted_faces = sorted_faces[sort_order]

    # 确定各面组及其出现次数
    diff = np.ones(len(sorted_faces), dtype=bool)
    diff[1:] = np.any(sorted_faces[1:] != sorted_faces[:-1], axis=1)
    group_starts = np.where(diff)[0]
    group_ends = np.append(group_starts[1:], len(sorted_faces))
    group_lengths = group_ends - group_starts

    # 提取仅出现一次的面
    surface_mask = (group_lengths == 1)
    surface_faces = sorted_faces[group_starts[surface_mask]]

    return surface_faces
