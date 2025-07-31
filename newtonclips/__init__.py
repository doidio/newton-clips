import hashlib
import json
import os
import shutil
import uuid
import warnings
from datetime import datetime
from pathlib import Path

import newton
import newton.utils.render
import numpy as np
import trimesh
from trimesh.graph import connected_components
from trimesh.transformations import rotation_matrix
from warp.render.utils import solidify_mesh

SAVE_DIR: str
SCALING: float = 1.0


def save_record(fps: int, name: str | None = None):
    try:
        save_dir = Path(SAVE_DIR)
    except NameError:
        raise Exception('You should set newtonclips.SAVE_DIR')

    from moviepy import ImageSequenceClip
    files = []
    while (f := save_dir / 'record' / f'{len(files) + 1}.png').exists():
        files.append(f)

    if len(files):
        if name is None:
            file = save_dir / f'{datetime.now().strftime('%Y%m%d%H%M%S')}.mp4'
        else:
            if not name.endswith('.mp4'):
                name += '.mp4'
            file = save_dir / f'{name}'
        clip = ImageSequenceClip([f.as_posix() for f in files], fps=fps)
        clip.write_videofile(file, audio=False, ffmpeg_params=['-crf', '1'])


class SimRenderer:
    def __init__(self, model: newton.Model):
        try:
            self._save_dir = Path(SAVE_DIR)
        except NameError:
            raise Exception('You should set newtonclips.SAVE_DIR before SimRenderer init')

        self._cache_dir = self._save_dir / '.cache'
        self._model_json = self._save_dir / 'model.json'
        self._frame_dir = self._save_dir / 'frame'

        self._model = model
        self._model_dict = {
            'Uuid': uuid.uuid4().hex,
            'Scale': SCALING,
            'UpAxis': model.up_axis,
            'ShapeMesh': [],
            'SoftMesh': [],
            'GranularFluid': [],
        }

        self._frames = []

        self.sim_time = 0.0
        self.delta_time = 0.0

        if model.shape_count > 0:
            a_body = model.shape_body.numpy()
            a_type = model.shape_geo.type.numpy()
            a_scale = model.shape_geo.scale.numpy()
            a_thickness = model.shape_geo.thickness.numpy()
            a_is_solid = model.shape_geo.is_solid.numpy()
            a_transform = model.shape_transform.numpy()
            a_flags = model.shape_flags.numpy()

            for i in range(model.shape_count):
                key, src = model.shape_key[i], model.shape_geo_src[i]
                body, ty, scale, th, is_solid, transform, flag = (
                    a_body[i], a_type[i], a_scale[i], a_thickness[i], a_is_solid[i], a_transform[i], a_flags[i],
                )

                if ty == newton.GEO_PLANE:
                    w = scale[0] if scale[0] > 0.0 else 100.0
                    l = scale[1] if scale[1] > 0.0 else 100.0

                    mesh = trimesh.Trimesh(
                        np.array([[-w, -l, 0.0], [w, -l, 0.0], [w, l, 0.0], [-w, l, 0.0]]),
                        np.array([[0, 1, 2], [0, 2, 3]]),
                        process=False,
                    )

                elif ty == newton.GEO_SPHERE:
                    mesh = trimesh.creation.icosphere(radius=scale[0])

                elif ty == newton.GEO_CAPSULE:
                    mesh = trimesh.creation.capsule(radius=scale[0], height=scale[1] * 2)
                    mesh = mesh.apply_transform(rotation_matrix(np.deg2rad(90), [0, 0, 1]))

                elif ty == newton.GEO_CYLINDER:
                    warnings.warn('Newton does not support collision for GEO_CYLINDER')
                    mesh = trimesh.creation.cylinder(radius=scale[0], height=scale[1] * 2)
                    mesh = mesh.apply_transform(rotation_matrix(np.deg2rad(90), [0, 0, 1]))

                elif ty == newton.GEO_CONE:
                    warnings.warn('Newton does not support collision for GEO_CONE')
                    mesh = trimesh.creation.cone(radius=scale[0], height=scale[1] * 2)
                    mesh = mesh.apply_transform(rotation_matrix(np.deg2rad(90), [0, 0, 1]))

                elif ty == newton.GEO_BOX:
                    mesh = trimesh.creation.box(extents=[scale[0] * 2, scale[1] * 2, scale[2] * 2])

                elif ty == newton.GEO_MESH:
                    if not is_solid:
                        faces, vertices = solidify_mesh(src.indices, src.vertices, th)
                    else:
                        faces, vertices = src.indices, src.vertices

                    mesh = trimesh.Trimesh(vertices.reshape(-1, 3), faces.reshape(-1, 3), process=False)

                elif ty == newton.GEO_SDF:
                    warnings.warn('Newton does not support collision for GEO_SDF')
                    warnings.warn('Not implemented GEO_SDF')
                    continue
                else:
                    continue

                self._model_dict['ShapeMesh'].append({
                    'Name': f'SH_{key}_{body}',
                    'Body': int(body),
                    'Transform': tuple(float(_) for _ in transform),
                    'Vertices': self.cache(mesh.vertices.flatten().astype(np.float32).tobytes()),
                    'Indices': self.cache(mesh.faces.flatten().astype(np.int32).tobytes()),
                })

        if model.particle_count > 0:
            particle_q = model.particle_q.numpy()

            # soft triangles
            if model.tri_indices is not None and len(model.tri_indices):
                tri_indices = model.tri_indices.numpy()

                tri_mesh = trimesh.Trimesh(particle_q, tri_indices, process=False)

                components = connected_components(
                    edges=tri_mesh.face_adjacency, nodes=np.arange(len(tri_mesh.faces)),
                )
                for face_idx in components:
                    faces = tri_mesh.faces[face_idx]
                    begin, end = np.min(faces), np.max(faces) + 1
                    count = end - begin
                    mesh = trimesh.Trimesh(tri_mesh.vertices[begin:end], faces - begin, process=False)

                    self._model_dict['SoftMesh'].append({
                        'Name': f'SF_{begin}_{count}',
                        'Begin': int(begin),
                        'Count': int(count),
                        'Vertices': self.cache(mesh.vertices.flatten().astype(np.float32).tobytes()),
                        'Indices': self.cache(mesh.faces.flatten().astype(np.int32).tobytes()),
                    })

            # granular & fluid, ignore tet, edge, spring
            indices = set()
            for _ in (model.tri_indices, model.tet_indices, model.edge_indices, model.spring_indices):
                if _ is not None:
                    indices.update(_.numpy().flatten())
            granular = sorted(set(range(model.particle_count)) - indices)

            isolated = np.array(granular, dtype=int)
            breaks = np.where(np.diff(isolated) > 1)[0] + 1
            for seg in np.split(isolated, breaks):
                begin, end = np.min(seg), np.max(seg) + 1
                count = end - begin
                particles = particle_q[begin:end]

                self._model_dict['GranularFluid'].append({
                    'Name': f'GF_{begin}_{count}',
                    'Begin': int(begin),
                    'Count': int(count),
                    'ParticlePositions': self.cache(particles.flatten().astype(np.float32).tobytes()),
                })

        shutil.rmtree(self._frame_dir, ignore_errors=True)
        os.makedirs(self._model_json.parent, exist_ok=True)
        self._model_json.write_text(json.dumps(self._model_dict, indent=4, ensure_ascii=False), 'utf-8')
        print(f'[newtonclips.SAVE_DIR] {self._model_json.parent.absolute()}')

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
        raise TypeError(f'Invalid cache {hash_data}')

    def begin_frame(self, sim_time: float):
        self.delta_time = sim_time - self.sim_time
        self.sim_time = sim_time

        self._frames.append({
            'DeltaTime': self.delta_time,
            'BodyTransforms': '',
            'ParticlePositions': '',
            'ShapeVertexHues': {},
            'ParticleHues': {},
        })

    def render(self, state: newton.State):
        body_q = state.body_q.numpy() if state.body_q is not None else []
        particle_q = state.particle_q.numpy() if state.particle_q is not None else []

        self._frames[-1]['BodyTransforms'] = self.cache(
            np.array(body_q, np.float32).reshape(-1, 7).flatten().tobytes()
        )
        self._frames[-1]['ParticlePositions'] = self.cache(
            np.array(particle_q, np.float32).reshape(-1, 3).flatten().tobytes()
        )

        body_qd = state.body_qd.numpy()
        shape_body = self._model.shape_body.numpy()

        for i in range(self._model.shape_count):
            body = shape_body[i]

            if body > -1:
                v = np.linalg.norm(body_qd[body][-3:])
                qd = 210 * 1.0 / (v + 1.0)
            else:
                qd = 210

            self.render_shape_vertex_hue(i, qd)

        particle_qd = state.particle_qd.numpy()
        v = np.linalg.norm(particle_qd, axis=1)
        self.render_particle_hue(210 * 1.0 / (v + 1.0))

    def render_shape_vertex_hue(self, i, hues):
        self._frames[-1]['ShapeVertexHues'][i] = self.cache(
            np.array(hues, np.float32).flatten().tobytes()
        )

    def render_particle_hue(self, hues):
        for item in (*self._model_dict['SoftMesh'], *self._model_dict['GranularFluid']):
            begin, count = item['Begin'], item['Count']
            self._frames[-1]['ParticleHues'][begin] = self.cache(
                np.array(hues[begin:begin + count], np.float32).flatten().tobytes()
            )

    def end_frame(self):
        os.makedirs(self._frame_dir, exist_ok=True)
        frame_json = self._frame_dir / f'{len(self._frames) - 1}.json'
        frame_json.write_text(json.dumps(self._frames[-1], indent=4, ensure_ascii=False), 'utf-8')

    def save(self):
        """"""


def _CreateSimRenderer(Super, no_super=False):
    class Renderer(SimRenderer, Super):
        def __init__(self, model, *args, **kwargs):
            SimRenderer.__init__(self, model)
            if no_super:
                Super.__init__(self, model, *args, **kwargs)

        def begin_frame(self, sim_time: float):
            SimRenderer.begin_frame(self, sim_time)
            if no_super:
                Super.begin_frame(self, sim_time)

        def render(self, state: newton.State):
            SimRenderer.render(self, state)
            if no_super:
                Super.render(self, state)

        def render_shape_vertex_hue(self, i, hues):
            SimRenderer.render_shape_vertex_hue(self, i, hues)

        def render_particle_hue(self, hues):
            SimRenderer.render_particle_hue(self, hues)

        def end_frame(self):
            SimRenderer.end_frame(self)
            if no_super:
                Super.end_frame(self)

        def save(self):
            SimRenderer.save(self)
            if no_super:
                Super.save(self)

    return Renderer


SimRendererOpenGL = _CreateSimRenderer(newton.utils.SimRendererOpenGL)
newton.utils.SimRendererOpenGL = _CreateSimRenderer(newton.utils.SimRendererOpenGL, no_super=True)

SimRendererUsd = _CreateSimRenderer(newton.utils.SimRendererUsd)
newton.utils.SimRendererUsd = _CreateSimRenderer(newton.utils.SimRendererUsd, no_super=True)

newton.utils.SimRenderer = newton.utils.SimRendererOpenGL
