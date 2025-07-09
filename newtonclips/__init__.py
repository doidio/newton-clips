import hashlib
import json
import os
from pathlib import Path

import newton
import numpy as np


class Save:
    def __init__(self, save_dir: str | os.PathLike):
        self._save_dir = Path(save_dir)
        self._save_json = self._save_dir / 'save.json'
        self._cache_dir = self._save_dir / '.cache'

        if self._save_json.exists():
            self.save_dict = json.loads(self._save_json.read_text('utf-8'))
            assert isinstance(self.save_dict['Model'], dict)
            assert isinstance(self.save_dict['Model']['ShapeMesh'], list)
            assert isinstance(self.save_dict['Model']['SoftMesh'], list)
            assert isinstance(self.save_dict['State'], list)
        else:
            self.save_dict = {
                'Model': {
                    'ShapeMesh': [],
                    'SoftMesh': [],
                },
                'State': [],
            }
            self.save()

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


class ModelShapeMesh:
    def __init__(self, save: Save, builder: newton.ModelBuilder):
        self._save = save
        self._builder = builder

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._builder.shape_geo_type[-1] != newton.GEO_MESH:
            raise RuntimeError('Only support GEO_MESH')

        self._save.save_dict['Model']['ShapeMesh'].append({
            'body': self._builder.shape_body[-1],
            'transform': tuple(self._builder.shape_transform[-1]),
            'scale': tuple(self._builder.shape_geo_scale[-1]),
            'vertices': self._save.cache(np.array(
                self._builder.shape_geo_src[-1].vertices).flatten().astype(np.float32).tobytes()),
            'indices': self._save.cache(np.array(
                self._builder.shape_geo_src[-1].indices).flatten().astype(np.int32).tobytes()),
            'is_solid': self._builder.shape_geo_src[-1].is_solid,
        })
        self._save.save()


class ModelSoftMesh:
    def __init__(self, save: Save, builder: newton.ModelBuilder):
        self._save = save
        self._builder = builder
        self._init = 0

    def __enter__(self):
        self._init = len(self._builder.particle_q)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        count = len(self._builder.particle_q) - self._init

        if count <= 0:
            raise RuntimeError(f'Invalid init {self._init} count {count}')

        a, b = self._init, self._init + count

        self._save.save_dict['Model']['ShapeMesh'].append({
            'init': self._init,
            'count': count,
            'vertices': self._save.cache(np.array(
                self._builder.particle_q)[a:b].flatten().astype(np.float32).tobytes()),
            'indices': self._save.cache(np.array(
                self._builder.tri_indices)[a:b].flatten().astype(np.int32).tobytes()),
        })
        self._save.save()
