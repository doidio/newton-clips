import newton
import tetgen
import trimesh
import warp as wp
from newton import ModelBuilder

import newtonclips

newtonclips.SAVE_DIR = '.clips'

cfg = ModelBuilder.ShapeConfig(mu=1e3)

builder = ModelBuilder('Z')

builder.add_ground_plane(cfg=cfg)

builder.add_shape_plane(
    body=-1,
    plane=(0.1, 0.0, 1.0, 1.0),
    width=5,
    length=5,
    cfg=cfg,
    key='shape_plane',
)

builder.add_shape_box(
    body=builder.add_body(),
    xform=(0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 1.0),
    hx=0.5,
    hy=0.2,
    hz=0.1,
    cfg=cfg,
    key='shape_box',
)

builder.add_shape_sphere(
    body=builder.add_body(),
    xform=(-1.0, 3.0, 2.0, 0.0, 0.0, 0.0, 1.0),
    radius=1.0,
    cfg=cfg,
    key='shape_sphere',
)

builder.add_shape_capsule(
    body=builder.add_body(),
    xform=(-1.0, -3.0, 2.0, 0.0, 0.0, 0.0, 1.0),
    radius=1.0,
    half_height=0.5,
    axis=1,
    cfg=cfg,
    key='shape_capsule',
)

mesh = trimesh.creation.torus(1.0, 0.5)
builder.add_shape_mesh(
    body=builder.add_body(),
    xform=(0, -3, 5, 0, 0, 0, 1),
    mesh=newton.Mesh(mesh.vertices, mesh.faces.flatten()),
    cfg=cfg,
    key='shape_mesh',
)

tet = tetgen.TetGen(mesh.vertices, mesh.faces)
vertices, indices = tet.tetrahedralize()

builder.add_soft_mesh(
    pos=(0, 3, 5),
    rot=(0.0, 0.0, 0.0, 1.0),
    vel=(0.0, 0.0, 0.0),
    scale=1.0,
    vertices=vertices.tolist(),
    indices=indices.flatten().tolist(),
    density=1e2,
    k_mu=5e4,
    k_lambda=5e4,
    k_damp=0.0,
    # key='soft_mesh',
)

dim, cell = 15, 0.1
center_x = dim * cell * 0.5
center_y = dim * cell * 0.5

mesh = trimesh.creation.icosphere(radius=0.5)
builder.add_cloth_grid(
    pos=(-center_x + 1.0, -center_y - 0.5, 4),
    dim_x=dim,
    dim_y=dim,
    cell_x=cell,
    cell_y=cell,
    mass=0.1,
    fix_bottom=True,
    rot=(0.0, 0.0, 0.0, 1.0),
    vel=(0.0, 0.0, 0.0),
    # key='cloth_grid',
)

dim, cell = 15, 0.1
center = dim * cell * 0.5

builder.add_soft_grid(
    pos=wp.vec3(-center, -center, 2.0),
    rot=wp.quat_identity(),
    vel=wp.vec3(),
    dim_x=dim,
    dim_y=dim,
    dim_z=dim,
    cell_x=cell,
    cell_y=cell,
    cell_z=cell,
    density=1e2,
    k_mu=5e4,
    k_lambda=5e4,
    k_damp=0.0,
    # key='soft_grid',
)

dim, cell = 10, 0.1
center = dim * cell * 0.5

builder.add_particle_grid(
    pos=wp.vec3(-center, -center, 5.0),
    rot=wp.quat_identity(),
    vel=wp.vec3(),
    dim_x=dim,
    dim_y=dim,
    dim_z=dim,
    cell_x=cell,
    cell_y=cell,
    cell_z=cell,
    mass=0.1,
    jitter=0.1,
)

builder.color()
model = builder.finalize()

rigid_solver = newton.solvers.SemiImplicitSolver(model)

state_0 = model.state()
state_1 = model.state()
control = model.control()

renderer = newtonclips.SimRenderer(model, enable_backface_culling=False)

fps = 60
frame_dt = 1.0 / fps
sim_substeps = 500
sim_dt = frame_dt / sim_substeps
sim_time = 0.0

for i in range(num_frames := 1500):
    for _ in range(sim_substeps):
        contacts = model.collide(state_0)
        state_0.clear_forces()
        state_1.clear_forces()

        rigid_solver.step(state_0, state_1, control, contacts, sim_dt)

        state_0, state_1 = state_1, state_0

    sim_time += frame_dt

    renderer.begin_frame(sim_time)
    renderer.render(state_0)
    renderer.end_frame()

renderer.save()
