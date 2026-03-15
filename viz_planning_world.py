"""Export mplib planning-world collision meshes to .glb files for inspection.

Usage:
    from viz_planning_world import save_planning_world

    planner.update_from_simulation()
    save_planning_world(planner.planning_world, "output_dir/stage_name")
"""
import os
import numpy as np
import trimesh
from mplib.collision_detection.fcl import (
    BVHModel, Box, Capsule, Convex, Cylinder, Halfspace, Sphere,
)

# Colours: robot links get blue, env objects get orange
_ROBOT_COLOR = [100, 149, 237, 180]   # cornflower blue, semi-transparent
_OBJECT_COLOR = [255, 165, 0, 180]    # orange, semi-transparent


def _pose_to_matrix(pose):
    """Convert an mplib Pose to a 4x4 homogeneous matrix."""
    mat = np.eye(4)
    p, q = np.array(pose.p), np.array(pose.q)  # q = [w, x, y, z]
    # quaternion to rotation matrix (w, x, y, z)
    w, x, y, z = q
    mat[:3, :3] = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - z*w),     2*(x*z + y*w)],
        [    2*(x*y + z*w), 1 - 2*(x*x + z*z),     2*(y*z - x*w)],
        [    2*(x*z - y*w),     2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ])
    mat[:3, 3] = p
    return mat


def _unflatten_convex_faces(flat_faces):
    """Convex.get_faces() returns [n, i0, i1, ..., n, i0, ...].

    Convert to Nx3 array of triangle indices (only handles triangular faces).
    """
    faces = []
    i = 0
    flat = list(flat_faces)
    while i < len(flat):
        n = flat[i]
        i += 1
        if n == 3:
            faces.append(flat[i:i+3])
        elif n > 3:
            # Fan-triangulate
            v0 = flat[i]
            for j in range(1, n - 1):
                faces.append([v0, flat[i + j], flat[i + j + 1]])
        i += n
    return np.array(faces, dtype=np.int32) if faces else np.zeros((0, 3), dtype=np.int32)


def _geom_to_trimesh(geom):
    """Convert a single CollisionGeometry to a trimesh object (at origin)."""
    if isinstance(geom, BVHModel):
        verts = np.array(geom.get_vertices())
        tri_list = geom.get_faces()  # list[Triangle]
        faces = np.array([[t[0], t[1], t[2]] for t in tri_list], dtype=np.int32)
        if len(verts) == 0 or len(faces) == 0:
            return None
        return trimesh.Trimesh(vertices=verts, faces=faces)

    if isinstance(geom, Convex):
        verts = np.array(geom.get_vertices())
        faces = _unflatten_convex_faces(geom.get_faces())
        if len(verts) == 0 or len(faces) == 0:
            return None
        return trimesh.Trimesh(vertices=verts, faces=faces)

    if isinstance(geom, Box):
        return trimesh.creation.box(extents=np.array(geom.side))

    if isinstance(geom, Sphere):
        return trimesh.creation.icosphere(radius=geom.radius)

    if isinstance(geom, Cylinder):
        return trimesh.creation.cylinder(radius=geom.radius, height=geom.lz)

    if isinstance(geom, Capsule):
        return trimesh.creation.capsule(radius=geom.radius, height=geom.lz)

    if isinstance(geom, Halfspace):
        # Represent as a large thin box (infinite plane can't be meshed)
        return trimesh.creation.box(extents=[4.0, 4.0, 0.002])

    return None


def _collect_robot_meshes(pw):
    """Yield (name, trimesh) for every collision shape on every robot link."""
    for art_name in pw.get_articulation_names():
        art = pw.get_articulation(art_name)
        fcl_model = art.get_fcl_model()
        link_names = fcl_model.get_collision_link_names()
        col_objs = fcl_model.get_collision_objects()

        for link_name, fcl_obj in zip(link_names, col_objs):
            # col_objs from FCLModel are FCLObjects (one per link)
            obj_pose = _pose_to_matrix(fcl_obj.pose)
            for shape, shape_pose in zip(fcl_obj.shapes, fcl_obj.shape_poses):
                geom = shape.get_collision_geometry()
                mesh = _geom_to_trimesh(geom)
                if mesh is None:
                    continue
                local_mat = _pose_to_matrix(shape_pose)
                mesh.apply_transform(obj_pose @ local_mat)
                yield f"{art_name}/{link_name}", mesh


def _collect_object_meshes(pw):
    """Yield (name, trimesh) for every collision shape on scene objects."""
    for obj_name in pw.get_object_names():
        fcl_obj = pw.get_object(obj_name)
        obj_pose = _pose_to_matrix(fcl_obj.pose)
        for shape, shape_pose in zip(fcl_obj.shapes, fcl_obj.shape_poses):
            geom = shape.get_collision_geometry()
            mesh = _geom_to_trimesh(geom)
            if mesh is None:
                continue
            local_mat = _pose_to_matrix(shape_pose)
            mesh.apply_transform(obj_pose @ local_mat)
            yield obj_name, mesh


def save_planning_world(pw, path_stem, fmt="glb"):
    """Save the full planning world collision geometry to a file.

    Args:
        pw: mplib PlanningWorld (or SapienPlanningWorld).
        path_stem: output path without extension, e.g. "output/01_pregrasp".
        fmt: file format — "glb", "obj", "stl", or "ply".
    """
    scene = trimesh.Scene()

    for name, mesh in _collect_robot_meshes(pw):
        mesh.visual.face_colors = _ROBOT_COLOR
        scene.add_geometry(mesh, node_name=name)

    for name, mesh in _collect_object_meshes(pw):
        mesh.visual.face_colors = _OBJECT_COLOR
        scene.add_geometry(mesh, node_name=name)

    out_path = f"{path_stem}.{fmt}"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    scene.export(out_path)
    n_geom = len(scene.geometry)
    print(f"  [viz] Saved {n_geom} collision meshes → {out_path}")
    return out_path
