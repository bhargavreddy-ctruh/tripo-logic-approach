#!/usr/bin/env python3
"""
Single-Mesh Unified Exporter:
Combines the retopo quad shell and the feature rails into EXACTLY ONE SINGLE MESH
(single object node, single draw call, unified vertex & normal buffer).
"""

import os
import sys
import numpy as np
import trimesh

def export_single_mesh(quad_obj_path, rails_obj_path, out_obj_path, out_glb_path):
    print(f"[*] Creating Single Unified Mesh: {out_obj_path}")
    
    # 1. Parse Quad Shell
    v_quad = []
    f_quad = []
    with open(quad_obj_path) as f:
        for line in f:
            if line.startswith('o Outline_Rails'):
                break
            if line.startswith('v '):
                parts = line.split()
                v_quad.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.split()[1:]
                f_quad.append([int(p.split('/')[0]) - 1 for p in parts])

    v_quad = np.array(v_quad, dtype=np.float64)

    # Compute quad normals
    tri_faces = []
    for f in f_quad:
        if len(f) == 4:
            tri_faces.append([f[0], f[1], f[2]])
            tri_faces.append([f[0], f[2], f[3]])
        elif len(f) == 3:
            tri_faces.append(f)

    quad_tri_mesh = trimesh.Trimesh(vertices=v_quad, faces=np.array(tri_faces, dtype=np.int64), process=False)
    vn_quad = quad_tri_mesh.vertex_normals

    # 2. Parse Rails
    rails_mesh = trimesh.load(rails_obj_path, process=False)
    if isinstance(rails_mesh, trimesh.Scene):
        rails_mesh = trimesh.util.concatenate(rails_mesh.dump())

    v_rails = rails_mesh.vertices
    vn_rails = rails_mesh.vertex_normals
    f_rails = rails_mesh.faces

    # 3. Merge into 1 single unified mesh buffer
    v_all = np.vstack([v_quad, v_rails])
    vn_all = np.vstack([vn_quad, vn_rails])
    offset = len(v_quad)
    f_rails_offset = [[idx + offset for idx in f] for f in f_rails]
    f_all = f_quad + f_rails_offset

    quad_count = sum(1 for f in f_all if len(f) == 4)
    tri_count = sum(1 for f in f_all if len(f) == 3)
    total_faces = len(f_all)

    # 4. Write Single-Mesh OBJ (1 Object, 1 Draw Call)
    os.makedirs(os.path.dirname(os.path.abspath(out_obj_path)), exist_ok=True)
    with open(out_obj_path, 'w') as f_out:
        f_out.write(f"# Unified Single-Mesh Retopo (1 Object Node, 1 Draw Call)\n")
        f_out.write(f"# Total Vertices: {len(v_all):,} | Total Faces: {total_faces:,}\n")
        f_out.write("o Unified_Model\n")
        for p in v_all:
            f_out.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
        for vn in vn_all:
            f_out.write(f"vn {vn[0]:.6f} {vn[1]:.6f} {vn[2]:.6f}\n")
        f_out.write("s 1\n")
        for f in f_all:
            f_out.write("f " + " ".join(f"{idx + 1}//{idx + 1}" for idx in f) + "\n")
    print(f"[+] Saved Single-Mesh OBJ to: {out_obj_path} ({len(v_all):,} verts, {total_faces:,} faces)")

    # 5. Write Single-Mesh GLB (1 Single Geometry Node)
    tri_all = []
    for f in f_all:
        if len(f) == 4:
            tri_all.append([f[0], f[1], f[2]])
            tri_all.append([f[0], f[2], f[3]])
        elif len(f) == 3:
            tri_all.append(f)

    unified_trimesh = trimesh.Trimesh(
        vertices=v_all,
        faces=np.array(tri_all, dtype=np.int64),
        process=False
    )
    # Neutral clean shader
    colors = np.zeros((len(v_all), 4), dtype=np.uint8)
    colors[:offset, :] = [225, 225, 230, 255] # Clean CAD white for quad body
    colors[offset:, :] = [255, 175, 20, 255]  # Vivid Golden Outline for rails
    unified_trimesh.visual = trimesh.visual.ColorVisuals(mesh=unified_trimesh, vertex_colors=colors)
    unified_trimesh.export(out_glb_path)
    print(f"[+] Saved Single-Mesh GLB to: {out_glb_path} (1 Geometry Node)")

if __name__ == "__main__":
    assets = [
        ("jade_sword", "/Users/bhargavreddy/tripo-logic-approach/vacuum_quad_engine/assets/outputs/jade_sword_final_watertight.obj", "/Users/bhargavreddy/tripo-logic-approach/vacuum_quad_engine/assets/outputs/jade_sword_rails_tuned.obj"),
        ("c1", "/Users/bhargavreddy/tripo-logic-approach/vacuum_quad_engine/assets/outputs/c1_final_watertight.obj", "/Users/bhargavreddy/tripo-logic-approach/vacuum_quad_engine/assets/outputs/c1_rails_tuned.obj"),
        ("002", "/Users/bhargavreddy/tripo-logic-approach/vacuum_quad_engine/assets/outputs/002_final_watertight.obj", "/Users/bhargavreddy/tripo-logic-approach/vacuum_quad_engine/assets/outputs/002_rails_tuned.obj"),
    ]

    for name, q_obj, r_obj in assets:
        out_obj = f"/Users/bhargavreddy/tripo-logic-approach/vacuum_quad_engine/assets/outputs/{name}_single_mesh.obj"
        out_glb = f"/Users/bhargavreddy/tripo-logic-approach/vacuum_quad_engine/assets/outputs/{name}_single_mesh.glb"
        export_single_mesh(q_obj, r_obj, out_obj, out_glb)

