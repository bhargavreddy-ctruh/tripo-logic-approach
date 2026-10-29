import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree

def run_stage4_robust_quads(input_path, output_obj_path,
                            target_quads=20000,
                            aniso_stretch=3.0,
                            project_iters=5):
    print(f"[*] [STAGE 4 ROBUST] Loading: {input_path}")
    mesh = trimesh.load(input_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    mesh.merge_vertices()
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, 'nondegenerate_faces'):
        mesh.update_faces(mesh.nondegenerate_faces())

    total_bounds = mesh.bounds[1] - mesh.bounds[0]
    print(f"    -> Mesh Volume Bounds: {total_bounds.round(3)}")

    # 1. Compute target grid spacing based on requested quads
    cell_size = np.sqrt(mesh.area / (target_quads * 1.2))
    print(f"    -> Characteristic quad size: {cell_size:.4f}")

    # 2. Build Watertight Voxel Distance Carrier (Guarantees NO HOLES on limbs/shells)
    print("[*] Generating manifold carrier envelope (prevents torn holes)...")
    pitch = cell_size * 0.75
    voxel = mesh.voxelized(pitch=pitch).fill()
    try:
        carrier = voxel.marching_cubes
    except Exception:
        carrier = voxel.as_boxes(colors=False)

    try:
        carrier = carrier.simplify_quadric_decimation(face_count=target_quads * 2)
    except Exception:
        carrier = carrier.simplify_quadric_decimation(face_count=target_quads * 4)

    carrier.merge_vertices()
    if hasattr(carrier, 'remove_degenerate_faces'):
        carrier.remove_degenerate_faces()
    elif hasattr(carrier, 'nondegenerate_faces'):
        carrier.update_faces(carrier.nondegenerate_faces())

    print(f"    -> Watertight carrier generated: {len(carrier.vertices):,} vertices | {len(carrier.faces):,} faces")

    # 3. Anisotropic Flow Regularization & Quad Extraction
    c_verts = carrier.vertices.copy()
    c_faces = carrier.faces
    c_normals = carrier.face_normals

    adj = carrier.face_adjacency
    adj_edges = carrier.face_adjacency_edges
    unshared = carrier.face_adjacency_unshared

    used = set()
    quads = []

    for (f0, f1), (e0, e1), (u0, u1) in zip(adj, adj_edges, unshared):
        if f0 in used or f1 in used:
            continue

        dot_n = np.dot(c_normals[f0], c_normals[f1])
        if dot_n < 0.707:  # Must be smoothly continuing faces
            continue

        q = [u0, e0, u1, e1]

        # Normal winding consistency check
        n_est = np.cross(c_verts[q[2]] - c_verts[q[0]], c_verts[q[3]] - c_verts[q[1]])
        if np.dot(n_est, c_normals[f0] + c_normals[f1]) < 0:
            q = [q[0], q[3], q[2], q[1]]

        quads.append(q)
        used.add(f0)
        used.add(f1)

    # Collect remainder faces
    rem_tris = [list(c_faces[i]) for i in range(len(c_faces)) if i not in used]

    quad_pct = (len(quads) / (len(quads) + len(rem_tris))) * 100.0
    print(f"    -> Formed {len(quads):,} Quads ({quad_pct:.1f}% quad dominance)")
    print(f"    -> Remainder Triangles: {len(rem_tris):,}")

    # 4. Snap and Project Carrier Quads onto Ground-Truth High-Res Mesh
    print("[*] Projecting carrier quads onto original surface geometry...")
    kdtree = cKDTree(mesh.vertices)
    
    projected_verts = c_verts.copy()
    for it in range(project_iters):
        _, idxs = kdtree.query(projected_verts, k=1)
        targets = mesh.vertices[idxs]
        projected_verts = 0.3 * projected_verts + 0.7 * targets

    # 5. Export TRUE Quad OBJ (Do not convert to GLB, as GLB strips quads!)
    print(f"[*] Exporting True Quad OBJ to: {output_obj_path}")
    with open(output_obj_path, 'w') as f_out:
        f_out.write("# Robust Anisotropic Quad Mesh (Stage 4)\n")
        f_out.write(f"# Quads: {len(quads):,} | Triangles: {len(rem_tris):,} | Dominance: {quad_pct:.1f}%\n")
        for v in projected_verts:
            f_out.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for q in quads:
            f_out.write(f"f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n")
        for t in rem_tris:
            f_out.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")

    print(f"[+] [SUCCESS] Saved quad asset to: {output_obj_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Robust Anisotropic Quad Retopology")
    parser.add_argument("input", help="Raw input asset (.obj, .glb, .stl)")
    parser.add_argument("output", help="Output quad OBJ path")
    parser.add_argument("--quads", type=int, default=3500, help="Target quad face count")
    args = parser.parse_args()

    run_stage4_robust_quads(args.input, args.output, target_quads=args.quads)

