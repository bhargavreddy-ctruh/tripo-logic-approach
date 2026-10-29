import argparse
import os
import numpy as np
import scipy.sparse as sp
import trimesh
from scipy.spatial import cKDTree

def compute_riemannian_tensor_v2(
    input_path, 
    output_annotated_path,
    target_quads=5000,
    max_aspect_ratio=6.0,
    smoothing_passes=8,
    rails_path=None
):
    print(f"[*] [STAGE 2 v2] Loading asset: {input_path}")
    mesh = trimesh.load(input_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    mesh.merge_vertices()
    if hasattr(mesh, 'remove_duplicate_faces'):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, 'nondegenerate_faces'):
        mesh.update_faces(mesh.nondegenerate_faces())

    v_pos = mesh.vertices
    faces = mesh.faces
    n_verts = len(v_pos)
    n_faces = len(faces)
    face_normals = mesh.face_normals
    face_centers = mesh.triangles_center

    print(f"    -> Mesh: {n_verts:,} vertices | {n_faces:,} faces")

    # 1. Base isotropic quad edge length
    total_area = float(mesh.area)
    h_nominal = np.sqrt(total_area / float(target_quads))
    h_min = h_nominal * 0.20
    h_max = h_nominal * max_aspect_ratio
    print(f"    -> Target budget: ~{target_quads:,} quads | Nominal: {h_nominal:.5f} (Min: {h_min:.5f}, Max: {h_max:.5f})")

    # 2. Local orthonormal tangent bases (t1, t2)
    arb = np.zeros_like(face_normals)
    arb[:, 0] = 1.0
    mask = np.abs(face_normals[:, 0]) > 0.9
    arb[mask, 0] = 0.0
    arb[mask, 1] = 1.0

    t1 = np.cross(face_normals, arb)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    t2 = np.cross(face_normals, t1)

    # 3. Multi-Ring Filtered Weingarten Tensor Estimation
    print("[*] Computing smoothed shape operators across multi-ring neighborhoods...")
    adj_faces = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges

    f0 = adj_faces[:, 0]
    f1 = adj_faces[:, 1]
    edge_vec = v_pos[adj_edges[:, 1]] - v_pos[adj_edges[:, 0]]
    edge_len = np.linalg.norm(edge_vec, axis=1)
    valid = edge_len > 1e-12

    f0 = f0[valid]
    f1 = f1[valid]
    edge_len = edge_len[valid]

    # Directional derivative across the dual edge between face centers
    d_center = np.linalg.norm(face_centers[f1] - face_centers[f0], axis=1) + 1e-12
    d_cent_dir = (face_centers[f1] - face_centers[f0]) / d_center[:, None]
    dn = face_normals[f1] - face_normals[f0]
    dn_de = dn / d_center[:, None]

    II_11 = np.zeros(n_faces)
    II_12 = np.zeros(n_faces)
    II_22 = np.zeros(n_faces)
    weights = np.zeros(n_faces)

    # Accumulate second fundamental form in local tangent bases
    for f_arr, sgn in [(f0, 1.0), (f1, -1.0)]:
        ex = sgn * np.sum(d_cent_dir * t1[f_arr], axis=1)
        ey = sgn * np.sum(d_cent_dir * t2[f_arr], axis=1)
        dn_x = sgn * np.sum(dn_de * t1[f_arr], axis=1)
        dn_y = sgn * np.sum(dn_de * t2[f_arr], axis=1)

        w = edge_len
        np.add.at(II_11, f_arr, w * (dn_x * ex))
        np.add.at(II_12, f_arr, w * 0.5 * (dn_x * ey + dn_y * ex))
        np.add.at(II_22, f_arr, w * (dn_y * ey))
        np.add.at(weights, f_arr, w)

    weights[weights == 0] = 1.0
    II_11 /= weights
    II_12 /= weights
    II_22 /= weights

    # Multi-pass smoothing of the second fundamental form via sparse adjacency graph
    # Eliminates micro-faceting and tessellation noise
    if smoothing_passes > 0 and len(adj_faces) > 0:
        row = np.hstack([f0, f1])
        col = np.hstack([f1, f0])
        val = np.ones(len(row))
        A = sp.csr_matrix((val, (row, col)), shape=(n_faces, n_faces))
        deg = np.array(A.sum(axis=1)).flatten()
        deg[deg == 0] = 1.0
        inv_deg = sp.diags(1.0 / deg)
        M_smooth = inv_deg @ A

        for _ in range(smoothing_passes):
            II_11 = 0.4 * II_11 + 0.6 * (M_smooth @ II_11)
            II_12 = 0.4 * II_12 + 0.6 * (M_smooth @ II_12)
            II_22 = 0.4 * II_22 + 0.6 * (M_smooth @ II_22)

    # 4. Diagonalization: Principal Curvatures & Frame
    tr = II_11 + II_22
    det = II_11 * II_22 - II_12**2
    disc = np.sqrt(np.maximum(0.0, (tr * 0.5)**2 - det))

    k1 = tr * 0.5 + disc
    k2 = tr * 0.5 - disc

    abs_k1 = np.abs(k1)
    abs_k2 = np.abs(k2)

    swap_mask = abs_k2 > abs_k1
    abs_k_max = np.where(swap_mask, abs_k2, abs_k1)
    abs_k_min = np.where(swap_mask, abs_k1, abs_k2)

    theta = 0.5 * np.arctan2(2.0 * II_12, II_11 - II_22)
    theta = np.where(swap_mask, theta + np.pi * 0.5, theta)

    e1 = np.cos(theta)[:, None] * t1 + np.sin(theta)[:, None] * t2
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12
    e2 = np.cross(face_normals, e1)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12

    # 5. Robust Anisotropic Ratio Formulation
    print("[*] Calculating directional aspect ratios with curvature-adaptive scaling...")
    p80 = np.percentile(abs_k_max, 80)
    c_scale = 1.0 / (np.sqrt(p80) + 1e-6)

    # h1 scales across the bend
    h1 = np.clip((c_scale / (np.sqrt(abs_k_max) + 1e-6)) * h_nominal, h_min, h_nominal)

    # Enhanced Cylindrical Aspect Ratio:
    # Uses difference metric (k_max - k_min) to avoid division by near-zero micro-noise
    curvature_asymmetry = (abs_k_max - abs_k_min) / (abs_k_max + abs_k_min + 1e-5)
    
    # Sigmoidal expansion to map cylindrical flow cleanly to 3:1 - 6:1
    aspect_ratio_field = 1.0 + (max_aspect_ratio - 1.0) * (curvature_asymmetry ** 1.5)

    # Apply planar stretching: If both curvatures are low, stretch along dominant mesh bounds / rails
    planar_mask = abs_k_max < np.percentile(abs_k_max, 25)
    aspect_ratio_field[planar_mask] = np.maximum(aspect_ratio_field[planar_mask], max_aspect_ratio * 0.75)

    # Auto-detect Stage 1 boundary rails if not explicitly passed
    if rails_path is None:
        cand_dir = os.path.dirname(output_annotated_path) or "."
        base_name = os.path.basename(input_path).split('.')[0].lower()
        for cand in [
            os.path.join(cand_dir, f"{base_name}_rails_robust.obj"),
            os.path.join(cand_dir, f"{base_name}_rails.obj"),
            os.path.join("vacuum_quad_engine/assets/outputs", f"{base_name}_rails_robust.obj")
        ]:
            if os.path.exists(cand):
                rails_path = cand
                break

    # If boundary rails are available, align planar regions to the nearest rail tangent
    if rails_path and os.path.exists(rails_path) and np.any(planar_mask):
        print(f"[*] Aligning planar stretch directions to Stage 1 boundary rails: {rails_path}")
        try:
            rails_mesh = trimesh.load(rails_path, force='mesh')
            r_edges = rails_mesh.edges_unique
            r_vec = rails_mesh.vertices[r_edges[:, 1]] - rails_mesh.vertices[r_edges[:, 0]]
            r_len = np.linalg.norm(r_vec, axis=1)
            r_valid = r_len > 1e-6
            if np.any(r_valid):
                r_tangents = r_vec[r_valid] / r_len[r_valid, None]
                r_centers = (rails_mesh.vertices[r_edges[r_valid, 0]] + rails_mesh.vertices[r_edges[r_valid, 1]]) * 0.5

                rail_tree = cKDTree(r_centers)
                planar_centers = face_centers[planar_mask]
                _, nearest_rail_idx = rail_tree.query(planar_centers)
                nearest_tangents = r_tangents[nearest_rail_idx]

                # Project rail tangent onto face tangent plane
                p_normals = face_normals[planar_mask]
                proj_t = nearest_tangents - np.sum(nearest_tangents * p_normals, axis=1)[:, None] * p_normals
                proj_norm = np.linalg.norm(proj_t, axis=1, keepdims=True) + 1e-12
                proj_t /= proj_norm

                # e2 is the longitudinal stretch axis along rail, e1 is across rail
                e2[planar_mask] = proj_t
                e1[planar_mask] = np.cross(p_normals, proj_t)
                e1[planar_mask] /= np.linalg.norm(e1[planar_mask], axis=1, keepdims=True) + 1e-12
                print(f"    -> Aligned {np.sum(planar_mask):,} planar faces to feature rail flow.")
        except Exception as e:
            print(f"    -> Rail alignment note: {e}")

    h2 = np.clip(h1 * aspect_ratio_field, h1, h_max)

    print(f"    -> Anisotropic Range: {np.min(aspect_ratio_field):.2f}:1 to {np.max(aspect_ratio_field):.2f}:1")
    print(f"    -> Mean Aspect: {np.mean(aspect_ratio_field):.2f}:1 | Median: {np.median(aspect_ratio_field):.2f}:1")

    # 6. Save Tensor Data
    tensor_data = np.hstack([e1, e2, h1[:, None], h2[:, None]])
    if output_annotated_path.endswith(".obj"):
        binary_save_path = output_annotated_path.replace(".obj", "_tensor.npy")
    elif output_annotated_path.endswith(".glb"):
        binary_save_path = output_annotated_path.replace(".glb", "_tensor.npy")
    else:
        binary_save_path = output_annotated_path + "_tensor.npy"
    np.save(binary_save_path, tensor_data)
    print(f"[+] Saved binary tensor field to: {binary_save_path}")

    # 7. Export High-Contrast Heatmap
    # Red/Orange = 4:1 to 6:1 (Cylindrical Limbs, Sword Blades, Obelisk)
    # Green/Yellow = 2:1 to 3:1 (Subtle torso curves, wheel sides)
    # Blue = 1:1 (Isotropic spheres, filigree, eyes)
    norm_aspect = np.clip((aspect_ratio_field - 1.0) / (max_aspect_ratio - 1.0 + 1e-12), 0.0, 1.0)

    # Fast vectorized vertex mapping
    vert_norm = np.zeros(n_verts)
    counts = np.zeros(n_verts)
    faces_flat = faces.reshape(-1)
    vals_flat = np.repeat(norm_aspect, 3)

    np.add.at(vert_norm, faces_flat, vals_flat)
    np.add.at(counts, faces_flat, 1.0)
    counts[counts == 0] = 1.0
    vert_norm /= counts

    # High-contrast colormap
    vertex_colors = np.zeros((n_verts, 4), dtype=np.uint8)
    vertex_colors[:, 3] = 255
    vertex_colors[:, 0] = (np.clip(vert_norm * 1.5, 0.0, 1.0) * 255).astype(np.uint8) # Red
    vertex_colors[:, 1] = (np.sin(vert_norm * np.pi) * 230).astype(np.uint8)          # Green
    vertex_colors[:, 2] = (np.clip((1.0 - vert_norm * 1.5), 0.0, 1.0) * 255).astype(np.uint8) # Blue

    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)
    mesh.export(output_annotated_path)
    print(f"[+] [STAGE 2 v2 COMPLETE] Inspection mesh saved to: {output_annotated_path}")

    # Also export companion GLB for instant viewing in 3D viewers
    if output_annotated_path.endswith(".obj"):
        glb_path = output_annotated_path.replace(".obj", ".glb")
        mesh.export(glb_path)
        print(f"[+] Also exported GLB companion: {glb_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2 v2: Robust Riemannian Anisotropic Tensor Solver")
    parser.add_argument("input", help="Input mesh file path (.obj, .glb, .stl, .ply)")
    parser.add_argument("output", help="Output annotated OBJ path")
    parser.add_argument("--quads", type=int, default=5000, help="Target quad budget (default: 5000)")
    parser.add_argument("--max-aspect", type=float, default=6.0, help="Max anisotropic aspect ratio (default: 6.0)")
    parser.add_argument("--passes", type=int, default=8, help="Smoothing passes to eliminate faceting noise (default: 8)")
    parser.add_argument("--rails", type=str, default=None, help="Optional Stage 1 boundary rails file (.obj) for planar alignment")
    args = parser.parse_args()

    compute_riemannian_tensor_v2(
        args.input, 
        args.output,
        target_quads=args.quads,
        max_aspect_ratio=args.max_aspect,
        smoothing_passes=args.passes,
        rails_path=args.rails
    )
