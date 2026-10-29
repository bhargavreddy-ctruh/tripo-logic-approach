import argparse
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import trimesh

def compute_riemannian_tensor(
    input_path, 
    output_annotated_path,
    target_quads=5000,
    max_aspect_ratio=6.0,
    grading_factor=0.35
):
    print(f"[*] [STAGE 2] Loading asset: {input_path}")
    mesh = trimesh.load(input_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # 1. Clean manifold geometry & purge floating noise
    mesh.merge_vertices()
    if hasattr(mesh, 'remove_duplicate_faces'):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, 'nondegenerate_faces'):
        mesh.update_faces(mesh.nondegenerate_faces())

    try:
        af = mesh.face_adjacency
        n_faces = len(mesh.faces)
        if len(af) > 0 and n_faces > 100:
            adj_mat = sp.csr_matrix((np.ones(len(af)), (af[:, 0], af[:, 1])), shape=(n_faces, n_faces))
            _, labels = csg.connected_components(adj_mat, directed=False)
            counts = np.bincount(labels)
            max_c = np.max(counts)
            keep_mask = counts[labels] >= (max_c * 0.01)
            if np.sum(keep_mask) < n_faces:
                mesh.update_faces(keep_mask)
                mesh.remove_unreferenced_vertices()
                print(f"    -> Filtered floating noise. Working faces: {len(mesh.faces):,}")
    except Exception as e:
        print(f"    -> Component filtering skipped ({e}).")

    v_pos = mesh.vertices
    faces = mesh.faces
    n_verts = len(v_pos)
    n_faces = len(faces)
    face_normals = mesh.face_normals
    face_centers = mesh.triangles_center

    print(f"    -> Working Mesh: {n_verts:,} vertices | {n_faces:,} faces")

    # 2. Base isotropic sizing budget
    total_area = float(mesh.area)
    h_nominal = np.sqrt(total_area / float(target_quads))
    h_min = h_nominal * 0.25
    h_max = h_nominal * max_aspect_ratio
    print(f"    -> Target budget: ~{target_quads:,} quads | Nominal size: {h_nominal:.5f} (Min: {h_min:.5f}, Max: {h_max:.5f})")

    # 3. Local Orthonormal Tangent Bases (t1, t2)
    arb = np.zeros_like(face_normals)
    arb[:, 0] = 1.0
    mask = np.abs(face_normals[:, 0]) > 0.9
    arb[mask, 0] = 0.0
    arb[mask, 1] = 1.0

    t1 = np.cross(face_normals, arb)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    t2 = np.cross(face_normals, t1)

    # 4. Discrete Weingarten Shape Operator via Normal Differences across Edges
    print("[*] Estimating discrete Weingarten shape operator & principal frames...")
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

    dn = face_normals[f1] - face_normals[f0]
    d_center = np.linalg.norm(face_centers[f1] - face_centers[f0], axis=1) + 1e-12
    dn_de = dn / d_center[:, None]
    d_cent_dir = (face_centers[f1] - face_centers[f0]) / d_center[:, None]

    II_11 = np.zeros(n_faces)
    II_12 = np.zeros(n_faces)
    II_22 = np.zeros(n_faces)
    II_weights = np.zeros(n_faces)

    # Fast vectorized tensor accumulation for f0 and f1
    for f_arr, sgn in [(f0, 1.0), (f1, -1.0)]:
        e_x = sgn * np.sum(d_cent_dir * t1[f_arr], axis=1)
        e_y = sgn * np.sum(d_cent_dir * t2[f_arr], axis=1)
        dn_x = sgn * np.sum(dn_de * t1[f_arr], axis=1)
        dn_y = sgn * np.sum(dn_de * t2[f_arr], axis=1)

        weight = edge_len
        np.add.at(II_11, f_arr, weight * (dn_x * e_x))
        np.add.at(II_12, f_arr, weight * 0.5 * (dn_x * e_y + dn_y * e_x))
        np.add.at(II_22, f_arr, weight * (dn_y * e_y))
        np.add.at(II_weights, f_arr, weight)

    II_weights[II_weights == 0] = 1.0
    II_11 /= II_weights
    II_12 /= II_weights
    II_22 /= II_weights

    # 5. Diagonalize 2x2 Shape Operator to extract (kappa1, kappa2) and (e1, e2)
    tr = II_11 + II_22
    det = II_11 * II_22 - II_12**2
    disc = np.sqrt(np.maximum(0.0, (tr * 0.5)**2 - det))

    k1 = tr * 0.5 + disc
    k2 = tr * 0.5 - disc

    abs_k1 = np.abs(k1)
    abs_k2 = np.abs(k2)

    # Sort so kappa_max >= kappa_min
    swap_mask = abs_k2 > abs_k1
    abs_k_max = np.where(swap_mask, abs_k2, abs_k1)
    abs_k_min = np.where(swap_mask, abs_k1, abs_k2)

    # Principal eigenvector for maximum curvature direction (bend direction)
    theta = 0.5 * np.arctan2(2.0 * II_12, II_11 - II_22)
    theta = np.where(swap_mask, theta + np.pi * 0.5, theta)

    e1 = np.cos(theta)[:, None] * t1 + np.sin(theta)[:, None] * t2
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12
    e2 = np.cross(face_normals, e1)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12

    # 6. Compute Anisotropic Metric Sizing (h1, h2)
    print("[*] Computing anisotropic sizing factors and aspect ratio field...")
    p85_curv = np.percentile(abs_k_max, 85)
    c_scale = 1.0 / (np.sqrt(p85_curv) + 1e-6)

    # h1 is across the curvature (small when curvature is high)
    h1 = np.clip(c_scale / (np.sqrt(abs_k_max) + 1e-6) * h_nominal, h_min, h_nominal)

    # h2 is along the curvature (stretching up to max_aspect_ratio * h1)
    aspect_potential = np.sqrt((abs_k_max + 1e-5) / (abs_k_min + 1e-5))
    aspect_potential = np.clip(aspect_potential, 1.0, max_aspect_ratio)
    h2 = np.clip(h1 * aspect_potential, h1, h_max)

    # 7. Fast Vectorized Metric Grading Smoothing across Dual Face Graph
    print(f"[*] Applying directional metric grading (limit: {grading_factor})...")
    d_f = np.linalg.norm(face_centers[f0] - face_centers[f1], axis=1) + 1e-12
    max_diff = grading_factor * d_f * (h_max / h_nominal)

    for it in range(6):
        np.minimum.at(h1, f1, h1[f0] + max_diff)
        np.minimum.at(h1, f0, h1[f1] + max_diff)
        np.minimum.at(h2, f1, h2[f0] + max_diff)
        np.minimum.at(h2, f0, h2[f1] + max_diff)

    h2 = np.maximum(h2, h1)
    aspect_ratio_field = h2 / (h1 + 1e-12)
    print(f"    -> Aspect Ratio Range: {np.min(aspect_ratio_field):.2f}:1 (square) to {np.max(aspect_ratio_field):.2f}:1 (stretched)")

    # 8. Save Binary Metric Tensor Data for Stages 3 & 4
    # Shape: (N_faces, 8) -> [e1 (3), e2 (3), h1, h2]
    metric_data = np.hstack([e1, e2, h1[:, None], h2[:, None]])
    if output_annotated_path.endswith(".obj"):
        binary_save_path = output_annotated_path.replace(".obj", "_tensor.npy")
    elif output_annotated_path.endswith(".glb"):
        binary_save_path = output_annotated_path.replace(".glb", "_tensor.npy")
    else:
        binary_save_path = output_annotated_path + "_tensor.npy"
    np.save(binary_save_path, metric_data)
    print(f"[+] Saved tensor array to: {binary_save_path}")

    # 9. Export Annotated Mesh with Aspect Ratio Heatmap
    # Blue = 1:1 (Square, isotropic)
    # Green/Yellow = 2:1 to 3:1 (Moderate stretch)
    # Red/Orange = 4:1 to 6:1 (High stretch: blades, limbs, poles)
    norm_aspect = np.clip((aspect_ratio_field - 1.0) / (max_aspect_ratio - 1.0 + 1e-12), 0.0, 1.0)

    # Fast vectorized vertex mapping
    vert_norm_aspect = np.zeros(n_verts)
    vert_face_counts = np.zeros(n_verts)
    faces_flat = faces.reshape(-1)
    vals_flat = np.repeat(norm_aspect, 3)

    np.add.at(vert_norm_aspect, faces_flat, vals_flat)
    np.add.at(vert_face_counts, faces_flat, 1.0)
    vert_face_counts[vert_face_counts == 0] = 1.0
    vert_norm_aspect /= vert_face_counts

    vertex_colors = np.zeros((n_verts, 4), dtype=np.uint8)
    vertex_colors[:, 3] = 255
    vertex_colors[:, 0] = (vert_norm_aspect * 255).astype(np.uint8)              # Red channel
    vertex_colors[:, 1] = (np.sin(vert_norm_aspect * np.pi) * 220).astype(np.uint8) # Green channel
    vertex_colors[:, 2] = ((1.0 - vert_norm_aspect) * 255).astype(np.uint8)      # Blue channel

    mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)
    mesh.export(output_annotated_path)
    print(f"[+] [STAGE 2 COMPLETE] Saved aspect-ratio inspection mesh to: {output_annotated_path}")

    # Also export .glb companion for seamless viewing in standard 3D viewers
    if output_annotated_path.endswith(".obj"):
        glb_companion = output_annotated_path.replace(".obj", ".glb")
        mesh.export(glb_companion)
        print(f"[+] Also exported GLB companion: {glb_companion}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 2: Riemannian Metric Tensor Field Solver")
    parser.add_argument("input", help="Input mesh path (.obj, .glb, .stl, .ply)")
    parser.add_argument("output", help="Output annotated OBJ path")
    parser.add_argument("--quads", type=int, default=5000, help="Target quad count budget (default: 5000)")
    parser.add_argument("--max-aspect", type=float, default=6.0, help="Maximum allowed stretch aspect ratio (default: 6.0)")
    parser.add_argument("--grading", type=float, default=0.35, help="Metric grading limit factor (default: 0.35)")
    args = parser.parse_args()

    compute_riemannian_tensor(
        args.input, 
        args.output,
        target_quads=args.quads,
        max_aspect_ratio=args.max_aspect,
        grading_factor=args.grading
    )
