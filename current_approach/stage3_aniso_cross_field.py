import argparse
import os
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh
from scipy.spatial import cKDTree

def compute_anisotropic_cross_field(
    input_path, 
    tensor_npy_path, 
    output_field_obj,
    cv_high=35.0, 
    cc_high=26.0,
    viz_sample_count=2500,
    rails_path=None
):
    print(f"[*] [STAGE 3] Loading asset: {input_path}")
    mesh = trimesh.load(input_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    mesh.merge_vertices()
    # Clean manifold geometry & weld spatial coordinate duplicates
    _, u_idx, inv = np.unique(np.round(mesh.vertices, 6), axis=0, return_index=True, return_inverse=True)
    mesh = trimesh.Trimesh(vertices=mesh.vertices[u_idx], faces=inv[mesh.faces], process=True)
    if hasattr(mesh, 'remove_duplicate_faces'):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    if hasattr(mesh, 'nondegenerate_faces'):
        mesh.update_faces(mesh.nondegenerate_faces())

    v_pos = mesh.vertices
    faces = mesh.faces
    n_faces = len(faces)
    face_normals = mesh.face_normals
    face_centers = mesh.triangles_center
    adj_faces = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges

    print(f"    -> Mesh: {len(v_pos):,} vertices | {n_faces:,} faces")

    # 1. Load Stage 2 Riemannian Tensor Field
    print(f"[*] Loading Stage 2 tensor field: {tensor_npy_path}")
    tensor_data = np.load(tensor_npy_path)
    # Shape: (N_faces, 8) -> [e1 (3), e2 (3), h1, h2]
    e1_in = tensor_data[:, 0:3]
    e2_in = tensor_data[:, 3:6]
    h1_in = tensor_data[:, 6]
    h2_in = tensor_data[:, 7]
    aspect_ratios = h2_in / (h1_in + 1e-12)

    # 2. Local Orthonormal Tangent Frame (t1, t2)
    arb = np.zeros_like(face_normals)
    arb[:, 0] = 1.0
    mask = np.abs(face_normals[:, 0]) > 0.9
    arb[mask, 0] = 0.0
    arb[mask, 1] = 1.0

    t1 = np.cross(face_normals, arb)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    t2 = np.cross(face_normals, t1)

    # 3. Establish Constraints (Boundary Rails + Strong Anisotropic Curvature)
    print("[*] Assembling boundary rail constraints and curvature guidance...")
    cos_cv = np.cos(np.radians(cv_high))
    cos_cc = np.cos(np.radians(cc_high))

    u_complex = np.zeros(n_faces, dtype=np.complex128)
    fixed_mask = np.zeros(n_faces, dtype=bool)
    constraint_weights = np.zeros(n_faces, dtype=np.float64)

    # Priority 0: Stage 1 Feature Creases & Rails
    f0 = adj_faces[:, 0]
    f1 = adj_faces[:, 1]
    edge_vec = v_pos[adj_edges[:, 1]] - v_pos[adj_edges[:, 0]]
    edge_len = np.linalg.norm(edge_vec, axis=1)
    valid = edge_len > 1e-12

    f0 = f0[valid]
    f1 = f1[valid]
    edge_vec = edge_vec[valid]
    edge_len = edge_len[valid]
    edge_dir = edge_vec / edge_len[:, None]

    dot_n = np.clip(np.sum(face_normals[f0] * face_normals[f1], axis=1), -1.0, 1.0)
    signed_sin = np.sum(np.cross(face_normals[f0], face_normals[f1]) * edge_dir, axis=1)
    is_convex = signed_sin >= 0.0
    is_crease = (is_convex & (dot_n <= cos_cv)) | ((~is_convex) & (dot_n <= cos_cc))

    crease_f0 = f0[is_crease]
    crease_f1 = f1[is_crease]
    crease_ed = edge_dir[is_crease]

    for f_arr in (crease_f0, crease_f1):
        ex = np.sum(crease_ed * t1[f_arr], axis=1)
        ey = np.sum(crease_ed * t2[f_arr], axis=1)
        theta = np.arctan2(ey, ex)
        # 4-RoSy complex representation: e^(4 * i * theta)
        np.add.at(u_complex, f_arr, np.exp(1j * 4.0 * theta))
        fixed_mask[f_arr] = True
        constraint_weights[f_arr] = 1000.0  # Rigid hard constraint

    # Auto-detect Stage 1 boundary rails if not explicitly passed
    if rails_path is None:
        cand_dir = os.path.dirname(output_field_obj) or "."
        base_name = os.path.basename(input_path).split('.')[0].lower()
        for cand in [
            os.path.join(cand_dir, f"{base_name}_stage1_rails_clean.obj"),
            os.path.join(cand_dir, f"{base_name}_rails_robust.obj"),
            os.path.join(cand_dir, f"{base_name}_stage1_rails.obj"),
            os.path.join(cand_dir, f"{base_name}_rails.obj"),
            os.path.join("vacuum_quad_engine/assets/outputs", f"{base_name}_rails_robust.obj")
        ]:
            if os.path.exists(cand):
                rails_path = cand
                break

    if rails_path and os.path.exists(rails_path):
        print(f"[*] Incorporating Stage 1 boundary rails into cross-field: {rails_path}")
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
                avg_edge = np.mean(edge_len)
                dists, nearest_r_idx = rail_tree.query(face_centers, distance_upper_bound=avg_edge * 0.75)
                near_rail = dists < np.inf
                near_indices = np.where(near_rail)[0]
                if len(near_indices) > 0:
                    t_rail = r_tangents[nearest_r_idx[near_indices]]
                    fn_near = face_normals[near_indices]
                    proj_t = t_rail - np.sum(t_rail * fn_near, axis=1)[:, None] * fn_near
                    norm_proj = np.linalg.norm(proj_t, axis=1, keepdims=True) + 1e-12
                    proj_t /= norm_proj

                    ex_r = np.sum(proj_t * t1[near_indices], axis=1)
                    ey_r = np.sum(proj_t * t2[near_indices], axis=1)
                    theta_r = np.arctan2(ey_r, ex_r)
                    u_complex[near_indices] += np.exp(1j * 4.0 * theta_r)
                    fixed_mask[near_indices] = True
                    constraint_weights[near_indices] = np.maximum(constraint_weights[near_indices], 1000.0)
                    print(f"    -> Locked {len(near_indices):,} faces touching Stage 1 boundary rails.")
        except Exception as e:
            print(f"    -> Boundary rail constraint note: {e}")

    # Priority 1: High Anisotropic Regions (stretch ratio >= 2.5)
    strong_aniso = (aspect_ratios >= 2.5) & (~fixed_mask)
    strong_indices = np.where(strong_aniso)[0]
    if len(strong_indices) > 0:
        dir_guide = e2_in[strong_indices]
        ex_g = np.sum(dir_guide * t1[strong_indices], axis=1)
        ey_g = np.sum(dir_guide * t2[strong_indices], axis=1)
        theta_g = np.arctan2(ey_g, ex_g)
        u_complex[strong_indices] = np.exp(1j * 4.0 * theta_g)
        fixed_mask[strong_indices] = True
        constraint_weights[strong_indices] = 100.0  # Flexible directional guide

    # Normalize constrained targets
    has_constraints = np.sum(fixed_mask) > 0
    if has_constraints:
        u_complex[fixed_mask] /= np.abs(u_complex[fixed_mask]) + 1e-12
    else:
        u_complex[0] = 1.0 + 0j
        fixed_mask[0] = True
        constraint_weights[0] = 100.0

    print(f"    -> Constrained guide faces: {np.sum(fixed_mask):,} / {n_faces:,} ({np.sum(fixed_mask)/n_faces*100:.1f}%)")

    # 4. Anisotropic Metric-Weighted Connection Laplacian Assembly
    print("[*] Assembling metric-weighted Connection Laplacian...")
    ed_dot_t1_0 = np.sum(edge_dir * t1[f0], axis=1)
    ed_dot_t2_0 = np.sum(edge_dir * t2[f0], axis=1)
    ed_dot_t1_1 = np.sum(edge_dir * t1[f1], axis=1)
    ed_dot_t2_1 = np.sum(edge_dir * t2[f1], axis=1)

    alpha0 = np.arctan2(ed_dot_t2_0, ed_dot_t1_0)
    alpha1 = np.arctan2(ed_dot_t2_1, ed_dot_t1_1)
    transport = np.exp(1j * 4.0 * (alpha1 - alpha0))

    d_center = np.linalg.norm(face_centers[f0] - face_centers[f1], axis=1) + 1e-12
    avg_aniso = 0.5 * (aspect_ratios[f0] + aspect_ratios[f1])
    weight = (edge_len / d_center) * np.sqrt(avg_aniso)

    rows = np.concatenate([f0, f1])
    cols = np.concatenate([f1, f0])
    vals = np.concatenate([weight * np.conj(transport), weight * transport])

    W = sp.csr_matrix((vals, (rows, cols)), shape=(n_faces, n_faces), dtype=np.complex128)
    L = sp.diags(np.array(np.abs(W).sum(axis=1)).flatten(), dtype=np.complex128) - W

    # Penalty system: (L + P) u = P u_target
    P = sp.diags(constraint_weights, dtype=np.complex128)
    rhs = constraint_weights * u_complex

    print("[*] Solving sparse linear system for global cross-field...")
    A = L + P + sp.eye(n_faces, dtype=np.complex128) * 1e-6

    if n_faces <= 30000:
        A_csc = A.tocsc()
        u_solved = spla.spsolve(A_csc, rhs)
    else:
        # High-performance iterative solver for dense meshes
        diag_A = A.diagonal()
        diag_A[np.abs(diag_A) < 1e-12] = 1.0
        M_prec = sp.diags(1.0 / diag_A)
        u_solved, info = spla.bicgstab(A, rhs, M=M_prec, maxiter=300, rtol=1e-5)
        if info != 0:
            print("    -> Note: Iterative solver fallback to direct solver...")
            A_csc = A.tocsc()
            u_solved = spla.spsolve(A_csc, rhs)

    # 5. Extract Orthogonal Unit Vectors (u, v) per Face
    angles = np.angle(u_solved) / 4.0
    u_dir = np.cos(angles)[:, None] * t1 + np.sin(angles)[:, None] * t2
    u_dir /= np.linalg.norm(u_dir, axis=1, keepdims=True) + 1e-12

    # Align u_dir with elongation axis e2 where anisotropy is present
    flip_mask = np.sum(u_dir * e2_in, axis=1) < 0.0
    u_dir[flip_mask] = -u_dir[flip_mask]

    v_dir = np.cross(face_normals, u_dir)
    v_dir /= np.linalg.norm(v_dir, axis=1, keepdims=True) + 1e-12

    # Save binary array for Stage 4
    binary_out = output_field_obj.replace(".obj", ".npy")
    field_data = np.stack([u_dir, v_dir], axis=1)  # Shape: (N_faces, 2, 3)
    np.save(binary_out, field_data)
    print(f"[+] Saved field array to: {binary_out}")

    dir_name = os.path.dirname(output_field_obj) or "."
    canonical_out = os.path.join(dir_name, "beetle_car_stage3_field.npy")
    if os.path.abspath(canonical_out) != os.path.abspath(binary_out):
        np.save(canonical_out, field_data)
        print(f"[+] Also saved canonical field array to: {canonical_out}")

    # 6. Generate Clean 3D Ribbon Inspection Overlay
    print(f"[*] Generating clean cross-ribbon visual overlay ({viz_sample_count} samples)...")
    sample_step = max(1, n_faces // viz_sample_count)
    samples = np.arange(0, n_faces, sample_step)

    char_len = np.mean(edge_len) * 1.1
    viz_verts = []
    viz_faces = []

    for idx in samples:
        c = face_centers[idx] + face_normals[idx] * (char_len * 0.04)
        
        # Scale ribbon arms using metric stretch ratio
        ratio = min(3.5, aspect_ratios[idx])
        len_u = char_len * np.sqrt(ratio)
        len_v = char_len / np.sqrt(ratio)

        u = u_dir[idx] * len_u
        v = v_dir[idx] * len_v
        w = char_len * 0.10

        # Ribbon 1 (along u)
        r1_0 = c - u - v * (w * 0.5)
        r1_1 = c + u - v * (w * 0.5)
        r1_2 = c + u + v * (w * 0.5)
        r1_3 = c - u + v * (w * 0.5)

        # Ribbon 2 (along v)
        r2_0 = c - v - u * (w * 0.5)
        r2_1 = c + v - u * (w * 0.5)
        r2_2 = c + v + u * (w * 0.5)
        r2_3 = c - v + u * (w * 0.5)

        b_idx = len(viz_verts)
        viz_verts.extend([r1_0, r1_1, r1_2, r1_3, r2_0, r2_1, r2_2, r2_3])
        # Two triangulated quads (4 triangles total)
        viz_faces.append([b_idx, b_idx + 1, b_idx + 2])
        viz_faces.append([b_idx, b_idx + 2, b_idx + 3])
        viz_faces.append([b_idx + 4, b_idx + 5, b_idx + 6])
        viz_faces.append([b_idx + 4, b_idx + 6, b_idx + 7])

    viz_mesh = trimesh.Trimesh(vertices=np.array(viz_verts), faces=np.array(viz_faces), process=False)
    viz_mesh.export(output_field_obj)
    print(f"[+] [STAGE 3 COMPLETE] Clean ribbon cross-field saved to: {output_field_obj}")

    # Also export companion GLB for direct preview
    if output_field_obj.endswith(".obj"):
        glb_out = output_field_obj.replace(".obj", ".glb")
        viz_mesh.export(glb_out)
        print(f"[+] Also exported GLB companion: {glb_out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 3: Feature-Locked Anisotropic Cross-Field")
    parser.add_argument("input", help="Input mesh (.obj, .glb, .stl)")
    parser.add_argument("tensor", help="Stage 2 _tensor.npy path")
    parser.add_argument("output", help="Output cross-field visualization OBJ path")
    parser.add_argument("--cv-high", type=float, default=35.0, help="Convex crease threshold (deg)")
    parser.add_argument("--cc-high", type=float, default=26.0, help="Concave crease threshold (deg)")
    parser.add_argument("--samples", type=int, default=2500, help="Number of visualization ribbon samples")
    parser.add_argument("--rails", type=str, default=None, help="Stage 1 boundary rails (.obj)")
    args = parser.parse_args()

    compute_anisotropic_cross_field(
        args.input, 
        args.tensor, 
        args.output,
        cv_high=args.cv_high, 
        cc_high=args.cc_high,
        viz_sample_count=args.samples,
        rails_path=args.rails
    )
