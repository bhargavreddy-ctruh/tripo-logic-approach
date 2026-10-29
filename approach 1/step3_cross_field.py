import argparse
from collections import defaultdict, deque
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh

def compute_signed_hysteresis_creases(
    mesh: trimesh.Trimesh, 
    convex_high=12.0, 
    convex_low=6.0, 
    concave_high=12.0, 
    concave_low=6.0
):
    """
    Module M06: Signed Convex/Concave Hysteresis Feature Detection.
    Extracts sharp ridges and concave isolation valleys.
    """
    if convex_high < convex_low:
        convex_high, convex_low = convex_low, convex_high
    if concave_high < concave_low:
        concave_high, concave_low = concave_low, concave_high

    face_normals = mesh.face_normals
    adj_faces = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges
    
    if len(adj_faces) == 0:
        return set(), set(), set()

    dot_n = np.clip(np.sum(face_normals[adj_faces[:, 0]] * face_normals[adj_faces[:, 1]], axis=1), -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(dot_n))
    
    try:
        is_convex = mesh.face_adjacency_convex
    except Exception:
        edge_vec = mesh.vertices[adj_edges[:, 1]] - mesh.vertices[adj_edges[:, 0]]
        edge_dir = edge_vec / (np.linalg.norm(edge_vec, axis=1, keepdims=True) + 1e-12)
        n0 = face_normals[adj_faces[:, 0]]
        n1 = face_normals[adj_faces[:, 1]]
        cross_val = np.sum(np.cross(n0, n1) * edge_dir, axis=1)
        is_convex = cross_val >= 0.0

    cv_seeds = (is_convex) & (angles_deg >= convex_high)
    cv_cands = (is_convex) & (angles_deg >= convex_low)
    
    cc_seeds = (~is_convex) & (angles_deg >= concave_high)
    cc_cands = (~is_convex) & (angles_deg >= concave_low)
    
    def propagate(seeds_mask, cands_mask):
        seed_indices = np.where(seeds_mask)[0]
        cand_indices = np.where(cands_mask)[0]
        if len(seed_indices) == 0:
            return set()
            
        v_to_e = defaultdict(list)
        for idx in cand_indices:
            v0, v1 = adj_edges[idx]
            v_to_e[v0].append(idx)
            v_to_e[v1].append(idx)
            
        visited = set(seed_indices)
        queue = deque(seed_indices)
        
        while queue:
            curr_idx = queue.popleft()
            v0, v1 = adj_edges[curr_idx]
            for v in (v0, v1):
                for nbr_idx in v_to_e[v]:
                    if nbr_idx not in visited:
                        visited.add(nbr_idx)
                        queue.append(nbr_idx)
        return visited

    active_cv = propagate(cv_seeds, cv_cands)
    active_cc = propagate(cc_seeds, cc_cands)
    
    sharp_adj_indices = active_cv.union(active_cc)
    return sharp_adj_indices, active_cv, active_cc


def compute_step3_cross_field(
    input_path, 
    output_path,
    cv_high=12.0, 
    cv_low=6.0,
    cc_high=12.0, 
    cc_low=6.0,
    line_density=2500,
    render_mode="glyphs"
):
    print(f"[*] [STEP 3] Loading asset: {input_path}")
    mesh = trimesh.load(input_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    mesh.merge_vertices()
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, 'nondegenerate_faces'):
        mesh.update_faces(mesh.nondegenerate_faces())

    n_faces = len(mesh.faces)
    v_pos = mesh.vertices
    face_normals = mesh.face_normals
    face_centers = mesh.triangles_center

    print(f"    -> Mesh Loaded: {len(v_pos):,} vertices | {n_faces:,} faces")

    # 1. Establish an orthonormal local tangent basis (t1, t2) for every face
    print("[*] Constructing local tangent bases...")
    arb = np.zeros_like(face_normals)
    arb[:, 0] = 1.0
    mask = np.abs(face_normals[:, 0]) > 0.9
    arb[mask, 0] = 0.0
    arb[mask, 1] = 1.0

    t1 = np.cross(face_normals, arb)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    t2 = np.cross(face_normals, t1)

    # 2. Extract Step 1 Feature Creases to act as Hard Field Constraints
    print("[*] Detecting feature alignment anchors from Step 1...")
    adj_faces = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges

    sharp_adj_indices, active_cv, active_cc = compute_signed_hysteresis_creases(
        mesh, convex_high=cv_high, convex_low=cv_low, concave_high=cc_high, concave_low=cc_low
    )

    u_complex = np.zeros(n_faces, dtype=np.complex128)
    fixed_mask = np.zeros(n_faces, dtype=bool)

    for adj_idx in sharp_adj_indices:
        f0, f1 = adj_faces[adj_idx]
        v0, v1 = adj_edges[adj_idx]

        edge_vec = v_pos[v1] - v_pos[v0]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-12:
            continue
        edge_dir = edge_vec / edge_len

        for f_idx in (f0, f1):
            x = np.dot(edge_dir, t1[f_idx])
            y = np.dot(edge_dir, t2[f_idx])
            angle = np.arctan2(y, x)

            # 4-RoSy complex representation: e^(4 * i * theta)
            u_complex[f_idx] += np.exp(1j * 4.0 * angle)
            fixed_mask[f_idx] = True

    anchored_count = np.sum(fixed_mask)
    if anchored_count > 0:
        u_complex[fixed_mask] /= np.abs(u_complex[fixed_mask]) + 1e-12
    else:
        u_complex[0] = 1.0 + 0j
        fixed_mask[0] = True
        anchored_count = 1

    print(f"    -> Feature-anchored guide faces: {anchored_count:,} / {n_faces:,} ({anchored_count / n_faces * 100:.1f}%)")

    # 3. Assemble Connection Laplacian on dual face graph
    print("[*] Assembling Connection Laplacian on dual face graph...")
    edge_vecs = v_pos[adj_edges[:, 1]] - v_pos[adj_edges[:, 0]]
    edge_lens = np.linalg.norm(edge_vecs, axis=1)
    valid_edges = edge_lens > 1e-12

    val_adj_faces = adj_faces[valid_edges]
    val_edge_vecs = edge_vecs[valid_edges]
    val_edge_lens = edge_lens[valid_edges]
    val_edge_dirs = val_edge_vecs / val_edge_lens[:, None]

    f0 = val_adj_faces[:, 0]
    f1 = val_adj_faces[:, 1]

    alpha0 = np.arctan2(np.sum(val_edge_dirs * t2[f0], axis=1), np.sum(val_edge_dirs * t1[f0], axis=1))
    alpha1 = np.arctan2(np.sum(val_edge_dirs * t2[f1], axis=1), np.sum(val_edge_dirs * t1[f1], axis=1))

    transport = np.exp(1j * 4.0 * (alpha1 - alpha0))
    d_center = np.linalg.norm(face_centers[f0] - face_centers[f1], axis=1) + 1e-12
    weight = val_edge_lens / d_center

    rows = np.concatenate([f0, f1])
    cols = np.concatenate([f1, f0])
    vals = np.concatenate([weight * np.conj(transport), weight * transport])

    W = sp.csr_matrix((vals, (rows, cols)), shape=(n_faces, n_faces), dtype=np.complex128)
    diagonal_weights = np.array(np.abs(W).sum(axis=1)).flatten()
    L = sp.diags(diagonal_weights, dtype=np.complex128) - W

    # 4. Harmonic Field Solve (Dirichlet / Penalty formulation)
    print("[*] Solving smooth harmonic field equations...")
    penalty = 1000.0
    P = sp.diags(fixed_mask.astype(np.float64) * penalty, dtype=np.complex128)
    rhs = penalty * (u_complex * fixed_mask)

    A = (L + P + sp.eye(n_faces, dtype=np.complex128) * 1e-6).tocsc()
    u_solved = spla.spsolve(A, rhs)

    u_norm = np.abs(u_solved)
    u_solved_safe = np.where(u_norm > 1e-12, u_solved / (u_norm + 1e-12), 1.0 + 0j)

    # 5. Extract Orthogonal Cross-Field Vectors per Face
    angles = np.angle(u_solved_safe) / 4.0

    dir_u = np.cos(angles)[:, None] * t1 + np.sin(angles)[:, None] * t2
    dir_u /= np.linalg.norm(dir_u, axis=1, keepdims=True) + 1e-12

    dir_v = np.cross(face_normals, dir_u)
    dir_v /= np.linalg.norm(dir_v, axis=1, keepdims=True) + 1e-12

    # Save raw field vectors for Step 4
    field_data = np.stack([dir_u, dir_v], axis=1)  # Shape: (N_faces, 2, 3)
    field_npy_path = output_path.replace(".obj", "_field.npy")
    np.save(field_npy_path, field_data)
    print(f"[+] Saved field array to: {field_npy_path}")

    # 6. Generate Clean Visual Cross-Field Geometry
    print(f"[*] Generating clean 3D cross glyphs (mode: {render_mode})...")
    step = max(1, n_faces // line_density)
    sample_indices = np.arange(0, n_faces, step)

    avg_edge_len = np.mean(mesh.edges_unique_length)
    arm_len = avg_edge_len * 0.75
    width = arm_len * 0.12

    with open(output_path, 'w') as f:
        f.write("# Visual Cross-Field (Step 3 Clean Inspection)\n")
        f.write(f"# Input: {input_path}\n")
        f.write(f"# Base Faces: {n_faces:,} | Sampled Crosses: {len(sample_indices):,}\n")
        
        # 1. Base Mesh Object
        f.write("o BaseMesh\n")
        for v in v_pos:
            # Subtle neutral slate-gray vertex color
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f} 0.35 0.38 0.42\n")
        for face in mesh.faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")

        # 2. Cross Field Glyphs Object
        # Exporting as micro-quad planar ribbons eliminates the WebGL / Three.js
        # line-strip bug that connects disjoint line elements into web artifacts.
        f.write("o CrossField_Glyphs\n")
        v_idx_base = len(v_pos) + 1

        if render_mode == "glyphs":
            for idx in sample_indices:
                c = face_centers[idx] + face_normals[idx] * (arm_len * 0.04)
                u = dir_u[idx] * arm_len
                v = dir_v[idx] * arm_len
                w_u = dir_u[idx] * width
                w_v = dir_v[idx] * width

                # Quad for Arm U: (c - u) to (c + u)
                p0 = c - u - w_v
                p1 = c + u - w_v
                p2 = c + u + w_v
                p3 = c - u + w_v

                # Quad for Arm V: (c - v) to (c + v)
                p4 = c - v - w_u
                p5 = c + v - w_u
                p6 = c + v + w_u
                p7 = c - v + w_u

                # Bright glowing cyan / electric blue cross color
                for p in [p0, p1, p2, p3, p4, p5, p6, p7]:
                    f.write(f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} 0.00 0.95 1.00\n")

                # Two distinct quads (4 triangles) per cross
                f.write(f"f {v_idx_base} {v_idx_base+1} {v_idx_base+2}\n")
                f.write(f"f {v_idx_base} {v_idx_base+2} {v_idx_base+3}\n")
                f.write(f"f {v_idx_base+4} {v_idx_base+5} {v_idx_base+6}\n")
                f.write(f"f {v_idx_base+4} {v_idx_base+6} {v_idx_base+7}\n")
                v_idx_base += 8
        else:
            # Traditional independent line segments
            for idx in sample_indices:
                c = face_centers[idx] + face_normals[idx] * (arm_len * 0.04)
                u = dir_u[idx] * arm_len
                v = dir_v[idx] * arm_len

                p_u_pos = c + u
                p_u_neg = c - u
                p_v_pos = c + v
                p_v_neg = c - v

                f.write(f"v {p_u_pos[0]:.6f} {p_u_pos[1]:.6f} {p_u_pos[2]:.6f} 0.00 0.95 1.00\n")
                f.write(f"v {p_u_neg[0]:.6f} {p_u_neg[1]:.6f} {p_u_neg[2]:.6f} 0.00 0.95 1.00\n")
                f.write(f"v {p_v_pos[0]:.6f} {p_v_pos[1]:.6f} {p_v_pos[2]:.6f} 0.00 0.95 1.00\n")
                f.write(f"v {p_v_neg[0]:.6f} {p_v_neg[1]:.6f} {p_v_neg[2]:.6f} 0.00 0.95 1.00\n")

                f.write(f"l {v_idx_base} {v_idx_base+1}\n")
                f.write(f"l {v_idx_base+2} {v_idx_base+3}\n")
                v_idx_base += 4

    print(f"[+] [STEP 3 COMPLETE] Saved cross-field inspection mesh to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 3: 4-RoSy Cross-Field Optimization")
    parser.add_argument("input", help="Input 3D mesh (.glb, .obj, .stl, .ply)")
    parser.add_argument("output", help="Output annotated OBJ path with cross lines")
    parser.add_argument("--cv-high", type=float, default=12.0, help="Convex ridge seed threshold (default: 12.0)")
    parser.add_argument("--cv-low", type=float, default=6.0, help="Convex ridge trace threshold (default: 6.0)")
    parser.add_argument("--cc-high", type=float, default=12.0, help="Concave valley seed threshold (default: 12.0)")
    parser.add_argument("--cc-low", type=float, default=6.0, help="Concave valley trace threshold (default: 6.0)")
    parser.add_argument("--density", type=int, default=2500, help="Target number of visual cross glyphs (default: 2500)")
    parser.add_argument("--mode", choices=["glyphs", "lines"], default="glyphs", help="Visual format: 'glyphs' (clean 3D cross ribbons, immune to web bugs) or 'lines'")
    args = parser.parse_args()

    compute_step3_cross_field(
        args.input, 
        args.output,
        cv_high=args.cv_high,
        cv_low=args.cv_low,
        cc_high=args.cc_high,
        cc_low=args.cc_low,
        line_density=args.density,
        render_mode=args.mode
    )
