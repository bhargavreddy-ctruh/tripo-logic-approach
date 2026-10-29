import argparse
from collections import defaultdict
import numpy as np
import networkx as nx
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
    """
    if convex_high < convex_low:
        convex_high, convex_low = convex_low, convex_high
    if concave_high < concave_low:
        concave_high, concave_low = concave_low, concave_high

    face_normals = mesh.face_normals
    adj_faces = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges
    
    if len(adj_faces) == 0:
        return set()

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
        from collections import deque
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
    return active_cv.union(active_cc)


def run_step4_quad_extraction(
    input_path, 
    output_path,
    field_npy_path=None,
    sharp_crease_deg=30.0,
    solver="auto",
    use_step1_creases=True
):
    print(f"[*] [STEP 4] Loading asset for Quad Extraction: {input_path}")
    raw_mesh = trimesh.load(input_path, force='mesh')
    if isinstance(raw_mesh, trimesh.Scene):
        raw_mesh = raw_mesh.dump(concatenate=True)
        
    raw_mesh.merge_vertices()
    if hasattr(raw_mesh, 'remove_degenerate_faces'):
        raw_mesh.remove_degenerate_faces()
    elif hasattr(raw_mesh, 'nondegenerate_faces'):
        raw_mesh.update_faces(raw_mesh.nondegenerate_faces())

    v = raw_mesh.vertices.copy()
    f = raw_mesh.faces.copy()
    face_normals = raw_mesh.face_normals
    face_centers = raw_mesh.triangles_center
    n_faces = len(f)

    print(f"    -> Mesh: {len(v):,} vertices | {n_faces:,} faces")

    # 1. Load Step 3 Cross-Field directions if provided
    has_field = False
    dir_u, dir_v = None, None
    if field_npy_path:
        try:
            field_data = np.load(field_npy_path)
            if field_data.shape[0] == n_faces:
                dir_u = field_data[:, 0, :]
                dir_v = field_data[:, 1, :]
                has_field = True
                print(f"    -> Successfully loaded Step 3 Flow Field: {field_data.shape}")
            else:
                print(f"    [!] Warning: Field shape {field_data.shape[0]} does not match mesh faces {n_faces:,}. Proceeding without field alignment.")
        except Exception as e:
            print(f"    [!] Warning: Could not load field ({e}). Proceeding with intrinsic geometry.")

    # 2. Extract Hard Crease Boundaries (Edges across which we CANNOT pair)
    print(f"[*] Locking hard feature creases (threshold: {sharp_crease_deg}°)...")
    cos_sharp = np.cos(np.radians(sharp_crease_deg))
    adj_faces = raw_mesh.face_adjacency
    adj_edges = raw_mesh.face_adjacency_edges
    unshared_v = raw_mesh.face_adjacency_unshared

    crease_adj_set = set()
    if use_step1_creases:
        try:
            crease_adj_set = compute_signed_hysteresis_creases(raw_mesh)
            print(f"    -> Locked {len(crease_adj_set):,} Step 1 hysteresis feature creases.")
        except Exception:
            pass

    # 3. Build Dual Face Pairing Graph
    print("[*] Evaluating triangle-to-quad candidate pairs...")
    candidate_edges = []
    G = nx.Graph()

    for idx, ((f0, f1), (e0, e1), (u0, u1)) in enumerate(zip(adj_faces, adj_edges, unshared_v)):
        # Condition A: Never pair across sharp creases or Step 1 feature lines
        if idx in crease_adj_set:
            continue
        dot_n = np.clip(np.dot(face_normals[f0], face_normals[f1]), -1.0, 1.0)
        if dot_n < cos_sharp:
            continue

        # Quad formed by: [u0, e0, u1, e1]
        q_idx = [int(u0), int(e0), int(u1), int(e1)]
        pts = v[q_idx]

        # Condition B: Interior angle orthogonality check
        edges = np.roll(pts, -1, axis=0) - pts
        lens = np.linalg.norm(edges, axis=1) + 1e-12
        dirs = edges / lens[:, None]

        # Dot products between consecutive edge vectors (ideal square = 0.0)
        cos_corners = [
            -np.dot(dirs[0], dirs[1]),
            -np.dot(dirs[1], dirs[2]),
            -np.dot(dirs[2], dirs[3]),
            -np.dot(dirs[3], dirs[0])
        ]
        max_deviation = np.max(np.abs(cos_corners))
        if max_deviation > 0.707:  # Reject corner angles < 45° or > 135°
            continue

        # Condition C: Alignment with Step 3 Flow Field
        align_score = 1.0
        if has_field and f0 < len(dir_u):
            u_ref = dir_u[f0]
            v_ref = dir_v[f0]
            alignments = [max(abs(np.dot(d, u_ref)), abs(np.dot(d, v_ref))) for d in dirs]
            align_score = float(np.mean(alignments))  # 0.707 to 1.0

        # Condition D: Aspect ratio / skewness
        aspect_ratio = np.min(lens) / np.max(lens)

        # Planarity penalty (angle between triangle normals)
        planarity = (dot_n + 1.0) * 0.5

        # Total pair quality score
        weight = float((1.0 - max_deviation) * 3.0 + align_score * 2.5 + aspect_ratio * 1.5 + planarity * 1.0)
        
        candidate_edges.append((int(f0), int(f1), weight, q_idx))
        G.add_edge(int(f0), int(f1), weight=weight, quad=q_idx)

    print(f"    -> Formed {len(candidate_edges):,} valid candidate quad pairs.")

    # 4. Maximum Weight Matching Solver
    choose_solver = solver
    if choose_solver == "auto":
        choose_solver = "blossom" if n_faces <= 25000 else "fast"

    print(f"[*] Solving maximum-weight graph matching (solver: {choose_solver})...")
    
    if choose_solver == "blossom":
        # Exact Edmonds' Blossom Algorithm via NetworkX
        matching = nx.max_weight_matching(G, maxcardinality=False)
        paired_faces = set()
        quad_faces = []

        for f0, f1 in matching:
            edge_data = G.get_edge_data(f0, f1)
            q = edge_data['quad']
            # Normal winding check
            n_est = np.cross(v[q[2]] - v[q[0]], v[q[3]] - v[q[1]])
            if np.dot(n_est, face_normals[f0] + face_normals[f1]) < 0:
                q = [q[0], q[3], q[2], q[1]]
            quad_faces.append(q)
            paired_faces.add(f0)
            paired_faces.add(f1)
    else:
        # High-Performance Priority Greedy Blossom-Approximation
        # Sort candidate pairs by descending quality weight
        candidate_edges.sort(key=lambda x: x[2], reverse=True)
        matched_flags = np.zeros(n_faces, dtype=bool)
        paired_faces = set()
        quad_faces = []

        for f0, f1, weight, q in candidate_edges:
            if not matched_flags[f0] and not matched_flags[f1]:
                matched_flags[f0] = True
                matched_flags[f1] = True
                paired_faces.add(f0)
                paired_faces.add(f1)
                
                n_est = np.cross(v[q[2]] - v[q[0]], v[q[3]] - v[q[1]])
                if np.dot(n_est, face_normals[f0] + face_normals[f1]) < 0:
                    q = [q[0], q[3], q[2], q[1]]
                quad_faces.append(q)

    # Collect remaining isolated triangles to maintain 100% watertight topology
    rem_tri_faces = [list(face) for i, face in enumerate(f) if i not in paired_faces]

    total_polys = len(quad_faces) + len(rem_tri_faces)
    quad_ratio = (len(quad_faces) / max(1, total_polys)) * 100.0

    print(f"    -> Extracted Polygons: {total_polys:,}")
    print(f"    -> Quads: {len(quad_faces):,} ({quad_ratio:.1f}%)")
    print(f"    -> Remainder Triangles: {len(rem_tri_faces):,} ({100.0 - quad_ratio:.1f}%)")

    # 5. Export Quad-Dominant OBJ
    print(f"[*] Writing structured quad mesh to: {output_path}")
    with open(output_path, 'w') as out_f:
        out_f.write("# Quad-Dominant Extracted Mesh (Step 4)\n")
        out_f.write(f"# Input: {input_path}\n")
        out_f.write(f"# Polygons: {total_polys:,} (Quads: {len(quad_faces):,}, Tris: {len(rem_tri_faces):,})\n")
        for vert in v:
            out_f.write(f"v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}\n")
        for q in quad_faces:
            out_f.write(f"f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n")
        for t in rem_tri_faces:
            out_f.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")

    print(f"[+] [STEP 4 COMPLETE] Saved quad mesh to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4: Quad Extraction via Blossom Graph Matching")
    parser.add_argument("input", help="Input mesh (.glb, .obj, .stl, .ply)")
    parser.add_argument("output", help="Output quad-dominant OBJ path")
    parser.add_argument("--field", default=None, help="Optional Step 3 .npy field path")
    parser.add_argument("--crease", type=float, default=30.0, help="Sharp crease angle limit in degrees (default: 30.0)")
    parser.add_argument("--solver", choices=["auto", "blossom", "fast"], default="auto", help="Matching solver: 'auto', 'blossom' (exact), or 'fast' (greedy priority)")
    args = parser.parse_args()

    run_step4_quad_extraction(
        args.input, 
        args.output, 
        field_npy_path=args.field, 
        sharp_crease_deg=args.crease,
        solver=args.solver
    )

