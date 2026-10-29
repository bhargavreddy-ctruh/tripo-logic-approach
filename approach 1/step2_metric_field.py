import argparse
from collections import defaultdict, deque
import numpy as np
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

    # 1. Dihedral Angles
    dot_n = np.clip(np.sum(face_normals[adj_faces[:, 0]] * face_normals[adj_faces[:, 1]], axis=1), -1.0, 1.0)
    angles_deg = np.degrees(np.arccos(dot_n))
    
    # 2. Signed Convex / Concave Determination
    try:
        is_convex = mesh.face_adjacency_convex
    except Exception:
        edge_vec = mesh.vertices[adj_edges[:, 1]] - mesh.vertices[adj_edges[:, 0]]
        edge_dir = edge_vec / (np.linalg.norm(edge_vec, axis=1, keepdims=True) + 1e-12)
        n0 = face_normals[adj_faces[:, 0]]
        n1 = face_normals[adj_faces[:, 1]]
        cross_val = np.sum(np.cross(n0, n1) * edge_dir, axis=1)
        is_convex = cross_val >= 0.0

    # 3. Classify Seeds and Candidates
    cv_seeds = (is_convex) & (angles_deg >= convex_high)
    cv_cands = (is_convex) & (angles_deg >= convex_low)
    
    cc_seeds = (~is_convex) & (angles_deg >= concave_high)
    cc_cands = (~is_convex) & (angles_deg >= concave_low)
    
    # 4. Hysteresis Graph Propagation (Seed High, Trace Low)
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
    
    sharp_edges = set()
    cv_edges = set()
    cc_edges = set()

    for idx in active_cv:
        v0, v1 = int(adj_edges[idx][0]), int(adj_edges[idx][1])
        sharp_edges.add((v0, v1))
        cv_edges.add((v0, v1))

    for idx in active_cc:
        v0, v1 = int(adj_edges[idx][0]), int(adj_edges[idx][1])
        sharp_edges.add((v0, v1))
        cc_edges.add((v0, v1))

    return sharp_edges, cv_edges, cc_edges


def compute_step2_adaptive_metric(
    input_path, 
    output_path,
    target_quads=5000,
    min_rel_scale=0.15,
    max_rel_scale=2.5,
    cv_high=12.0, 
    cv_low=6.0,
    cc_high=12.0, 
    cc_low=6.0,
    gradient_limit=0.3
):
    print(f"[*] [STEP 2] Loading asset: {input_path}")
    mesh = trimesh.load(input_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    mesh.merge_vertices()
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, 'nondegenerate_faces'):
        mesh.update_faces(mesh.nondegenerate_faces())
        
    n_verts = len(mesh.vertices)
    n_faces = len(mesh.faces)
    print(f"    -> Mesh: {n_verts:,} vertices | {n_faces:,} faces")

    # 1. Base Target Edge Length based on total surface area and target quad budget
    total_area = float(mesh.area)
    # Area of one ideal square quad of edge length h is h^2. Total quads N ~ Area / h^2
    h_nominal = np.sqrt(total_area / float(target_quads))
    h_min = h_nominal * min_rel_scale
    h_max = h_nominal * max_rel_scale
    print(f"    -> Surface Area: {total_area:.4f}")
    print(f"    -> Sizing Budget: Nominal = {h_nominal:.5f} | Min = {h_min:.5f} | Max = {h_max:.5f}")

    # 2. Extract Step 1 Feature Creases (Convex + Concave Hysteresis)
    print("[*] Detecting locked boundary edges from Step 1...")
    sharp_edges, cv_edges, cc_edges = compute_signed_hysteresis_creases(
        mesh, convex_high=cv_high, convex_low=cv_low, concave_high=cc_high, concave_low=cc_low
    )
    
    locked_feature_verts = set()
    for v0, v1 in sharp_edges:
        locked_feature_verts.add(v0)
        locked_feature_verts.add(v1)

    print(f"    -> Convex edges: {len(cv_edges):,} | Concave valleys: {len(cc_edges):,}")
    print(f"    -> Feature-anchored vertices locked at minimum size: {len(locked_feature_verts):,}")

    # 3. Continuous Curvature Metric Estimation
    print("[*] Estimating local curvature variation across surfaces...")
    try:
        # Gaussian curvature (angle defects at vertices)
        defects = np.abs(trimesh.curvature.vertex_defects(mesh))
    except Exception:
        defects = np.zeros(n_verts)

    try:
        # Mean curvature / surface bending (angular divergence between vertex and face normals)
        faces = mesh.faces
        fn = mesh.face_normals
        vn = mesh.vertex_normals
        dots = np.sum(vn[faces] * fn[:, np.newaxis, :], axis=2)
        dev = 1.0 - np.clip(dots, -1.0, 1.0)
        vert_bend = np.zeros(n_verts)
        np.maximum.at(vert_bend, faces.reshape(-1), dev.reshape(-1))
    except Exception:
        vert_bend = np.zeros(n_verts)

    # Combined raw curvature (Gaussian defects + surface bending)
    curv = np.maximum(defects, vert_bend)

    # --- IMPROVED ADAPTIVE LOG-SCALING ---
    # 1. Logarithmic transform of curvature to prevent 95% red saturation
    curv_log = np.log1p(np.abs(curv))

    # Use robust percentile clipping (10th to 90th percentile)
    p10 = np.percentile(curv_log, 10)
    p90 = np.percentile(curv_log, 90)

    norm_curv = np.clip((curv_log - p10) / (p90 - p10 + 1e-12), 0.0, 1.0)

    # 2. Smoothly map to edge lengths (Flat areas get h_max, curved areas approach h_min)
    h_field = h_max - norm_curv * (h_max - h_min)

    # 3. Hard-lock ONLY the detected sharp crease edges (from Step 1) to h_min
    if len(locked_feature_verts) > 0:
        for v_idx in locked_feature_verts:
            h_field[v_idx] = h_min

    # 4. Metric Field Smoothing & Grading Propagation
    # Enforces |h_i - h_j| <= gradient_limit * dist(i, j)
    # High-performance vectorized Bellman-Ford style relaxation across mesh edges
    print(f"[*] Propagating sizing field grading (|grad(h)| <= {gradient_limit})...")
    edges_unique = mesh.edges_unique
    v_pos = mesh.vertices
    e0 = edges_unique[:, 0]
    e1 = edges_unique[:, 1]
    edge_lengths = np.linalg.norm(v_pos[e0] - v_pos[e1], axis=1)
    max_allowed_diff = gradient_limit * edge_lengths

    # Run grading relaxation until converged (or max 16 iterations)
    for it in range(16):
        h_prev = h_field.copy()
        # Propagate from smaller to larger symmetrically
        np.minimum.at(h_field, e1, h_field[e0] + max_allowed_diff)
        np.minimum.at(h_field, e0, h_field[e1] + max_allowed_diff)
        max_delta = np.max(np.abs(h_field - h_prev))
        if max_delta < 1e-6:
            print(f"    -> Grading converged in {it + 1} iterations.")
            break
    else:
        print(f"    -> Completed 16 relaxation iterations.")

    # 5. Visual Inspection Export
    # Heatmap:
    # Red/Yellow = Tiny Quads (features, concave part junctions, micro-bumps)
    # Green      = Medium Quads (gentle transitions)
    # Blue/Violet = Large Quads (flat spans, e.g. roofs, walls, expansive cushions)
    print(f"[*] Sizing range achieved: [{np.min(h_field):.5f}, {np.max(h_field):.5f}]")
    normalized_h = (h_field - h_min) / (h_max - h_min + 1e-12)
    normalized_h = np.clip(normalized_h, 0.0, 1.0)

    vertex_colors = np.zeros((n_verts, 4), dtype=np.uint8)
    vertex_colors[:, 3] = 255

    # Color gradient mapping:
    # 0.0 (Dense / Small h)  -> Pure Red (255, 0, 0)
    # 0.5 (Medium h)         -> Yellow/Green (128, 255, 0)
    # 1.0 (Sparse / Large h) -> Blue/Violet (0, 120, 255)
    vertex_colors[:, 0] = ((1.0 - normalized_h) * 255).astype(np.uint8)
    vertex_colors[:, 1] = (np.sin(normalized_h * np.pi) * 220).astype(np.uint8)
    vertex_colors[:, 2] = (normalized_h * 255).astype(np.uint8)

    mesh.visual.vertex_colors = vertex_colors
    mesh.export(output_path)

    # Save numeric scalar array for Step 3 and Step 4
    metric_npy_path = output_path.replace(".obj", "_metric.npy")
    np.save(metric_npy_path, h_field)
    print(f"[+] [STEP 2 COMPLETE] Output saved to: {output_path}")
    print(f"    -> Scalar metric data saved to: {metric_npy_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 2: Curvature-Adaptive Metric Field Sizing")
    parser.add_argument("input", help="Input mesh file path (.glb, .obj, .stl, .ply)")
    parser.add_argument("output", help="Output annotated OBJ path")
    parser.add_argument("--quads", type=int, default=5000, help="Target total quad budget (default: 5000)")
    parser.add_argument("--min-scale", type=float, default=0.15, help="Relative min edge scale (default 0.15 * nominal)")
    parser.add_argument("--max-scale", type=float, default=2.5, help="Relative max edge scale (default 2.5 * nominal)")
    parser.add_argument("--grad-limit", type=float, default=0.3, help="Grading factor (controls sizing transitions, default 0.3)")
    parser.add_argument("--cv-high", type=float, default=12.0, help="Convex ridge seed threshold (default: 12.0)")
    parser.add_argument("--cv-low", type=float, default=6.0, help="Convex ridge trace threshold (default: 6.0)")
    parser.add_argument("--cc-high", type=float, default=12.0, help="Concave valley seed threshold (default: 12.0)")
    parser.add_argument("--cc-low", type=float, default=6.0, help="Concave valley trace threshold (default: 6.0)")
    args = parser.parse_args()

    compute_step2_adaptive_metric(
        args.input, 
        args.output,
        target_quads=args.quads,
        min_rel_scale=args.min_scale,
        max_rel_scale=args.max_scale,
        cv_high=args.cv_high,
        cv_low=args.cv_low,
        cc_high=args.cc_high,
        cc_low=args.cc_low,
        gradient_limit=args.grad_limit
    )

