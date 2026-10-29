import argparse
from collections import defaultdict, deque
import numpy as np
import trimesh

def compute_signed_hysteresis_creases(
    mesh: trimesh.Trimesh, 
    convex_high=6.0, 
    convex_low=2.0, 
    concave_high=6.0, 
    concave_low=2.0
):
    """
    Module M06: Signed Convex/Concave Hysteresis Feature Detection.
    
    IMPORTANT NOTE ON THRESHOLDS:
    - 'high' is the SEED threshold (minimum angle to START a feature chain).
    - 'low' is the TRACE threshold (minimum angle to EXTEND an existing chain).
    - It is NOT a range [low, high]. Any angle >= high is automatically a seed!
    - On dense meshes (e.g. 500k-1M faces), a 90° turn is spread over 15+ triangles,
      so individual triangle-to-triangle angles are only 2° to 8°.
      Setting 'high=90°' results in 0 seeds!
    """
    # Guard against swapped or inverted thresholds
    if convex_high < convex_low:
        convex_high, convex_low = convex_low, convex_high
    if concave_high < concave_low:
        concave_high, concave_low = concave_low, concave_high

    # Guard against accidental seed suppression on dense meshes
    if convex_high >= 60.0:
        print(f"[!] Warning: convex_high={convex_high}° is too high for subdivided/dense meshes and will yield 0 seeds.")
        print(f"    Auto-capping seed trigger to 8.0° (trace floor: {convex_low}°).")
        convex_high = 8.0
    if concave_high >= 60.0:
        concave_high = 8.0

    print("[*] Running Module M06: Signed Hysteresis Feature Detection...")
    print(f"    -> Thresholds: Convex Seed={convex_high}°, Trace={convex_low}° | Concave Seed={concave_high}°, Trace={concave_low}°")
    
    face_normals = mesh.face_normals
    adj_faces = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges
    
    if len(adj_faces) == 0:
        print("    -> No face adjacency found.")
        return set(), {}, set(), set()

    # 1. Compute Dihedral Angles
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
    edge_types = {}  # 0: Smooth, 1: Convex Ridge, -1: Concave Valley
    
    for idx in range(len(adj_faces)):
        edge_types[idx] = 0

    for idx in active_cv:
        v0, v1 = int(adj_edges[idx][0]), int(adj_edges[idx][1])
        sharp_edges.add((v0, v1))
        cv_edges.add((v0, v1))
        edge_types[idx] = 1

    for idx in active_cc:
        v0, v1 = int(adj_edges[idx][0]), int(adj_edges[idx][1])
        sharp_edges.add((v0, v1))
        cc_edges.add((v0, v1))
        edge_types[idx] = -1

    rejected_noise = (cv_cands.sum() - len(active_cv)) + (cc_cands.sum() - len(active_cc))
    print(f"    -> Convex Ridges:    {len(active_cv):,} feature edges (seeds: {cv_seeds.sum():,}, candidates: {cv_cands.sum():,})")
    print(f"    -> Concave Valleys:  {len(active_cc):,} feature edges (seeds: {cc_seeds.sum():,}, candidates: {cc_cands.sum():,})")
    print(f"    -> Total Feature-Locked Edges: {len(sharp_edges):,} (rejected {rejected_noise:,} isolated noise edges)")
    
    return sharp_edges, edge_types, cv_edges, cc_edges


def analyze_asset(
    input_path, 
    output_path, 
    convex_high=6.0, 
    convex_low=2.0, 
    concave_high=6.0, 
    concave_low=2.0,
    crease_angle=None
):
    print(f"[*] Loading asset: {input_path}")
    mesh = trimesh.load(input_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    mesh.merge_vertices()
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, 'nondegenerate_faces'):
        mesh.update_faces(mesh.nondegenerate_faces())
    
    print(f"    -> Vertices: {len(mesh.vertices):,} | Faces: {len(mesh.faces):,}")

    # 1. Compute discrete curvatures (Gaussian angle defects + Mean normal bending)
    print("[*] Calculating surface curvatures for bump and rounded bevel preservation...")
    try:
        defects = np.abs(trimesh.curvature.vertex_defects(mesh))
    except Exception:
        defects = np.zeros(len(mesh.vertices))

    try:
        faces = mesh.faces
        fn = mesh.face_normals
        vn = mesh.vertex_normals
        dots = np.sum(vn[faces] * fn[:, np.newaxis, :], axis=2)
        dev = 1.0 - np.clip(dots, -1.0, 1.0)
        vert_bend = np.zeros(len(mesh.vertices))
        np.maximum.at(vert_bend, faces.reshape(-1), dev.reshape(-1))
    except Exception:
        vert_bend = np.zeros(len(mesh.vertices))

    # Normalize both curvature channels
    def norm_channel(arr):
        if len(arr) > 0 and np.ptp(arr) > 0:
            p99 = np.percentile(arr, 99.0)
            p_min = np.min(arr)
            denom = p99 - p_min if p99 > p_min else np.ptp(arr)
            return np.clip((arr - p_min) / denom, 0.0, 1.0)
        return np.zeros(len(mesh.vertices))

    norm_gauss = norm_channel(defects)
    norm_bend = norm_channel(vert_bend)
    curv_norm = np.maximum(norm_gauss, norm_bend)

    # 2. Module M06: Signed Hysteresis Crease Detection
    if crease_angle is not None:
        cv_h, cv_l = crease_angle, crease_angle * 0.4
        cc_h, cc_l = crease_angle, crease_angle * 0.4
    else:
        cv_h, cv_l = convex_high, convex_low
        cc_h, cc_l = concave_high, concave_low

    sharp_edges, edge_types, cv_edges, cc_edges = compute_signed_hysteresis_creases(
        mesh, convex_high=cv_h, convex_low=cv_l, concave_high=cc_h, concave_low=cc_l
    )

    # 3. Color-code the mesh for inspection
    # - Smooth/Flat: Green / muted slate
    # - Curvature bumps / wrinkles / bevels: Red/Orange gradient
    # - Convex Ridges: Bright Red ([255, 30, 30])
    # - Concave Valleys: Bright Electric Violet/Magenta ([255, 0, 220])
    vertex_colors = np.zeros((len(mesh.vertices), 4), dtype=np.uint8)
    vertex_colors[:, 3] = 255  # Alpha
    
    # Base gradient from combined curvature
    vertex_colors[:, 0] = (curv_norm * 255).astype(np.uint8)          # Red channel
    vertex_colors[:, 1] = ((1.0 - curv_norm) * 200).astype(np.uint8)   # Green channel
    vertex_colors[:, 2] = 50                                           # Blue baseline

    # Apply Convex Ridge Highlights (Bright Red)
    if len(cv_edges) > 0:
        cv_verts = np.unique(list(cv_edges))
        vertex_colors[cv_verts, 0] = 255
        vertex_colors[cv_verts, 1] = 30
        vertex_colors[cv_verts, 2] = 30

    # Apply Concave Valley Highlights (Bright Electric Violet / Magenta)
    if len(cc_edges) > 0:
        cc_verts = np.unique(list(cc_edges))
        vertex_colors[cc_verts, 0] = 255
        vertex_colors[cc_verts, 1] = 0
        vertex_colors[cc_verts, 2] = 220

    mesh.visual.vertex_colors = vertex_colors
    
    # Export inspected mesh
    mesh.export(output_path)
    print(f"[+] Step 1 Complete! Saved inspection mesh to: {output_path}")
    print("    -> Visual Guide:")
    print("       * Red / Bright Tones   : Convex feature ridges, bevels & surface bumps")
    print("       * Violet / Magenta     : Concave part-isolation valleys (locked boundary loops)")
    print("       * Green / Slate Muted  : Smooth flat / gentle organic surfaces (safe for quad decimation)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 1: Feature Crease & Curvature Analysis with Module M06 Signed Hysteresis")
    parser.add_argument("input", help="Input 3D asset (.obj, .glb, .stl, .ply)")
    parser.add_argument("output", help="Output annotated inspection mesh (.obj)")
    parser.add_argument("--convex-high", type=float, default=6.0, help="Convex ridge seed threshold in degrees (default: 6.0)")
    parser.add_argument("--convex-low", type=float, default=2.0, help="Convex ridge trace threshold in degrees (default: 2.0)")
    parser.add_argument("--concave-high", type=float, default=6.0, help="Concave valley seed threshold in degrees (default: 6.0)")
    parser.add_argument("--concave-low", type=float, default=2.0, help="Concave valley trace threshold in degrees (default: 2.0)")
    parser.add_argument("--preset", choices=["dense", "cad"], default="dense", help="Preset profile: 'dense' (default, 6/2) or 'cad' (35/22)")
    parser.add_argument("--crease-angle", type=float, default=None, help="Legacy single static angle override")
    args = parser.parse_args()
    
    if args.preset == "cad" and args.convex_high == 6.0:
        cv_h, cv_l, cc_h, cc_l = 35.0, 22.0, 26.0, 16.0
    else:
        cv_h, cv_l, cc_h, cc_l = args.convex_high, args.convex_low, args.concave_high, args.concave_low

    analyze_asset(
        args.input, 
        args.output, 
        convex_high=cv_h, 
        convex_low=cv_l,
        concave_high=cc_h, 
        concave_low=cc_l,
        crease_angle=args.crease_angle
    )
