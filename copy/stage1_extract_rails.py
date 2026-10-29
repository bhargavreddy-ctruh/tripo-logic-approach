import argparse
from collections import defaultdict, deque
import numpy as np
import trimesh

def extract_boundary_rails(
    input_path, 
    output_rails_path, 
    output_annotated_path=None,
    cv_high=35.0, 
    cv_low=22.0,
    cc_high=26.0, 
    cc_low=16.0,
    min_loop_length=3
):
    print(f"[*] [STAGE 1] Loading asset: {input_path}")
    mesh = trimesh.load(input_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # 1. Manifold Sanitization
    mesh.merge_vertices()
    if hasattr(mesh, 'remove_duplicate_faces'):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, 'nondegenerate_faces'):
        mesh.update_faces(mesh.nondegenerate_faces())

    # Isolate main components (drop microscopic floating noise)
    try:
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            max_faces = max(len(c.faces) for c in components)
            valid_components = [c for c in components if len(c.faces) >= max_faces * 0.01]
            mesh = trimesh.util.concatenate(valid_components)
            print(f"    -> Filtered floating noise. Working faces: {len(mesh.faces):,}")
    except Exception as e:
        print(f"    -> Component splitting skipped ({e}).")

    v_pos = mesh.vertices
    n_verts = len(v_pos)
    face_normals = mesh.face_normals
    adj_faces = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges

    print(f"    -> Mesh Loaded: {n_verts:,} vertices | {len(mesh.faces):,} faces")

    # 2. Signed Convex / Concave Hysteresis Edge Detection
    print("[*] Running Signed Hysteresis Edge Classification...")
    print(f"    -> Convex: Seed={cv_high}° / Trace={cv_low}° | Concave: Seed={cc_high}° / Trace={cc_low}°")
    cos_cv_high = np.cos(np.radians(cv_high))
    cos_cv_low  = np.cos(np.radians(cv_low))
    cos_cc_high = np.cos(np.radians(cc_high))
    cos_cc_low  = np.cos(np.radians(cc_low))

    raw_candidates = []
    vert_to_edge_map = defaultdict(list)
    edge_to_adjacent_edges = defaultdict(list)

    for idx, ((f0, f1), (v0, v1)) in enumerate(zip(adj_faces, adj_edges)):
        n0 = face_normals[f0]
        n1 = face_normals[f1]
        dot_n = np.clip(np.dot(n0, n1), -1.0, 1.0)

        edge_vec = v_pos[v1] - v_pos[v0]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-12:
            continue
        edge_dir = edge_vec / edge_len

        # Outward convex ridge (> 0) vs. Inward concave valley (< 0)
        signed_sin = np.dot(np.cross(n0, n1), edge_dir)
        is_convex = signed_sin >= 0.0

        raw_candidates.append({
            'edge_idx': idx,
            'verts': (int(v0), int(v1)),
            'dot_n': dot_n,
            'is_convex': is_convex
        })
        vert_to_edge_map[v0].append(idx)
        vert_to_edge_map[v1].append(idx)

    for v, edges in vert_to_edge_map.items():
        for e1 in edges:
            for e2 in edges:
                if e1 != e2:
                    edge_to_adjacent_edges[e1].append(e2)

    # Seed and propagate hysteresis chains using O(1) deque
    locked_convex_edges = set()
    locked_concave_edges = set()
    seed_queue = deque()
    edge_lookup = {c['edge_idx']: c for c in raw_candidates}

    for c in raw_candidates:
        e_idx = c['edge_idx']
        dot_n = c['dot_n']
        if c['is_convex']:
            if dot_n <= cos_cv_high:
                locked_convex_edges.add(e_idx)
                seed_queue.append(e_idx)
        else:
            if dot_n <= cos_cc_high:
                locked_concave_edges.add(e_idx)
                seed_queue.append(e_idx)

    while seed_queue:
        curr_e = seed_queue.popleft()
        curr_data = edge_lookup[curr_e]
        curr_is_convex = curr_data['is_convex']

        for nbr_e in edge_to_adjacent_edges[curr_e]:
            nbr_data = edge_lookup[nbr_e]
            if nbr_data['is_convex'] == curr_is_convex:
                if curr_is_convex:
                    if nbr_e not in locked_convex_edges and nbr_data['dot_n'] <= cos_cv_low:
                        locked_convex_edges.add(nbr_e)
                        seed_queue.append(nbr_e)
                else:
                    if nbr_e not in locked_concave_edges and nbr_data['dot_n'] <= cos_cc_low:
                        locked_concave_edges.add(nbr_e)
                        seed_queue.append(nbr_e)

    all_locked_edges = locked_convex_edges | locked_concave_edges
    print(f"    -> Extracted {len(locked_convex_edges):,} convex edges and {len(locked_concave_edges):,} concave edges.")

    # 3. Assemble Continuous 1D Rail Loops & Chains
    print("[*] Assembling 1D topological rail networks...")
    # Build graph of feature edges
    graph_adj = defaultdict(set)
    for e_idx in all_locked_edges:
        v0, v1 = edge_lookup[e_idx]['verts']
        graph_adj[v0].add(v1)
        graph_adj[v1].add(v0)

    visited_edges = set()
    chains = []

    # 3a. Trace open chains starting at endpoints or branching nodes (degree != 2)
    for start_v in list(graph_adj.keys()):
        degree = len(graph_adj[start_v])
        if degree != 2:
            for next_v in list(graph_adj[start_v]):
                edge_key = tuple(sorted((start_v, next_v)))
                if edge_key in visited_edges:
                    continue

                chain = [start_v, next_v]
                visited_edges.add(edge_key)
                curr_node = next_v

                while True:
                    unvisited_neighbors = [
                        nbr for nbr in graph_adj[curr_node]
                        if tuple(sorted((curr_node, nbr))) not in visited_edges
                    ]
                    # Continue if path is unambiguous (degree 2 along chain)
                    if len(unvisited_neighbors) == 1 and len(graph_adj[curr_node]) == 2:
                        next_step = unvisited_neighbors[0]
                        visited_edges.add(tuple(sorted((curr_node, next_step))))
                        chain.append(next_step)
                        curr_node = next_step
                    else:
                        break
                
                if len(chain) >= min_loop_length:
                    chains.append(chain)

    # 3b. Trace remaining closed loops (all nodes degree == 2)
    for start_v in list(graph_adj.keys()):
        for next_v in list(graph_adj[start_v]):
            edge_key = tuple(sorted((start_v, next_v)))
            if edge_key in visited_edges:
                continue

            chain = [start_v, next_v]
            visited_edges.add(edge_key)
            curr_node = next_v

            while True:
                unvisited_neighbors = [
                    nbr for nbr in graph_adj[curr_node]
                    if tuple(sorted((curr_node, nbr))) not in visited_edges
                ]
                if len(unvisited_neighbors) == 1 and len(graph_adj[curr_node]) == 2:
                    next_step = unvisited_neighbors[0]
                    visited_edges.add(tuple(sorted((curr_node, next_step))))
                    chain.append(next_step)
                    curr_node = next_step
                    if curr_node == start_v:
                        break
                else:
                    break

            if len(chain) >= min_loop_length:
                chains.append(chain)

    closed_loops = sum(1 for c in chains if c[0] == c[-1])
    print(f"    -> Extracted {len(chains):,} continuous 1D boundary rails ({closed_loops:,} closed loops).")

    # 4. Export Standalone 1D Boundary Rails as Polyline OBJ
    print(f"[*] Exporting boundary rails to: {output_rails_path}")
    vert_map = {}

    with open(output_rails_path, 'w') as f_rail:
        f_rail.write("# Stage 1: 1D Boundary Rail Network\n")
        f_rail.write(f"# Input: {input_path}\n")
        f_rail.write(f"# Rails: {len(chains):,} (Closed Loops: {closed_loops:,})\n")
        
        # Write unique vertices referenced in chains
        for chain in chains:
            for v_idx in chain:
                if v_idx not in vert_map:
                    vert_map[v_idx] = len(vert_map) + 1
                    pos = v_pos[v_idx]
                    f_rail.write(f"v {pos[0]:.6f} {pos[1]:.6f} {pos[2]:.6f}\n")

        # Write line elements
        for chain in chains:
            line_indices = [str(vert_map[v]) for v in chain]
            f_rail.write(f"l {' '.join(line_indices)}\n")

    # 5. Optional Annotated Mesh Export (for visual inspection of classifications)
    if output_annotated_path:
        print(f"[*] Exporting annotated inspection mesh to: {output_annotated_path}")
        vertex_colors = np.zeros((n_verts, 4), dtype=np.uint8)
        vertex_colors[:, :] = [180, 180, 180, 255]  # Neutral grey base

        # Concave loops (Gold: [255, 215, 0])
        for e_idx in locked_concave_edges:
            v0, v1 = edge_lookup[e_idx]['verts']
            vertex_colors[v0] = [255, 215, 0, 255]
            vertex_colors[v1] = [255, 215, 0, 255]

        # Convex loops (Red: [255, 30, 30])
        for e_idx in locked_convex_edges:
            v0, v1 = edge_lookup[e_idx]['verts']
            vertex_colors[v0] = [255, 30, 30, 255]
            vertex_colors[v1] = [255, 30, 30, 255]

        mesh.visual.vertex_colors = vertex_colors
        mesh.export(output_annotated_path)

    print(f"[+] [STAGE 1 COMPLETE] Rails saved to: {output_rails_path}")
    if output_annotated_path:
        print(f"    -> Annotated inspection mesh saved to: {output_annotated_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: 1D Boundary Rail Network Extraction")
    parser.add_argument("input", help="Input mesh file path (.obj, .glb, .stl, .ply)")
    parser.add_argument("output_rails", help="Output boundary rails OBJ path")
    parser.add_argument("--annotated", default=None, help="Optional output annotated mesh OBJ path")
    parser.add_argument("--cv-high", type=float, default=35.0, help="Convex ridge seed threshold in degrees (default: 35.0)")
    parser.add_argument("--cv-low", type=float, default=22.0, help="Convex ridge chain threshold in degrees (default: 22.0)")
    parser.add_argument("--cc-high", type=float, default=26.0, help="Concave valley seed threshold in degrees (default: 26.0)")
    parser.add_argument("--cc-low", type=float, default=16.0, help="Concave valley chain threshold in degrees (default: 16.0)")
    parser.add_argument("--min-len", type=int, default=3, help="Minimum loop/chain length in edges (default: 3)")
    parser.add_argument("--preset", choices=["cad", "dense"], default=None, help="Threshold preset: 'cad' (35/22/26/16) or 'dense' (8/3/8/3)")
    args = parser.parse_args()

    cv_h, cv_l = args.cv_high, args.cv_low
    cc_h, cc_l = args.cc_high, args.cc_low

    if args.preset == "dense":
        cv_h, cv_l, cc_h, cc_l = 8.0, 3.0, 8.0, 3.0
    elif args.preset == "cad":
        cv_h, cv_l, cc_h, cc_l = 35.0, 22.0, 26.0, 16.0

    extract_boundary_rails(
        args.input, 
        args.output_rails, 
        args.annotated,
        cv_high=cv_h, 
        cv_low=cv_l,
        cc_high=cc_h, 
        cc_low=cc_l,
        min_loop_length=args.min_len
    )

