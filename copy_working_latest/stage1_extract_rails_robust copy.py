import argparse
from collections import defaultdict, deque
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
import trimesh
import networkx as nx

def extract_boundary_rails_ribbons(
    input_path, 
    output_rails_path, 
    output_annotated_path=None,
    cv_high=18.0, 
    cv_low=10.0,
    cc_high=16.0, 
    cc_low=9.0,
    ribbon_width=0.0,
    include_boundaries=True,
    min_chain_len=4
):
    print(f"[*] [STAGE 1 MULTI-SCALE] Loading asset: {input_path}")
    mesh = trimesh.load(input_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # 1. Clean manifold geometry & weld spatial coordinate duplicates (e.g. UV seams, split normals)
    raw_verts_count = len(mesh.vertices)
    _, u_idx, inv = np.unique(np.round(mesh.vertices, 6), axis=0, return_index=True, return_inverse=True)
    mesh = trimesh.Trimesh(vertices=mesh.vertices[u_idx], faces=inv[mesh.faces], process=True)
    mesh.merge_vertices()
    if hasattr(mesh, 'remove_duplicate_faces'):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, 'nondegenerate_faces'):
        mesh.update_faces(mesh.nondegenerate_faces())

    welded_verts_count = len(mesh.vertices)
    print(f"    -> Coordinate welding: {raw_verts_count:,} -> {welded_verts_count:,} vertices ({raw_verts_count - welded_verts_count:,} duplicates eliminated)")

    # Preserve 100% of all components (no deletion of small appendages, antennae, gems, hatches)

    v_pos = mesh.vertices
    n_verts = len(v_pos)
    face_normals = mesh.face_normals
    adj_faces = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges

    print(f"    -> Working Mesh: {n_verts:,} vertices | {len(mesh.faces):,} faces")

    # 2. Extract Open Perimeter / Boundary Edges (cutting edges, razor borders, panel seams)
    edge_to_faces = defaultdict(list)
    for f_idx, face in enumerate(mesh.faces):
        for i in range(3):
            e = tuple(sorted((face[i], face[(i + 1) % 3])))
            edge_to_faces[e].append(f_idx)

    open_boundary_edges = []
    if include_boundaries:
        for e, flist in edge_to_faces.items():
            if len(flist) == 1:
                open_boundary_edges.append((e, flist[0]))
        print(f"    -> Open Boundary / Seam Edges: {len(open_boundary_edges):,} edges locked as mandatory rails.")

    # 3. Signed Hysteresis with Multi-Scale Dynamic Adaptation for Elongated Edges
    print("[*] Running Multi-Scale Signed Hysteresis Edge Classification...")
    avg_edge_len = float(np.mean(mesh.edges_unique_length))
    print(f"    -> Base Thresholds: Convex={cv_high}°/{cv_low}° | Concave={cc_high}°/{cc_low}° | Avg Edge: {avg_edge_len:.5f}")

    raw_candidates = []
    vert_to_edge_map = defaultdict(list)

    for idx, ((f0, f1), (v0, v1)) in enumerate(zip(adj_faces, adj_edges)):
        n0 = face_normals[f0]
        n1 = face_normals[f1]
        dot_n = np.clip(np.dot(n0, n1), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot_n))

        edge_vec = v_pos[v1] - v_pos[v0]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-12:
            continue
        edge_dir = edge_vec / edge_len

        # Length-Adaptive Boost for structural edges (blade spines, car panels)
        length_boost = float(np.clip(edge_len / (avg_edge_len + 1e-12), 1.0, 2.0))
        ad_cv_high = max(cv_low, cv_high / length_boost)
        ad_cv_low  = max(cv_low * 0.75, cv_low / length_boost)
        ad_cc_high = max(cc_low, cc_high / length_boost)
        ad_cc_low  = max(cc_low * 0.75, cc_low / length_boost)

        # Signed check: Outward convex ridge (> 0) vs. Inward concave valley (< 0)
        signed_sin = np.dot(np.cross(n0, n1), edge_dir)
        is_convex = signed_sin >= 0.0

        raw_candidates.append({
            'edge_idx': idx,
            'verts': (int(v0), int(v1)),
            'f0': f0,
            'f1': f1,
            'angle_deg': angle_deg,
            'is_convex': is_convex,
            'edge_dir': edge_dir,
            'seed_thresh': ad_cv_high if is_convex else ad_cc_high,
            'trace_thresh': ad_cv_low if is_convex else ad_cc_low,
        })
        vert_to_edge_map[v0].append(idx)
        vert_to_edge_map[v1].append(idx)

    # Propagate collinear hysteresis using O(1) deque
    locked_edges = set()
    seed_queue = deque()
    edge_lookup = {c['edge_idx']: c for c in raw_candidates}

    for c in raw_candidates:
        if c['angle_deg'] >= c['seed_thresh']:
            locked_edges.add(c['edge_idx'])
            seed_queue.append(c['edge_idx'])

    print(f"    -> Initial High-Confidence Seeds: {len(seed_queue):,} edges")

    while seed_queue:
        curr_e = seed_queue.popleft()
        curr_data = edge_lookup[curr_e]
        t_curr = curr_data['edge_dir']
        v0, v1 = curr_data['verts']

        for nbr_e in vert_to_edge_map[v0] + vert_to_edge_map[v1]:
            if nbr_e not in locked_edges:
                nbr_data = edge_lookup[nbr_e]
                if nbr_data['angle_deg'] >= nbr_data['trace_thresh']:
                    # Smooth turning around circular arcs and cylinders (>= 0.50, turn up to 60 deg)
                    if abs(float(np.dot(t_curr, nbr_data['edge_dir']))) >= 0.50:
                        locked_edges.add(nbr_e)
                        seed_queue.append(nbr_e)

    print(f"    -> After Collinear Tracing: {len(locked_edges):,} edges")

    # Topological Collinear Gap Bridging on Dead Ends (bridges 1-step and 2-step staggered facet gaps)
    def get_dead_ends(edges_set):
        deg = defaultdict(int)
        for e in edges_set:
            u, v = edge_lookup[e]['verts']
            deg[u] += 1
            deg[v] += 1
        return [v for v, d in deg.items() if d == 1]

    total_bridged = 0
    for pass_idx in range(2):
        dead_ends = get_dead_ends(locked_edges)
        v_to_locked = defaultdict(list)
        for e in locked_edges:
            u, v = edge_lookup[e]['verts']
            v_to_locked[u].append(e)
            v_to_locked[v].append(e)

        pass_bridged = 0
        for vA in dead_ends:
            inc = v_to_locked[vA]
            if not inc:
                continue
            tA = edge_lookup[inc[0]]['edge_dir']
            for nbr_e in vert_to_edge_map[vA]:
                if nbr_e not in locked_edges:
                    nbr_data = edge_lookup[nbr_e]
                    if nbr_data['angle_deg'] >= 14.0:
                        if abs(float(np.dot(tA, nbr_data['edge_dir']))) >= 0.60:
                            locked_edges.add(nbr_e)
                            pass_bridged += 1
        total_bridged += pass_bridged

    print(f"    -> Topological Gap Bridging: {total_bridged:,} micro-gaps closed across dead ends.")

    # Filter short disconnected noise blips (chains < min_chain_len)
    locked_convex_edges = set()
    locked_concave_edges = set()

    if min_chain_len > 1 and len(locked_edges) > 0:
        rail_graph = nx.Graph()
        for e_idx in locked_edges:
            v0, v1 = edge_lookup[e_idx]['verts']
            rail_graph.add_edge(int(v0), int(v1), edge_idx=e_idx)

        pruned_locked = set()
        for comp in nx.connected_components(rail_graph):
            sub = rail_graph.subgraph(comp)
            if sub.number_of_edges() >= min_chain_len:
                for u, v, d in sub.edges(data=True):
                    pruned_locked.add(d['edge_idx'])

        pruned_count = len(locked_edges) - len(pruned_locked)
        print(f"    -> Pruned {pruned_count:,} short noise blips (< {min_chain_len} connected edges).")
        locked_edges = pruned_locked

    for e in locked_edges:
        if edge_lookup[e]['is_convex']:
            locked_convex_edges.add(e)
        else:
            locked_concave_edges.add(e)

    print(f"    -> Locked Convex Ridges (Red):   {len(locked_convex_edges):,} edges")
    print(f"    -> Locked Concave Valleys (Gold): {len(locked_concave_edges):,} edges")

    # 4. Build 3D Solid Geometry (Ribbon Quads) - ZERO WEBS / PHANTOM WIRES
    print("[*] Generating solid 3D ribbon strips for boundary rails (eliminates webs)...")
    ribbon_verts = []
    ribbon_faces = []
    ribbon_colors = []

    # Characteristic width based on mesh scale
    w = ribbon_width if ribbon_width > 0 else avg_edge_len * 0.35
    print(f"    -> Ribbon Width: {w:.6f} (normal lift offset: {w * 0.2:.6f})")

    # 4a. Add Internal Crease Ribbons (Convex and Concave)
    internal_locked = [(e, True) for e in locked_convex_edges] + [(e, False) for e in locked_concave_edges]

    for e_idx, is_convex in internal_locked:
        data = edge_lookup[e_idx]
        v0, v1 = data['verts']
        p0 = v_pos[v0]
        p1 = v_pos[v1]

        edge_dir = p1 - p0
        edge_len = np.linalg.norm(edge_dir)
        if edge_len < 1e-12:
            continue
        edge_dir /= edge_len

        # Combined face normal
        avg_n = face_normals[data['f0']] + face_normals[data['f1']]
        norm_val = np.linalg.norm(avg_n)
        if norm_val < 1e-12:
            continue
        avg_n /= norm_val

        # Slight normal offset to prevent z-fighting with the underlying surface
        offset = avg_n * (w * 0.2)
        p0_lift = p0 + offset
        p1_lift = p1 + offset

        # Lateral vector perpendicular to edge in the surface tangent plane
        lateral = np.cross(avg_n, edge_dir)
        lateral_len = np.linalg.norm(lateral)
        if lateral_len < 1e-12:
            continue
        lateral = (lateral / lateral_len) * (w * 0.5)

        q0 = p0_lift - lateral
        q1 = p1_lift - lateral
        q2 = p1_lift + lateral
        q3 = p0_lift + lateral

        base_v = len(ribbon_verts)
        ribbon_verts.extend([q0, q1, q2, q3])
        ribbon_faces.append([base_v, base_v + 1, base_v + 2])
        ribbon_faces.append([base_v, base_v + 2, base_v + 3])

        color = [255, 30, 30, 255] if is_convex else [255, 215, 0, 255]
        ribbon_colors.extend([color, color, color, color])

    # 4b. Add Open Boundary / Cutting Edge Ribbons (Electric Cyan)
    for (v0, v1), f0 in open_boundary_edges:
        p0 = v_pos[v0]
        p1 = v_pos[v1]

        edge_dir = p1 - p0
        edge_len = np.linalg.norm(edge_dir)
        if edge_len < 1e-12:
            continue
        edge_dir /= edge_len

        n0 = face_normals[f0]
        offset = n0 * (w * 0.2)
        p0_lift = p0 + offset
        p1_lift = p1 + offset

        lateral = np.cross(n0, edge_dir)
        lateral_len = np.linalg.norm(lateral)
        if lateral_len < 1e-12:
            continue
        lateral = (lateral / lateral_len) * (w * 0.5)

        q0 = p0_lift - lateral
        q1 = p1_lift - lateral
        q2 = p1_lift + lateral
        q3 = p0_lift + lateral

        base_v = len(ribbon_verts)
        ribbon_verts.extend([q0, q1, q2, q3])
        ribbon_faces.append([base_v, base_v + 1, base_v + 2])
        ribbon_faces.append([base_v, base_v + 2, base_v + 3])

        # Electric Cyan for open borders / cutting edges: [0, 220, 255]
        cyan = [0, 220, 255, 255]
        ribbon_colors.extend([cyan, cyan, cyan, cyan])

    total_ribbons = len(internal_locked) + len(open_boundary_edges)
    if len(ribbon_verts) == 0:
        print("[!] Warning: No boundary rails detected. Exporting empty ribbon mesh.")
        ribbon_mesh = trimesh.Trimesh()
    else:
        ribbon_mesh = trimesh.Trimesh(
            vertices=np.array(ribbon_verts, dtype=np.float64),
            faces=np.array(ribbon_faces, dtype=np.int64),
            vertex_colors=np.array(ribbon_colors, dtype=np.uint8),
            process=False
        )

    ribbon_mesh.export(output_rails_path)
    print(f"[+] [STAGE 1 COMPLETE] Clean 3D ribbon rails exported to: {output_rails_path}")
    if output_rails_path.endswith('.obj'):
        rails_glb = output_rails_path.replace('.obj', '.glb')
        try:
            ribbon_mesh.export(rails_glb)
            print(f"    -> Also exported rails GLB companion: {rails_glb}")
        except Exception:
            pass
    print(f"    -> Total Rails: {total_ribbons:,} (Convex: {len(locked_convex_edges):,}, Concave: {len(locked_concave_edges):,}, Open Borders: {len(open_boundary_edges):,})")

    # 5. Export Annotated base mesh for inspection if requested
    if output_annotated_path:
        print(f"[*] Exporting annotated base inspection mesh to: {output_annotated_path}")
        vertex_colors = np.zeros((n_verts, 4), dtype=np.uint8)
        vertex_colors[:, :] = [180, 180, 180, 255]  # Neutral grey base

        for e_idx in locked_concave_edges:
            v0, v1 = edge_lookup[e_idx]['verts']
            vertex_colors[v0] = [255, 215, 0, 255]
            vertex_colors[v1] = [255, 215, 0, 255]

        for e_idx in locked_convex_edges:
            v0, v1 = edge_lookup[e_idx]['verts']
            vertex_colors[v0] = [255, 30, 30, 255]
            vertex_colors[v1] = [255, 30, 30, 255]

        for (v0, v1), _ in open_boundary_edges:
            vertex_colors[v0] = [0, 220, 255, 255]
            vertex_colors[v1] = [0, 220, 255, 255]

        mesh.visual = trimesh.visual.ColorVisuals(mesh=mesh, vertex_colors=vertex_colors)
        mesh.export(output_annotated_path)
        print(f"    -> Annotated base mesh saved to: {output_annotated_path}")

        if output_annotated_path.endswith('.obj'):
            glb_path = output_annotated_path.replace('.obj', '.glb')
            try:
                mesh.export(glb_path)
                print(f"    -> Also exported GLB companion: {glb_path}")
            except Exception:
                pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Multi-Scale Adaptive 3D Ribbon Boundary Rails")
    parser.add_argument("input", help="Input mesh (.obj, .glb, .stl, .ply)")
    parser.add_argument("output_rails", help="Output ribbon rails OBJ path")
    parser.add_argument("--annotated", default=None, help="Optional annotated base mesh OBJ path")
    parser.add_argument("--cv-high", type=float, default=18.0, help="Convex ridge seed threshold in degrees (default: 18.0)")
    parser.add_argument("--cv-low", type=float, default=10.0, help="Convex ridge chain threshold in degrees (default: 10.0)")
    parser.add_argument("--cc-high", type=float, default=16.0, help="Concave valley seed threshold in degrees (default: 16.0)")
    parser.add_argument("--cc-low", type=float, default=9.0, help="Concave valley chain threshold in degrees (default: 9.0)")
    parser.add_argument("--min-chain", type=int, default=4, help="Minimum connected chain length (default: 4)")
    parser.add_argument("--width", type=float, default=0.0, help="Ribbon width in world units (default: 0.0 for auto)")
    parser.add_argument("--preset", choices=["cad", "dense"], default=None, help="Threshold preset: 'cad' (35/22/26/16) or 'dense' (8/3/8/3)")
    parser.add_argument("--no-boundaries", action="store_true", help="Disable open perimeter / seam edge inclusion")
    args = parser.parse_args()

    cv_h, cv_l = args.cv_high, args.cv_low
    cc_h, cc_l = args.cc_high, args.cc_low

    if args.preset == "dense":
        cv_h, cv_l, cc_h, cc_l = 8.0, 3.0, 8.0, 3.0
    elif args.preset == "cad":
        cv_h, cv_l, cc_h, cc_l = 35.0, 22.0, 26.0, 16.0

    extract_boundary_rails_ribbons(
        args.input, 
        args.output_rails, 
        args.annotated,
        cv_high=cv_h, 
        cv_low=cv_l,
        cc_high=cc_h, 
        cc_low=cc_l,
        ribbon_width=args.width,
        include_boundaries=(not args.no_boundaries),
        min_chain_len=args.min_chain
    )
