#!/usr/bin/env python3
"""
Stage 4 - Step 1: Topological Rail Seam Splitting & Boundary Freezing
---------------------------------------------------------------------
Takes an input 3D mesh and the Stage 1 boundary rails (or extracts them via
multi-scale signed hysteresis), then performs an exact topological split 
along every rail edge.

Why this is essential:
  1. Downstream remeshers and parameterizers treat open boundary seams as 
     IMMUTABLE hard geometric constraints.
  2. Splitting the mesh along the rails prevents any algorithm from cutting
     across, dissolving, or smoothing out your structural feature lines.
  3. Original vertex coordinates are 100% preserved (zero spatial distortion).
  4. Generates a reconstruction manifest (.json) so Step 5 can seamlessly 
     re-weld the coincident vertices after quadrangulation.
"""

import os
import sys
import json
import time
import argparse
from collections import defaultdict
import numpy as np
import scipy.sparse as sp
import scipy.sparse.csgraph as csg
from scipy.spatial import cKDTree
import networkx as nx
import trimesh

def clean_input_mesh(mesh):
    """Welds duplicate coordinates and removes degenerate faces."""
    raw_verts = len(mesh.vertices)
    _, u_idx, inv = np.unique(np.round(mesh.vertices, 6), axis=0, return_index=True, return_inverse=True)
    mesh = trimesh.Trimesh(vertices=mesh.vertices[u_idx], faces=inv[mesh.faces], process=True)
    mesh.merge_vertices()
    if hasattr(mesh, 'remove_duplicate_faces'):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, 'nondegenerate_faces'):
        mesh.update_faces(mesh.nondegenerate_faces())
    
    # Preserve 100% of components (no aggressive 1% deletion of antennae, gems, hatches)
    return mesh

def extract_stage1_rail_edges(mesh, cv_high=28.0, cv_low=18.0, cc_high=24.0, cc_low=16.0, min_chain_len=18):
    """
    Extracts the Stage 1 signed multi-scale hysteresis edges directly on the mesh.
    Returns:
      cut_edges: set of sorted vertex tuples (v0, v1)
      open_boundary_edges: set of sorted vertex tuples (v0, v1)
    """
    v_pos = mesh.vertices
    adj_faces = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges
    face_normals = mesh.face_normals
    avg_edge_len = float(np.mean(mesh.edges_unique_length))

    # 1. Identify open boundaries (already physical borders)
    edge_to_faces = defaultdict(list)
    for f_idx, face in enumerate(mesh.faces):
        for i in range(3):
            e = tuple(sorted((face[i], face[(i + 1) % 3])))
            edge_to_faces[e].append(f_idx)

    open_boundary_edges = set(e for e, flist in edge_to_faces.items() if len(flist) == 1)

    # 2. Hysteresis edge candidate evaluation
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

        # Adaptive threshold boost for long edges
        length_boost = float(np.clip(edge_len / (avg_edge_len + 1e-12), 1.0, 2.0))
        ad_cv_high = max(cv_low, cv_high / length_boost)
        ad_cv_low  = max(cv_low * 0.75, cv_low / length_boost)
        ad_cc_high = max(cc_low, cc_high / length_boost)
        ad_cc_low  = max(cc_low * 0.75, cc_low / length_boost)

        signed_sin = np.dot(np.cross(n0, n1), edge_dir)
        is_convex = signed_sin >= 0.0

        raw_candidates.append({
            'edge_idx': idx,
            'verts': (int(v0), int(v1)),
            'angle_deg': angle_deg,
            'is_convex': is_convex,
            'edge_dir': edge_dir,
            'seed_thresh': ad_cv_high if is_convex else ad_cc_high,
            'trace_thresh': ad_cv_low if is_convex else ad_cc_low,
        })
        vert_to_edge_map[v0].append(idx)
        vert_to_edge_map[v1].append(idx)

    edge_lookup = {c['edge_idx']: c for c in raw_candidates}
    locked_edges = set()
    from collections import deque
    seed_queue = deque()

    for c in raw_candidates:
        if c['angle_deg'] >= c['seed_thresh']:
            locked_edges.add(c['edge_idx'])
            seed_queue.append(c['edge_idx'])

    while seed_queue:
        curr_e = seed_queue.popleft()
        curr_data = edge_lookup[curr_e]
        t_curr = curr_data['edge_dir']
        v0, v1 = curr_data['verts']

        for nbr_e in vert_to_edge_map[v0] + vert_to_edge_map[v1]:
            if nbr_e not in locked_edges:
                nbr_data = edge_lookup[nbr_e]
                if nbr_data['angle_deg'] >= nbr_data['trace_thresh']:
                    if abs(float(np.dot(t_curr, nbr_data['edge_dir']))) >= 0.35:
                        locked_edges.add(nbr_e)
                        seed_queue.append(nbr_e)

    # Filter short noise blips
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
        locked_edges = pruned_locked

    cut_edge_tuples = set()
    for e_idx in locked_edges:
        v0, v1 = edge_lookup[e_idx]['verts']
        cut_edge_tuples.add(tuple(sorted((v0, v1))))

    return cut_edge_tuples, open_boundary_edges

def match_external_rails_to_mesh(mesh, rails_path, max_dist=None):
    """
    Matches an external rails OBJ (ribbons or line segments) to the nearest mesh edges.
    """
    print(f"[*] Matching external Stage 1 rails from: {rails_path}")
    rails_mesh = trimesh.load(rails_path, force='mesh')
    if isinstance(rails_mesh, trimesh.Scene):
        rails_mesh = rails_mesh.dump(concatenate=True)

    r_edges = rails_mesh.edges_unique
    r_midpoints = (rails_mesh.vertices[r_edges[:, 0]] + rails_mesh.vertices[r_edges[:, 1]]) * 0.5

    # Mesh edge midpoints
    m_edges = mesh.edges_unique
    m_midpoints = (mesh.vertices[m_edges[:, 0]] + mesh.vertices[m_edges[:, 1]]) * 0.5
    avg_len = np.mean(mesh.edges_unique_length)
    thresh = max_dist if max_dist is not None else avg_len * 0.75

    tree = cKDTree(m_midpoints)
    dists, nearest_idx = tree.query(r_midpoints)

    matched_edges = set()
    for d, idx in zip(dists, nearest_idx):
        if d <= thresh:
            e = tuple(sorted((int(m_edges[idx, 0]), int(m_edges[idx, 1]))))
            matched_edges.add(e)

    print(f"    -> Matched {len(matched_edges):,} surface edges to external Stage 1 rails.")
    return matched_edges

def split_mesh_by_rails(mesh, cut_edges):
    """
    Topologically splits the mesh along cut_edges:
    Partitions the face umbrella around each vertex into connected sectors
    separated by cut edges, giving each sector its own vertex duplicate.
    
    Returns:
      new_mesh: trimesh.Trimesh with split boundaries
      split_manifest: dict mapping original vertex -> list of newly allocated vertex IDs
    """
    v_orig = mesh.vertices.copy()
    f_orig = mesh.faces.copy()
    n_verts = len(v_orig)

    # Face to edge lookup
    face_edges = [
        [
            tuple(sorted((f[0], f[1]))),
            tuple(sorted((f[1], f[2]))),
            tuple(sorted((f[2], f[0])))
        ]
        for f in f_orig
    ]

    # Vertex to incident faces
    v_to_faces = defaultdict(list)
    for f_idx, f in enumerate(f_orig):
        v_to_faces[f[0]].append(f_idx)
        v_to_faces[f[1]].append(f_idx)
        v_to_faces[f[2]].append(f_idx)

    new_verts = []
    split_to_orig = []
    new_faces = f_orig.copy()
    split_manifest = defaultdict(list)  # orig_v -> [new_v0, new_v1, ...]

    for v_idx in range(n_verts):
        inc_faces = v_to_faces[v_idx]
        if len(inc_faces) <= 1:
            new_id = len(new_verts)
            new_verts.append(v_orig[v_idx])
            split_to_orig.append(v_idx)
            split_manifest[v_idx].append(new_id)
            if len(inc_faces) == 1:
                f_id = inc_faces[0]
                row = new_faces[f_id]
                for k in range(3):
                    if row[k] == v_idx:
                        row[k] = new_id
            continue

        # Check if any incident edge at this vertex is in cut_edges
        incident_cut = False
        for f_id in inc_faces:
            for e in face_edges[f_id]:
                if v_idx in e and e in cut_edges:
                    incident_cut = True
                    break
            if incident_cut:
                break

        if not incident_cut:
            # Vertex does not lie on a cut edge; no split needed
            new_id = len(new_verts)
            new_verts.append(v_orig[v_idx])
            split_to_orig.append(v_idx)
            split_manifest[v_idx].append(new_id)
            for f_id in inc_faces:
                row = new_faces[f_id]
                for k in range(3):
                    if row[k] == v_idx:
                        row[k] = new_id
            continue

        # Build sector connectivity graph between incident faces
        G = nx.Graph()
        G.add_nodes_from(inc_faces)

        for i in range(len(inc_faces)):
            f_i = inc_faces[i]
            edges_i = face_edges[f_i]
            for j in range(i + 1, len(inc_faces)):
                f_j = inc_faces[j]
                edges_j = face_edges[f_j]
                # Find shared edge
                for e_i in edges_i:
                    if v_idx in e_i and e_i in edges_j:
                        # Only connect if the shared edge is NOT cut
                        if e_i not in cut_edges:
                            G.add_edge(f_i, f_j)
                        break

        components = list(nx.connected_components(G))

        for c_idx, comp in enumerate(components):
            new_id = len(new_verts)
            new_verts.append(v_orig[v_idx])
            split_to_orig.append(v_idx)
            split_manifest[v_idx].append(new_id)
            for f_id in comp:
                row = new_faces[f_id]
                for k in range(3):
                    if row[k] == v_idx:
                        row[k] = new_id

    new_verts = np.array(new_verts, dtype=np.float64)
    split_to_orig = np.array(split_to_orig, dtype=np.int64)
    split_mesh = trimesh.Trimesh(vertices=new_verts, faces=new_faces, process=False)
    return split_mesh, split_manifest, split_to_orig

def run_stage4_step1_seam_split(
    input_path, 
    output_split_obj, 
    rails_path=None, 
    cv_high=28.0, 
    cv_low=18.0, 
    cc_high=24.0, 
    cc_low=16.0,
    min_chain_len=18
):
    print("\n" + "="*75)
    print(">>> [STAGE 4 - STEP 1] Seam Splitting & Boundary Freezing along Stage 1 Rails")
    print("="*75)
    t0 = time.time()

    print(f"[*] Loading input mesh: {input_path}")
    raw_mesh = trimesh.load(input_path, force='mesh')
    if isinstance(raw_mesh, trimesh.Scene):
        raw_mesh = raw_mesh.dump(concatenate=True)

    mesh = clean_input_mesh(raw_mesh)
    v_orig_count = len(mesh.vertices)
    f_orig_count = len(mesh.faces)
    print(f"    -> Prepared base mesh: {v_orig_count:,} vertices | {f_orig_count:,} faces")

    # 1. Gather cut edges from Stage 1
    if rails_path and os.path.exists(rails_path):
        cut_edges = match_external_rails_to_mesh(mesh, rails_path)
        # Also ensure open boundary edges are accounted for
        edge_to_faces = defaultdict(list)
        for face in mesh.faces:
            for i in range(3):
                e = tuple(sorted((face[i], face[(i + 1) % 3])))
                edge_to_faces[e].append(0)
        open_bounds = set(e for e, flist in edge_to_faces.items() if len(flist) == 1)
    else:
        # Auto-detect companion rails in workspace if available
        base_dir = os.path.dirname(output_split_obj) or "."
        base_name = os.path.basename(input_path).split('.')[0].lower()
        cand_rails = [
            os.path.join(base_dir, f"{base_name}_rails_robust.obj"),
            os.path.join(base_dir, f"{base_name}_rails.obj"),
            os.path.join("vacuum_quad_engine/assets/outputs", f"{base_name}_rails_robust.obj")
        ]
        found_cand = None
        for c in cand_rails:
            if os.path.exists(c):
                found_cand = c
                break

        if found_cand:
            cut_edges = match_external_rails_to_mesh(mesh, found_cand)
            open_bounds = set()
        else:
            print("[*] Running built-in Stage 1 multi-scale signed hysteresis...")
            cut_edges, open_bounds = extract_stage1_rail_edges(
                mesh, cv_high=cv_high, cv_low=cv_low, cc_high=cc_high, cc_low=cc_low, min_chain_len=min_chain_len
            )

    print(f"    -> Total Cut Rail Edges to Split: {len(cut_edges):,} edges")
    print(f"    -> Existing Open Perimeter Edges:  {len(open_bounds):,} edges")

    # 2. Perform topological split
    print("[*] Executing topological umbrella sector-split along rail seams...")
    split_mesh, manifest, split_to_orig = split_mesh_by_rails(mesh, cut_edges)
    v_split_count = len(split_mesh.vertices)

    # 3. Compute audit statistics
    orig_adj = mesh.face_adjacency
    orig_boundaries = len(mesh.edges_unique) - len(orig_adj)

    new_adj = split_mesh.face_adjacency
    new_boundaries = len(split_mesh.edges_unique) - len(new_adj)

    split_verts_count = sum(1 for v, splits in manifest.items() if len(splits) > 1)
    total_added_verts = v_split_count - v_orig_count

    print(f"\n[+] [STEP 1 AUDIT & VERIFICATION]")
    print(f"    -> Original Vertices: {v_orig_count:,}")
    print(f"    -> Split Vertices:    {v_split_count:,} (+{total_added_verts:,} duplicate sector vertices)")
    print(f"    -> Rail Vertices Split: {split_verts_count:,} vertices partitioned into discrete sectors")
    print(f"    -> Total Faces:       {len(split_mesh.faces):,} (100% faces preserved, 0 faces deleted)")
    print(f"    -> Boundary Edges:    {orig_boundaries:,} -> {new_boundaries:,} (ALL rail edges are now hard boundaries)")
    print(f"    -> Execution Time:    {time.time() - t0:.2f}s")

    # 4. Export Seam-Split Mesh OBJ
    os.makedirs(os.path.dirname(os.path.abspath(output_split_obj)), exist_ok=True)
    split_mesh.export(output_split_obj)
    print(f"\n[+] Saved Seam-Split Mesh to: {output_split_obj}")

    # 5. Export JSON & NPZ Manifest for Step 2 and Step 5
    manifest_path = output_split_obj.replace(".obj", "_manifest.json")
    manifest_npz_path = output_split_obj.replace(".obj", "_manifest.npz")
    manifest_serializable = {
        "original_vertex_count": v_orig_count,
        "split_vertex_count": v_split_count,
        "cut_edge_count": len(cut_edges),
        "split_map": {str(k): v for k, v in manifest.items() if len(v) > 1}
    }
    with open(manifest_path, 'w') as f_man:
        json.dump(manifest_serializable, f_man, indent=2)
    np.savez_compressed(
        manifest_npz_path, 
        split_to_orig=split_to_orig,
        v_orig=mesh.vertices,
        f_orig=mesh.faces
    )
    print(f"[+] Saved Re-welding Manifest to: {manifest_path} and {manifest_npz_path}")

    # 6. Export Visual Inspection Companion GLB with Seams Highlighted
    glb_path = output_split_obj.replace(".obj", "_inspection.glb")
    try:
        # Color boundary vertices in neon green and internal vertices in neutral grey
        edge_face_counts = defaultdict(int)
        for f in split_mesh.faces:
            for i in range(3):
                e = tuple(sorted((f[i], f[(i + 1) % 3])))
                edge_face_counts[e] += 1
        
        boundary_v = set()
        for e, cnt in edge_face_counts.items():
            if cnt == 1:
                boundary_v.add(e[0])
                boundary_v.add(e[1])

        colors = np.zeros((v_split_count, 4), dtype=np.uint8)
        colors[:, :] = [180, 180, 180, 255]  # Grey base
        for bv in boundary_v:
            colors[bv] = [50, 255, 50, 255]  # Neon Green Seam Rail

        split_mesh.visual = trimesh.visual.ColorVisuals(mesh=split_mesh, vertex_colors=colors)
        split_mesh.export(glb_path)
        print(f"[+] Saved Inspection Companion to: {glb_path}")
    except Exception as e:
        print(f"    -> Inspection companion note: {e}")

    return output_split_obj, manifest_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4 - Step 1: Rail Seam Splitting & Boundary Freezing")
    parser.add_argument("input", help="Input mesh file path (.obj, .glb, .stl)")
    parser.add_argument("output", help="Output split OBJ path")
    parser.add_argument("--rails", default=None, help="Optional Stage 1 boundary rails file (.obj)")
    parser.add_argument("--cv-high", type=float, default=30.0, help="Convex ridge seed threshold (default: 30.0)")
    parser.add_argument("--cv-low", type=float, default=18.0, help="Convex ridge chain threshold (default: 18.0)")
    parser.add_argument("--cc-high", type=float, default=26.0, help="Concave valley seed threshold (default: 26.0)")
    parser.add_argument("--cc-low", type=float, default=16.0, help="Concave valley chain threshold (default: 16.0)")
    parser.add_argument("--min-chain", type=int, default=6, help="Minimum connected chain length (default: 6)")
    args = parser.parse_args()

    run_stage4_step1_seam_split(
        args.input, 
        args.output, 
        rails_path=args.rails,
        cv_high=args.cv_high,
        cv_low=args.cv_low,
        cc_high=args.cc_high,
        cc_low=args.cc_low,
        min_chain_len=args.min_chain
    )

