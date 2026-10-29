#!/usr/bin/env python3
import os
import sys
import time
import argparse
from collections import defaultdict, deque
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh
import networkx as nx
from scipy.spatial import cKDTree

def clean_trimesh(mesh):
    mesh.merge_vertices()
    if hasattr(mesh, 'remove_duplicate_faces'):
        mesh.remove_duplicate_faces()
    elif hasattr(mesh, 'unique_faces'):
        mesh.update_faces(mesh.unique_faces())
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, 'nondegenerate_faces'):
        mesh.update_faces(mesh.nondegenerate_faces())
    return mesh

# ==============================================================================
# STAGE 1: FEATURE RAILS (CONVEX RIDGES & CONCAVE PART-JUNCTIONS)
# ==============================================================================
def run_stage_1(mesh, base_prefix, cv_high=28.0, cv_low=18.0, cc_high=24.0, cc_low=16.0, min_chain_len=18):
    print("\n" + "="*70)
    print(">>> [STAGE 1/5] Extracting 1D Boundary Rails (Structural Bones)...")
    print("="*70)
    t0 = time.time()

    v_pos = mesh.vertices
    adj_faces = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges
    face_normals = mesh.face_normals
    avg_edge_len = float(np.mean(mesh.edges_unique_length))

    raw_candidates = []
    vert_to_edge_map = defaultdict(list)

    for idx, ((f0, f1), (v0, v1)) in enumerate(zip(adj_faces, adj_edges)):
        edge_vec = v_pos[v1] - v_pos[v0]
        edge_len = float(np.linalg.norm(edge_vec))
        if edge_len < 1e-12:
            continue
        edge_dir = edge_vec / edge_len

        dot_n = np.clip(np.dot(face_normals[f0], face_normals[f1]), -1.0, 1.0)
        angle_deg = float(np.degrees(np.arccos(dot_n)))

        length_boost = float(np.clip(edge_len / (avg_edge_len + 1e-12), 1.0, 2.0))
        ad_cv_high = max(cv_low, cv_high / length_boost)
        ad_cv_low  = max(cv_low * 0.75, cv_low / length_boost)
        ad_cc_high = max(cc_low, cc_high / length_boost)
        ad_cc_low  = max(cc_low * 0.75, cc_low / length_boost)

        signed_sin = float(np.dot(np.cross(face_normals[f0], face_normals[f1]), edge_dir))
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

    edge_lookup = {c['edge_idx']: c for c in raw_candidates}
    locked_edges = set()
    seed_queue = deque()

    for c in raw_candidates:
        if c['angle_deg'] >= c['seed_thresh']:
            locked_edges.add(c['edge_idx'])
            seed_queue.append(c['edge_idx'])

    # Collinear Hysteresis Tracing
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

    # Topological Collinear Gap Bridging on Dead Ends
    def get_dead_ends(edges_set):
        deg = defaultdict(int)
        for e in edges_set:
            u, v = edge_lookup[e]['verts']
            deg[u] += 1
            deg[v] += 1
        return [v for v, d in deg.items() if d == 1]

    for pass_idx in range(2):
        dead_ends = get_dead_ends(locked_edges)
        v_to_locked = defaultdict(list)
        for e in locked_edges:
            u, v = edge_lookup[e]['verts']
            v_to_locked[u].append(e)
            v_to_locked[v].append(e)

        for vA in dead_ends:
            inc = v_to_locked[vA]
            if not inc: continue
            tA = edge_lookup[inc[0]]['edge_dir']
            for nbr_e in vert_to_edge_map[vA]:
                if nbr_e not in locked_edges:
                    nbr_data = edge_lookup[nbr_e]
                    if nbr_data['angle_deg'] >= 10.0:
                        if abs(float(np.dot(tA, nbr_data['edge_dir']))) >= 0.45:
                            locked_edges.add(nbr_e)

    # Prune micro-noise (< min_chain_len edges)
    if min_chain_len > 1 and len(locked_edges) > 0:
        rail_graph = nx.Graph()
        for e_idx in locked_edges:
            v0, v1 = edge_lookup[e_idx]['verts']
            rail_graph.add_edge(int(v0), int(v1), edge_idx=e_idx)

        pruned = set()
        for comp in nx.connected_components(rail_graph):
            sub = rail_graph.subgraph(comp)
            if sub.number_of_edges() >= min_chain_len:
                for u, v, d in sub.edges(data=True):
                    pruned.add(d['edge_idx'])
        locked_edges = pruned

    # Export clean 3D solid ribbon rails
    ribbon_verts = []
    ribbon_faces = []
    w = avg_edge_len * 0.35

    for idx in locked_edges:
        data = edge_lookup[idx]
        p0, p1 = v_pos[data['verts'][0]], v_pos[data['verts'][1]]
        e_dir = data['edge_dir']
        avg_n = face_normals[data['f0']] + face_normals[data['f1']]
        norm_val = np.linalg.norm(avg_n)
        if norm_val < 1e-12: continue
        avg_n /= norm_val

        offset = avg_n * (w * 0.2)
        lateral = np.cross(avg_n, e_dir)
        lat_norm = np.linalg.norm(lateral)
        if lat_norm < 1e-12: continue
        lateral = (lateral / lat_norm) * (w * 0.5)

        b = len(ribbon_verts)
        ribbon_verts.extend([p0 + offset - lateral, p1 + offset - lateral,
                             p1 + offset + lateral, p0 + offset + lateral])
        ribbon_faces.append([b, b + 1, b + 2])
        ribbon_faces.append([b, b + 2, b + 3])

    rails_obj_path = f"{base_prefix}_stage1_rails.obj"
    trimesh.Trimesh(vertices=ribbon_verts, faces=ribbon_faces, process=False).export(rails_obj_path)
    print(f"[*] Stage 1 Complete in {time.time()-t0:.2f}s -> Saved: {rails_obj_path} ({len(locked_edges):,} rails)")
    return locked_edges


# ==============================================================================
# STAGE 2: RIEMANNIAN ANISOTROPIC METRIC TENSOR
# ==============================================================================
def run_stage_2(mesh, base_prefix, target_quads=4000, max_aspect=6.0, passes=8):
    print("\n" + "="*70)
    print(f">>> [STAGE 2/5] Computing Riemannian Metric Tensor (Budget: ~{target_quads:,} quads)...")
    print("="*70)
    t0 = time.time()

    v_pos = mesh.vertices
    faces = mesh.faces
    n_faces = len(faces)
    face_normals = mesh.face_normals
    face_centers = mesh.triangles_center
    adj_faces = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges

    total_area = float(mesh.area)
    h_nominal = np.sqrt(total_area / float(target_quads))
    h_min = h_nominal * 0.20
    h_max = h_nominal * max_aspect

    # Orthonormal tangent basis (t1, t2)
    arb = np.zeros_like(face_normals)
    arb[:, 0] = 1.0
    mask = np.abs(face_normals[:, 0]) > 0.9
    arb[mask, 0] = 0.0
    arb[mask, 1] = 1.0
    t1 = np.cross(face_normals, arb)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    t2 = np.cross(face_normals, t1)

    f0 = adj_faces[:, 0]
    f1 = adj_faces[:, 1]
    edge_vec = v_pos[adj_edges[:, 1]] - v_pos[adj_edges[:, 0]]
    edge_len = np.linalg.norm(edge_vec, axis=1)
    valid = edge_len > 1e-12

    f0 = f0[valid]
    f1 = f1[valid]
    edge_len = edge_len[valid]

    d_center = np.linalg.norm(face_centers[f1] - face_centers[f0], axis=1) + 1e-12
    d_cent_dir = (face_centers[f1] - face_centers[f0]) / d_center[:, None]
    dn = face_normals[f1] - face_normals[f0]
    dn_de = dn / d_center[:, None]

    II_11 = np.zeros(n_faces)
    II_12 = np.zeros(n_faces)
    II_22 = np.zeros(n_faces)
    weights = np.zeros(n_faces)

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

    if passes > 0 and len(adj_faces) > 0:
        row = np.hstack([f0, f1])
        col = np.hstack([f1, f0])
        A = sp.csr_matrix((np.ones(len(row)), (row, col)), shape=(n_faces, n_faces))
        deg = np.array(A.sum(axis=1)).flatten()
        deg[deg == 0] = 1.0
        M_smooth = sp.diags(1.0 / deg) @ A

        for _ in range(passes):
            II_11 = 0.4 * II_11 + 0.6 * (M_smooth @ II_11)
            II_12 = 0.4 * II_12 + 0.6 * (M_smooth @ II_12)
            II_22 = 0.4 * II_22 + 0.6 * (M_smooth @ II_22)

    tr = II_11 + II_22
    det = II_11 * II_22 - II_12**2
    disc = np.sqrt(np.maximum(0.0, (tr * 0.5)**2 - det))
    k1, k2 = tr * 0.5 + disc, tr * 0.5 - disc
    abs_k1, abs_k2 = np.abs(k1), np.abs(k2)

    swap = abs_k2 > abs_k1
    abs_k_max = np.where(swap, abs_k2, abs_k1)
    abs_k_min = np.where(swap, abs_k1, abs_k2)

    theta = 0.5 * np.arctan2(2.0 * II_12, II_11 - II_22)
    theta = np.where(swap, theta + np.pi * 0.5, theta)
    e1 = np.cos(theta)[:, None] * t1 + np.sin(theta)[:, None] * t2
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True) + 1e-12
    e2 = np.cross(face_normals, e1)
    e2 /= np.linalg.norm(e2, axis=1, keepdims=True) + 1e-12

    p80 = np.percentile(abs_k_max, 80)
    c_scale = 1.0 / (np.sqrt(p80) + 1e-6)
    h1 = np.clip((c_scale / (np.sqrt(abs_k_max) + 1e-6)) * h_nominal, h_min, h_nominal)

    curv_asymm = (abs_k_max - abs_k_min) / (abs_k_max + abs_k_min + 1e-5)
    aspect_field = 1.0 + (max_aspect - 1.0) * (curv_asymm ** 1.5)
    planar = abs_k_max < np.percentile(abs_k_max, 25)
    aspect_field[planar] = np.maximum(aspect_field[planar], max_aspect * 0.75)
    h2 = np.clip(h1 * aspect_field, h1, h_max)

    tensor_path = f"{base_prefix}_stage2_tensor.npy"
    np.save(tensor_path, np.hstack([e1, e2, h1[:, None], h2[:, None]]))

    # Heatmap preview OBJ
    norm_aspect = np.clip((aspect_field - 1.0) / (max_aspect - 1.0 + 1e-12), 0.0, 1.0)
    vert_norm = np.zeros(len(v_pos))
    counts = np.zeros(len(v_pos))
    np.add.at(vert_norm, faces.reshape(-1), np.repeat(norm_aspect, 3))
    np.add.at(counts, faces.reshape(-1), 1.0)
    counts[counts == 0] = 1.0
    vert_norm /= counts

    v_colors = np.zeros((len(v_pos), 4), dtype=np.uint8)
    v_colors[:, 3] = 255
    v_colors[:, 0] = (np.clip(vert_norm * 1.5, 0.0, 1.0) * 255).astype(np.uint8)
    v_colors[:, 1] = (np.sin(vert_norm * np.pi) * 230).astype(np.uint8)
    v_colors[:, 2] = (np.clip(1.0 - vert_norm * 1.5, 0.0, 1.0) * 255).astype(np.uint8)

    mesh_viz = mesh.copy()
    mesh_viz.visual = trimesh.visual.ColorVisuals(mesh=mesh_viz, vertex_colors=v_colors)
    mesh_viz.export(f"{base_prefix}_stage2_heatmap.obj")

    print(f"[*] Stage 2 Complete in {time.time()-t0:.2f}s -> Saved: {tensor_path}")
    return tensor_path


# ==============================================================================
# STAGE 3: ANISOTROPIC CROSS-FIELD SOLVE (CONNECTION LAPLACIAN)
# ==============================================================================
def run_stage_3(mesh, tensor_path, base_prefix, locked_edges):
    print("\n" + "="*70)
    print(">>> [STAGE 3/5] Solving Feature-Locked Anisotropic Cross-Field...")
    print("="*70)
    t0 = time.time()

    tensor_data = np.load(tensor_path)
    e2_in = tensor_data[:, 3:6]
    h1_in, h2_in = tensor_data[:, 6], tensor_data[:, 7]
    aspect_ratios = h2_in / (h1_in + 1e-12)

    v_pos = mesh.vertices
    n_faces = len(mesh.faces)
    face_normals = mesh.face_normals
    face_centers = mesh.triangles_center
    adj_faces = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges

    arb = np.zeros_like(face_normals)
    arb[:, 0] = 1.0
    arb[np.abs(face_normals[:, 0]) > 0.9, 0] = 0.0
    arb[np.abs(face_normals[:, 0]) > 0.9, 1] = 1.0
    t1 = np.cross(face_normals, arb)
    t1 /= np.linalg.norm(t1, axis=1, keepdims=True) + 1e-12
    t2 = np.cross(face_normals, t1)

    u_complex = np.zeros(n_faces, dtype=np.complex128)
    fixed_mask = np.zeros(n_faces, dtype=bool)
    weights = np.zeros(n_faces, dtype=np.float64)

    # Lock Stage 1 Feature Rails
    for e_idx in locked_edges:
        (f0, f1) = adj_faces[e_idx]
        (v0, v1) = adj_edges[e_idx]
        edge_vec = v_pos[v1] - v_pos[v0]
        e_dir = edge_vec / (np.linalg.norm(edge_vec) + 1e-12)
        for f_idx in (f0, f1):
            theta = np.arctan2(np.dot(e_dir, t2[f_idx]), np.dot(e_dir, t1[f_idx]))
            u_complex[f_idx] += np.exp(1j * 4.0 * theta)
            fixed_mask[f_idx] = True
            weights[f_idx] = 1000.0

    # Lock Anisotropic Stretch Directions
    for f_idx in np.where((aspect_ratios >= 2.5) & (~fixed_mask))[0]:
        theta = np.arctan2(np.dot(e2_in[f_idx], t2[f_idx]), np.dot(e2_in[f_idx], t1[f_idx]))
        u_complex[f_idx] = np.exp(1j * 4.0 * theta)
        fixed_mask[f_idx] = True
        weights[f_idx] = 100.0

    if np.any(fixed_mask):
        u_complex[fixed_mask] /= np.abs(u_complex[fixed_mask]) + 1e-12
    else:
        u_complex[0], fixed_mask[0], weights[0] = 1.0 + 0j, True, 100.0

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

    ed_dot_t1_0 = np.sum(edge_dir * t1[f0], axis=1)
    ed_dot_t2_0 = np.sum(edge_dir * t2[f0], axis=1)
    ed_dot_t1_1 = np.sum(edge_dir * t1[f1], axis=1)
    ed_dot_t2_1 = np.sum(edge_dir * t2[f1], axis=1)

    alpha0 = np.arctan2(ed_dot_t2_0, ed_dot_t1_0)
    alpha1 = np.arctan2(ed_dot_t2_1, ed_dot_t1_1)
    transport = np.exp(1j * 4.0 * (alpha1 - alpha0))

    d_c = np.linalg.norm(face_centers[f0] - face_centers[f1], axis=1) + 1e-12
    w = (edge_len / d_c) * np.sqrt(0.5 * (aspect_ratios[f0] + aspect_ratios[f1]))

    rows = np.concatenate([f0, f1])
    cols = np.concatenate([f1, f0])
    vals = np.concatenate([w * np.conj(transport), w * transport])

    W = sp.csr_matrix((vals, (rows, cols)), shape=(n_faces, n_faces), dtype=np.complex128)
    L = sp.diags(np.array(np.abs(W).sum(axis=1)).flatten(), dtype=np.complex128) - W
    P = sp.diags(weights, dtype=np.complex128)
    A = L + P + sp.eye(n_faces, dtype=np.complex128) * 1e-6
    rhs = weights * u_complex

    if n_faces <= 30000:
        u_sol = spla.spsolve(A.tocsc(), rhs)
    else:
        diag_A = A.diagonal()
        diag_A[np.abs(diag_A) < 1e-12] = 1.0
        M_prec = sp.diags(1.0 / diag_A)
        u_sol, info = spla.bicgstab(A, rhs, M=M_prec, maxiter=300, rtol=1e-5)
        if info != 0:
            u_sol = spla.spsolve(A.tocsc(), rhs)

    angles = np.angle(u_sol) / 4.0
    u_dir = np.cos(angles)[:, None] * t1 + np.sin(angles)[:, None] * t2
    u_dir /= np.linalg.norm(u_dir, axis=1, keepdims=True) + 1e-12
    v_dir = np.cross(face_normals, u_dir)
    v_dir /= np.linalg.norm(v_dir, axis=1, keepdims=True) + 1e-12

    field_path = f"{base_prefix}_stage3_field.npy"
    np.save(field_path, np.stack([u_dir, v_dir], axis=1))
    print(f"[*] Stage 3 Complete in {time.time()-t0:.2f}s -> Saved: {field_path}")
    return field_path


# ==============================================================================
# STAGE 4: ANISOTROPIC QUAD EXTRACTION VIA BLOSSOM MATCHING
# ==============================================================================
def run_stage_4(mesh, tensor_path, field_path, base_prefix, target_faces=4000, crease_deg=28.0):
    print("\n" + "="*70)
    print(f">>> [STAGE 4/5] Anisotropic Quad Extraction (Budget ~{target_faces:,} quads)...")
    print("="*70)
    t0 = time.time()

    target_tris = max(1200, target_faces * 2)
    split_bodies = mesh.split(only_watertight=False)
    simplified_bodies = []
    total_raw_faces = max(1, len(mesh.faces))

    for body in split_bodies:
        body_face_ratio = len(body.faces) / total_raw_faces
        body_target = max(60, int(target_tris * body_face_ratio))
        if len(body.faces) > body_target:
            simp = body.simplify_quadric_decimation(face_count=body_target)
            simp = clean_trimesh(simp)
            simplified_bodies.append(simp)
        else:
            simplified_bodies.append(body)

    carrier = trimesh.util.concatenate(simplified_bodies)
    carrier = clean_trimesh(carrier)

    cv_pos = carrier.vertices.copy()
    c_faces = carrier.faces
    c_normals = carrier.face_normals
    c_centers = carrier.triangles_center
    n_c_faces = len(c_faces)

    raw_kdtree = cKDTree(mesh.triangles_center)
    _, nearest = raw_kdtree.query(c_centers, k=1)

    tensor_data = np.load(tensor_path)
    carrier_h1 = tensor_data[nearest, 6]
    carrier_h2 = tensor_data[nearest, 7]
    carrier_aspect = carrier_h2 / (carrier_h1 + 1e-12)

    field_data = np.load(field_path)
    carrier_u = field_data[nearest, 0, :]
    carrier_v = field_data[nearest, 1, :]

    adj = carrier.face_adjacency
    adj_edges = carrier.face_adjacency_edges
    unshared = carrier.face_adjacency_unshared

    # Pass 1: Strict orthogonal quads aligned with streamlines (skew <= 0.80)
    G1 = nx.Graph()
    for (f0, f1), (e0, e1), (u0, u1) in zip(adj, adj_edges, unshared):
        dot_n = np.dot(c_normals[f0], c_normals[f1])
        if dot_n < 0.5: continue
        q_idx = [u0, e0, u1, e1]
        pts = cv_pos[q_idx]
        edges = np.roll(pts, -1, axis=0) - pts
        lens = np.linalg.norm(edges, axis=1) + 1e-12
        dirs = edges / lens[:, None]
        cos_corners = [-np.dot(dirs[0], dirs[1]), -np.dot(dirs[1], dirs[2]),
                       -np.dot(dirs[2], dirs[3]), -np.dot(dirs[3], dirs[0])]
        skew = np.max(np.abs(cos_corners))
        if skew > 0.80: continue

        u_ref, v_ref = carrier_u[f0], carrier_v[f0]
        field_align = float(np.mean([max(abs(np.dot(d, u_ref)), abs(np.dot(d, v_ref))) for d in dirs]))

        current_ratio = max(lens[0], lens[1]) / (min(lens[0], lens[1]) + 1e-12)
        target_ratio = min(5.0, carrier_aspect[f0])
        aspect_score = 1.0 - min(1.0, abs(current_ratio - target_ratio) / target_ratio)

        weight = (1.0 - skew) * 3.0 + field_align * 3.5 + aspect_score * 2.0 + dot_n
        G1.add_edge(int(f0), int(f1), weight=float(weight), quad=q_idx)

    matching1 = nx.max_weight_matching(G1, maxcardinality=False)
    paired = set()
    quad_faces = []

    for f0, f1 in matching1:
        q = G1.get_edge_data(f0, f1)['quad']
        n_est = np.cross(cv_pos[q[2]] - cv_pos[q[0]], cv_pos[q[3]] - cv_pos[q[1]])
        if np.dot(n_est, c_normals[f0] + c_normals[f1]) < 0:
            q = [q[0], q[3], q[2], q[1]]
        quad_faces.append(q)
        paired.add(f0); paired.add(f1)

    # Pass 2: Secondary coplanar relaxation (skew <= 0.90) for remaining flat triangles
    G2 = nx.Graph()
    for (f0, f1), (e0, e1), (u0, u1) in zip(adj, adj_edges, unshared):
        if f0 in paired or f1 in paired: continue
        dot_n = np.dot(c_normals[f0], c_normals[f1])
        if dot_n < 0.65: continue
        q_idx = [u0, e0, u1, e1]
        pts = cv_pos[q_idx]
        edges = np.roll(pts, -1, axis=0) - pts
        lens = np.linalg.norm(edges, axis=1) + 1e-12
        dirs = edges / lens[:, None]
        cos_corners = [-np.dot(dirs[0], dirs[1]), -np.dot(dirs[1], dirs[2]),
                       -np.dot(dirs[2], dirs[3]), -np.dot(dirs[3], dirs[0])]
        skew = np.max(np.abs(cos_corners))
        if skew > 0.90: continue

        weight = (1.0 - skew) * 2.0 + dot_n
        G2.add_edge(int(f0), int(f1), weight=float(weight), quad=q_idx)

    matching2 = nx.max_weight_matching(G2, maxcardinality=False)
    for f0, f1 in matching2:
        q = G2.get_edge_data(f0, f1)['quad']
        n_est = np.cross(cv_pos[q[2]] - cv_pos[q[0]], cv_pos[q[3]] - cv_pos[q[1]])
        if np.dot(n_est, c_normals[f0] + c_normals[f1]) < 0:
            q = [q[0], q[3], q[2], q[1]]
        quad_faces.append(q)
        paired.add(f0); paired.add(f1)

    rem_tri = [list(c_faces[i]) for i in range(n_c_faces) if i not in paired]
    total_polys = len(quad_faces) + len(rem_tri)
    quad_pct = (len(quad_faces) / max(1, total_polys)) * 100.0

    print(f"    -> Extracted Polygons: {total_polys:,} | Quads: {len(quad_faces):,} ({quad_pct:.1f}%) | Tris: {len(rem_tri):,}")

    stage4_obj = f"{base_prefix}_stage4_quads.obj"
    with open(stage4_obj, 'w') as f_out:
        f_out.write("# Stage 4 Quad-Dominant Extraction\n")
        for v in cv_pos:
            f_out.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for q in quad_faces:
            f_out.write(f"f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n")
        for t in rem_tri:
            f_out.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")

    print(f"[*] Stage 4 Complete in {time.time()-t0:.2f}s -> Saved: {stage4_obj}")
    return stage4_obj


# ==============================================================================
# STAGE 5: SURFACE SHRINKWRAP & PRODUCTION SMOOTH FINISH
# ==============================================================================
def run_stage_5(raw_mesh, stage4_obj, base_prefix, iterations=10):
    print("\n" + "="*70)
    print(">>> [STAGE 5/5] Surface Shrinkwrap & Production Smooth Finish...")
    print("="*70)
    t0 = time.time()

    verts, quad_faces, tri_faces = [], [], []
    with open(stage4_obj, 'r') as f:
        for line in f:
            if line.startswith('v '):
                p = line.strip().split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith('f '):
                p = line.strip().split()
                idx = [int(x.split('/')[0]) - 1 for x in p[1:]]
                if len(idx) == 4:
                    quad_faces.append(idx)
                elif len(idx) == 3:
                    tri_faces.append(idx)

    v_pos = np.array(verts, dtype=np.float64)
    n_verts = len(v_pos)

    # Build reference KD-Tree from dense original mesh
    pts_surface, _ = trimesh.sample.sample_surface(raw_mesh, count=min(200000, max(50000, len(raw_mesh.faces))))
    all_ref = np.vstack([pts_surface, raw_mesh.vertices])
    kdtree = cKDTree(all_ref)

    # Topology neighborhoods for tangential smoothing
    v_adj = [set() for _ in range(n_verts)]
    for q in quad_faces:
        for i in range(4):
            v_adj[q[i]].add(q[(i + 1) % 4])
            v_adj[q[(i + 1) % 4]].add(q[i])
    for t in tri_faces:
        for i in range(3):
            v_adj[t[i]].add(t[(i + 1) % 3])
            v_adj[t[(i + 1) % 3]].add(t[i])

    poly_tris = list(tri_faces)
    for q in quad_faces:
        poly_tris.append([q[0], q[1], q[2]])
        poly_tris.append([q[0], q[2], q[3]])

    # Iterative tangential relaxation + ground truth projection
    for it in range(iterations):
        temp_mesh = trimesh.Trimesh(vertices=v_pos, faces=poly_tris, process=False)
        curr_vn = temp_mesh.vertex_normals
        v_next = v_pos.copy()

        for i in range(n_verts):
            nbrs = list(v_adj[i])
            if not nbrs: continue
            delta = np.mean(v_pos[nbrs], axis=0) - v_pos[i]
            norm = curr_vn[i]
            disp = delta - np.dot(delta, norm) * norm
            v_next[i] = v_pos[i] + 0.35 * disp

        _, idxs = kdtree.query(v_next, k=1)
        v_pos = 0.30 * v_next + 0.70 * all_ref[idxs]

    # Recompute area-weighted smooth vertex normals
    final_mesh = trimesh.Trimesh(vertices=v_pos, faces=poly_tris, process=False)
    smooth_vn = final_mesh.vertex_normals

    final_obj = f"{base_prefix}_FINAL_RETOPO.obj"
    with open(final_obj, 'w') as out:
        out.write("# Production Quad-Dominant Asset (Smooth Shaded)\n")
        for v in v_pos:
            out.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in smooth_vn:
            out.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        out.write("s 1\n")
        for q in quad_faces:
            out.write(f"f {q[0]+1}//{q[0]+1} {q[1]+1}//{q[1]+1} {q[2]+1}//{q[2]+1} {q[3]+1}//{q[3]+1}\n")
        for t in tri_faces:
            out.write(f"f {t[0]+1}//{t[0]+1} {t[1]+1}//{t[1]+1} {t[2]+1}//{t[2]+1}\n")

    # Export companion GLB
    glb_path = f"{base_prefix}_FINAL_RETOPO.glb"
    try:
        final_mesh.export(glb_path)
        print(f"[*] Exported companion smooth GLB: {glb_path}")
    except Exception as e:
        print(f"    -> GLB export note: {e}")

    print(f"[*] Stage 5 Complete in {time.time()-t0:.2f}s -> Saved: {final_obj}")
    return final_obj


# ==============================================================================
# PIPELINE ENTRY POINT
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Master Automated Anisotropic Retopology Pipeline")
    parser.add_argument("input", help="Raw input asset (.glb, .obj, .stl)")
    parser.add_argument("--faces", type=int, default=4000, help="Target quad face count budget (default: 4000)")
    parser.add_argument("--aspect", type=float, default=6.0, help="Maximum anisotropic aspect ratio (default: 6.0)")
    parser.add_argument("--crease", type=float, default=28.0, help="Sharp crease cutoff in degrees (default: 28.0)")
    args = parser.parse_args()

    total_start = time.time()
    
    # Standardize outputs directory
    output_dir = "vacuum_quad_engine/assets/outputs"
    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(args.input))[0]
    base_prefix = os.path.join(output_dir, base_name)

    print("="*70)
    print("MASTER ANISOTROPIC RETOPOLOGY PIPELINE STARTED")
    print(f"Asset:  {args.input}")
    print(f"Target: {base_prefix}")
    print(f"Budget: ~{args.faces:,} quads | Max Aspect: {args.aspect}:1")
    print("="*70)

    # 1. Load & Coordinate-Weld Raw Input
    mesh = trimesh.load(args.input, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    
    # Weld coordinate duplicates to eliminate UV seams
    _, u_idx, inv = np.unique(np.round(mesh.vertices, 6), axis=0, return_index=True, return_inverse=True)
    mesh = trimesh.Trimesh(vertices=mesh.vertices[u_idx], faces=inv[mesh.faces], process=True)
    mesh = clean_trimesh(mesh)

    # Sequential Pipeline Execution (Stages 1 through 5)
    locked_edges = run_stage_1(mesh, base_prefix)
    tensor_path = run_stage_2(mesh, base_prefix, target_quads=args.faces, max_aspect=args.aspect)
    field_path = run_stage_3(mesh, tensor_path, base_prefix, locked_edges)
    stage4_obj = run_stage_4(mesh, tensor_path, field_path, base_prefix, target_faces=args.faces, crease_deg=args.crease)
    final_mesh = run_stage_5(mesh, stage4_obj, base_prefix, iterations=10)

    print("\n" + "="*70)
    print("PIPELINE EXECUTION FINISHED SUCCESSFULLY")
    print(f"Total Processing Time: {time.time()-total_start:.2f}s")
    print(f"Final Model Exported:  {final_mesh}")
    print("="*70)

if __name__ == "__main__":
    main()
