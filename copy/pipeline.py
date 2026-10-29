#!/usr/bin/env python3
import os
import sys
import time
import argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import trimesh
import networkx as nx
from scipy.spatial import cKDTree

# ==============================================================================
# STAGE 1: FEATURE RAILS (CONVEX RIDGES & CONCAVE PART-JUNCTIONS)
# ==============================================================================
def run_stage_1(mesh, base_prefix, cv_high=35.0, cv_low=22.0, cc_high=26.0, cc_low=16.0):
    print("\n" + "="*70)
    print(">>> [STAGE 1/5] Extracting 1D Boundary Rails (Structural Bones)...")
    print("="*70)
    t0 = time.time()

    v_pos = mesh.vertices
    adj_faces = mesh.face_adjacency
    adj_edges = mesh.face_adjacency_edges
    face_normals = mesh.face_normals

    avg_edge_len = np.mean(mesh.edges_unique_length)
    cos_cv_low = np.cos(np.radians(cv_low))
    cos_cc_low = np.cos(np.radians(cc_low))

    locked_convex_edges = set()
    locked_concave_edges = set()

    for idx, ((f0, f1), (v0, v1)) in enumerate(zip(adj_faces, adj_edges)):
        edge_vec = v_pos[v1] - v_pos[v0]
        edge_len = np.linalg.norm(edge_vec)
        if edge_len < 1e-12:
            continue
        edge_dir = edge_vec / edge_len

        dot_n = np.clip(np.dot(face_normals[f0], face_normals[f1]), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(dot_n))

        # Dynamic sensitivity for elongated hard-surface edges
        length_mult = min(2.5, max(1.0, edge_len / (avg_edge_len + 1e-12)))
        ad_cv_high = cv_high / length_mult
        ad_cc_high = cc_high / length_mult

        signed_sin = np.dot(np.cross(face_normals[f0], face_normals[f1]), edge_dir)
        is_convex = signed_sin >= 0.0

        if is_convex and angle_deg >= ad_cv_high:
            locked_convex_edges.add(idx)
        elif (not is_convex) and angle_deg >= ad_cc_high:
            locked_concave_edges.add(idx)

    # Export clean 3D ribbon rails (Zero phantom webs)
    ribbon_verts = []
    ribbon_faces = []
    w = avg_edge_len * 0.25

    all_locked = list(locked_convex_edges | locked_concave_edges)
    for idx in all_locked:
        (f0, f1) = adj_faces[idx]
        (v0, v1) = adj_edges[idx]
        p0, p1 = v_pos[v0], v_pos[v1]
        e_dir = (p1 - p0) / (np.linalg.norm(p1 - p0) + 1e-12)
        avg_n = face_normals[f0] + face_normals[f1]
        avg_n /= (np.linalg.norm(avg_n) + 1e-12)

        offset = avg_n * (w * 0.2)
        lateral = np.cross(avg_n, e_dir) * (w * 0.5)

        b = len(ribbon_verts)
        ribbon_verts.extend([p0 + offset - lateral, p1 + offset - lateral,
                             p1 + offset + lateral, p0 + offset + lateral])
        ribbon_faces.append([b, b + 1, b + 2])
        ribbon_faces.append([b, b + 2, b + 3])

    rails_obj_path = f"{base_prefix}_stage1_rails.obj"
    trimesh.Trimesh(vertices=ribbon_verts, faces=ribbon_faces, process=False).export(rails_obj_path)
    print(f"[*] Stage 1 Complete in {time.time()-t0:.2f}s -> Saved: {rails_obj_path}")
    return all_locked


# ==============================================================================
# STAGE 2: RIEMANNIAN ANISOTROPIC METRIC TENSOR
# ==============================================================================
def run_stage_2(mesh, base_prefix, target_quads=4500, max_aspect=6.0, passes=8):
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

    # Multi-Ring Weingarten Operator with cross-edge directional displacement
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

    # Filter out micro-faceting triangle noise via sparse normalized Laplacian
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

    # Principal frame decomposition
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

    # Directional sizing
    p80 = np.percentile(abs_k_max, 80)
    c_scale = 1.0 / (np.sqrt(p80) + 1e-6)
    h1 = np.clip((c_scale / (np.sqrt(abs_k_max) + 1e-6)) * h_nominal, h_min, h_nominal)

    curv_asymm = (abs_k_max - abs_k_min) / (abs_k_max + abs_k_min + 1e-5)
    aspect_field = 1.0 + (max_aspect - 1.0) * (curv_asymm ** 1.5)
    planar = abs_k_max < np.percentile(abs_k_max, 25)
    aspect_field[planar] = np.maximum(aspect_field[planar], max_aspect * 0.75)
    h2 = np.clip(h1 * aspect_field, h1, h_max)

    # Save Tensor NPY
    tensor_path = f"{base_prefix}_stage2_tensor.npy"
    np.save(tensor_path, np.hstack([e1, e2, h1[:, None], h2[:, None]]))

    # Heatmap preview
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

    # Lock Strong Anisotropic Stretch Directions
    for f_idx in np.where((aspect_ratios >= 2.5) & (~fixed_mask))[0]:
        theta = np.arctan2(np.dot(e2_in[f_idx], t2[f_idx]), np.dot(e2_in[f_idx], t1[f_idx]))
        u_complex[f_idx] = np.exp(1j * 4.0 * theta)
        fixed_mask[f_idx] = True
        weights[f_idx] = 100.0

    if np.any(fixed_mask):
        u_complex[fixed_mask] /= np.abs(u_complex[fixed_mask]) + 1e-12
    else:
        u_complex[0], fixed_mask[0], weights[0] = 1.0 + 0j, True, 100.0

    # Vectorized Connection Laplacian Assembly
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
    print(f">>> [STAGE 4/5] Anisotropic Quad-Dominant Extraction (Carrier ~{target_faces:,} faces)...")
    print("="*70)
    t0 = time.time()

    carrier_target = target_faces * 2
    if len(mesh.faces) > carrier_target:
        try:
            carrier = mesh.simplify_quadric_decimation(face_count=carrier_target)
        except Exception:
            ratio = float(carrier_target) / float(len(mesh.faces))
            carrier = mesh.simplify_quadric_decimation(percent=min(0.99, max(0.01, ratio)))
    else:
        carrier = mesh.copy()

    carrier.merge_vertices()
    if hasattr(carrier, 'remove_duplicate_faces'): carrier.remove_duplicate_faces()
    if hasattr(carrier, 'remove_degenerate_faces'): carrier.remove_degenerate_faces()
    if hasattr(carrier, 'nondegenerate_faces'): carrier.update_faces(carrier.nondegenerate_faces())

    cv_pos = carrier.vertices
    c_faces = carrier.faces
    c_normals = carrier.face_normals

    raw_kdtree = cKDTree(mesh.triangles_center)
    _, nearest = raw_kdtree.query(carrier.triangles_center, k=1)

    tensor_data = np.load(tensor_path)
    carrier_aspect = (tensor_data[nearest, 7] / (tensor_data[nearest, 6] + 1e-12))

    field_data = np.load(field_path)
    carrier_u = field_data[nearest, 0, :]
    carrier_v = field_data[nearest, 1, :]

    cos_crease = np.cos(np.radians(crease_deg))
    adj_faces = carrier.face_adjacency
    adj_edges = carrier.face_adjacency_edges
    unshared_v = carrier.face_adjacency_unshared

    G = nx.Graph()
    for (f0, f1), (e0, e1), (u0, u1) in zip(adj_faces, adj_edges, unshared_v):
        dot_n = np.dot(c_normals[f0], c_normals[f1])
        if dot_n < cos_crease: continue

        q_idx = [u0, e0, u1, e1]
        pts = cv_pos[q_idx]
        edges = np.roll(pts, -1, axis=0) - pts
        lens = np.linalg.norm(edges, axis=1) + 1e-12
        dirs = edges / lens[:, None]

        cos_corners = [-np.dot(dirs[0], dirs[1]), -np.dot(dirs[1], dirs[2]),
                       -np.dot(dirs[2], dirs[3]), -np.dot(dirs[3], dirs[0])]
        max_skew = np.max(np.abs(cos_corners))
        if max_skew > 0.85: continue

        u_ref, v_ref = carrier_u[f0], carrier_v[f0]
        field_align = float(np.mean([max(abs(np.dot(d, u_ref)), abs(np.dot(d, v_ref))) for d in dirs]))

        current_aspect = max(lens[0], lens[1]) / (min(lens[0], lens[1]) + 1e-12)
        target_asp = min(5.0, carrier_aspect[f0])
        aspect_score = 1.0 - min(1.0, abs(current_aspect - target_asp) / target_asp)

        weight = (1.0 - max_skew) * 3.0 + field_align * 3.0 + aspect_score * 2.0 + dot_n
        G.add_edge(int(f0), int(f1), weight=float(weight), quad=q_idx)

    # Component-wise Blossom matching
    matching = set()
    for comp in nx.connected_components(G):
        if len(comp) > 1:
            matching.update(nx.max_weight_matching(G.subgraph(comp), maxcardinality=False))

    paired = set()
    quad_faces = []
    for f0, f1 in matching:
        q = G.get_edge_data(f0, f1)['quad']
        n_est = np.cross(cv_pos[q[2]] - cv_pos[q[0]], cv_pos[q[3]] - cv_pos[q[1]])
        if np.dot(n_est, c_normals[f0] + c_normals[f1]) < 0:
            q = [q[0], q[3], q[2], q[1]]
        quad_faces.append(q)
        paired.add(f0); paired.add(f1)

    rem_tri = [list(c_faces[i]) for i in range(len(c_faces)) if i not in paired]
    quad_ratio = (len(quad_faces) / max(1, len(quad_faces) + len(rem_tri))) * 100.0

    print(f"    -> Extracted Polygons: {len(quad_faces) + len(rem_tri):,} | Quads: {len(quad_faces):,} ({quad_ratio:.1f}%)")

    stage4_obj = f"{base_prefix}_stage4_quads.obj"
    with open(stage4_obj, 'w') as f_out:
        f_out.write("# Stage 4 Quad-Dominant Extraction\n")
        v_lines = [f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n" for p in cv_pos]
        f_out.writelines(v_lines)
        q_lines = [f"f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n" for q in quad_faces]
        f_out.writelines(q_lines)
        t_lines = [f"f {t[0]+1} {t[1]+1} {t[2]+1}\n" for t in rem_tri]
        f_out.writelines(t_lines)

    print(f"[*] Stage 4 Complete in {time.time()-t0:.2f}s -> Saved: {stage4_obj}")
    return stage4_obj


# ==============================================================================
# STAGE 5: SURFACE REPRODUCTION & HIGH-FIDELITY SHRINKWRAP
# ==============================================================================
def run_stage_5(raw_mesh, stage4_obj, base_prefix, iterations=12, weight=0.45):
    print("\n" + "="*70)
    print(">>> [STAGE 5/5] Tangential Relaxation & Surface Reproduction (Shrinkwrap)...")
    print("="*70)
    t0 = time.time()

    # Load Stage 4 Polygons
    verts, polygons = [], []
    with open(stage4_obj, 'r') as f:
        for line in f:
            if line.startswith('v '):
                p = line.strip().split()
                verts.append([float(p[1]), float(p[2]), float(p[3])])
            elif line.startswith('f '):
                p = line.strip().split()
                polygons.append([int(x.split('/')[0]) - 1 for x in p[1:]])

    v_pos = np.array(verts, dtype=np.float64)
    n_verts = len(v_pos)

    # Reference KD-Tree from dense original
    pts_surface, _ = trimesh.sample.sample_surface(raw_mesh, count=min(200000, max(50000, len(raw_mesh.faces))))
    all_ref = np.vstack([pts_surface, raw_mesh.vertices])
    kdtree = cKDTree(all_ref)

    # Topology neighborhoods
    nbrs = [set() for _ in range(n_verts)]
    for poly in polygons:
        m = len(poly)
        for i in range(m):
            nbrs[poly[i]].add(poly[(i + 1) % m])
            nbrs[poly[(i + 1) % m]].add(poly[i])

    # Iterative Tangent Smoothing + Ground-Truth Projection
    for _ in range(iterations):
        v_next = v_pos.copy()
        for i in range(n_verts):
            if not nbrs[i]: continue
            centroid = np.mean(v_pos[list(nbrs[i])], axis=0)
            disp = centroid - v_pos[i]
            v_next[i] = v_pos[i] + weight * disp

        _, idxs = kdtree.query(v_next, k=1)
        v_pos = 0.20 * v_next + 0.80 * all_ref[idxs]

    final_obj = f"{base_prefix}_FINAL_RETOPO.obj"
    with open(final_obj, 'w') as out:
        out.write("# Final Production Quad-Dominant Asset\n")
        v_lines = [f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n" for v in v_pos]
        out.writelines(v_lines)
        poly_lines = [f"f {' '.join(str(idx + 1) for idx in poly)}\n" for poly in polygons]
        out.writelines(poly_lines)

    # Also export companion GLB
    glb_path = f"{base_prefix}_FINAL_RETOPO.glb"
    try:
        tri_list = []
        for poly in polygons:
            if len(poly) == 4:
                tri_list.append([poly[0], poly[1], poly[2]])
                tri_list.append([poly[0], poly[2], poly[3]])
            elif len(poly) == 3:
                tri_list.append(poly)
        trimesh.Trimesh(vertices=v_pos, faces=np.array(tri_list), process=False).export(glb_path)
        print(f"[*] Exported companion GLB: {glb_path}")
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
    base_prefix = os.path.splitext(args.input)[0]

    print("="*70)
    print("MASTER ANISOTROPIC RETOPOLOGY PIPELINE STARTED")
    print(f"Asset:  {args.input}")
    print(f"Budget: ~{args.faces:,} quads | Max Aspect: {args.aspect}:1")
    print("="*70)

    # Load & Sanitize Raw Input
    mesh = trimesh.load(args.input, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.merge_vertices()
    if hasattr(mesh, 'remove_duplicate_faces'): mesh.remove_duplicate_faces()
    if hasattr(mesh, 'remove_degenerate_faces'): mesh.remove_degenerate_faces()
    if hasattr(mesh, 'nondegenerate_faces'): mesh.update_faces(mesh.nondegenerate_faces())

    # Sequential Pipeline Execution
    locked_edges = run_stage_1(mesh, base_prefix, cv_high=35.0, cc_high=26.0)
    tensor_path = run_stage_2(mesh, base_prefix, target_quads=args.faces, max_aspect=args.aspect)
    field_path = run_stage_3(mesh, tensor_path, base_prefix, locked_edges)
    stage4_obj = run_stage_4(mesh, tensor_path, field_path, base_prefix, target_faces=args.faces, crease_deg=args.crease)
    final_mesh = run_stage_5(mesh, stage4_obj, base_prefix, iterations=12)

    print("\n" + "="*70)
    print("PIPELINE EXECUTION FINISHED SUCCESSFULLY")
    print(f"Total Processing Time: {time.time()-total_start:.2f}s")
    print(f"Final Model Exported:  {final_mesh}")
    print("="*70)

if __name__ == "__main__":
    main()
