import argparse
import os
import time
import numpy as np
import scipy.sparse as sp
import trimesh
import networkx as nx

def run_stage4_aniso_extraction(
    input_path, 
    tensor_npy, 
    field_npy, 
    output_path,
    sharp_crease_deg=28.0,
    solver="auto",
    rails_path=None
):
    print(f"[*] [STAGE 4] Loading asset: {input_path}")
    raw_mesh = trimesh.load(input_path, force='mesh')
    if isinstance(raw_mesh, trimesh.Scene):
        raw_mesh = raw_mesh.dump(concatenate=True)
    raw_mesh.merge_vertices()
    if hasattr(raw_mesh, 'remove_duplicate_faces'):
        raw_mesh.remove_duplicate_faces()
    if hasattr(raw_mesh, 'remove_degenerate_faces'):
        raw_mesh.remove_degenerate_faces()
    if hasattr(raw_mesh, 'nondegenerate_faces'):
        raw_mesh.update_faces(raw_mesh.nondegenerate_faces())

    v = raw_mesh.vertices.copy()
    f = raw_mesh.faces.copy()
    face_normals = raw_mesh.face_normals
    n_faces = len(f)

    print(f"    -> Mesh: {len(v):,} vertices | {n_faces:,} faces")

    # 1. Load Precomputed Fields
    print(f"[*] Loading Stage 2 Tensor: {tensor_npy}")
    tensor_data = np.load(tensor_npy)
    h1 = tensor_data[:, 6]
    h2 = tensor_data[:, 7]
    target_aspect = h2 / (h1 + 1e-12)

    print(f"[*] Loading Stage 3 Cross-Field: {field_npy}")
    field_data = np.load(field_npy)
    u_dir = field_data[:, 0, :]
    v_dir = field_data[:, 1, :]

    # 2. Extract Hard Crease Rails
    cos_sharp = np.cos(np.radians(sharp_crease_deg))
    adj_faces = raw_mesh.face_adjacency
    adj_edges = raw_mesh.face_adjacency_edges
    unshared_v = raw_mesh.face_adjacency_unshared

    # Auto-detect boundary rails if not given
    if rails_path is None:
        cand_dir = os.path.dirname(output_path) or "."
        base_name = os.path.basename(input_path).split('.')[0].lower()
        for cand in [
            os.path.join(cand_dir, f"{base_name}_rails_robust.obj"),
            os.path.join("vacuum_quad_engine/assets/outputs", f"{base_name}_rails_robust.obj")
        ]:
            if os.path.exists(cand):
                rails_path = cand
                break

    # 3. Assemble Anisotropic Pairing Graph
    print("[*] Evaluating field-guided anisotropic triangle pairings...")
    t0 = time.time()
    
    # Vectorized initial filtering for speed
    f0_arr = adj_faces[:, 0]
    f1_arr = adj_faces[:, 1]
    dot_n_arr = np.sum(face_normals[f0_arr] * face_normals[f1_arr], axis=1)
    
    # Filter out sharp crease edges
    non_crease_mask = dot_n_arr >= cos_sharp
    valid_indices = np.where(non_crease_mask)[0]

    G = nx.Graph()
    candidate_edges = []

    for idx in valid_indices:
        f0 = int(adj_faces[idx, 0])
        f1 = int(adj_faces[idx, 1])
        e0 = int(adj_edges[idx, 0])
        e1 = int(adj_edges[idx, 1])
        u0 = int(unshared_v[idx, 0])
        u1 = int(unshared_v[idx, 1])
        dot_n = float(dot_n_arr[idx])

        q_idx = [u0, e0, u1, e1]
        pts = v[q_idx]

        # Geometry of candidate quad
        edges = np.roll(pts, -1, axis=0) - pts
        lens = np.linalg.norm(edges, axis=1) + 1e-12
        dirs = edges / lens[:, None]

        # Orthogonality metric (ideal rectangle has opposite dot ~ 0)
        cos_corners = [
            -np.dot(dirs[0], dirs[1]),
            -np.dot(dirs[1], dirs[2]),
            -np.dot(dirs[2], dirs[3]),
            -np.dot(dirs[3], dirs[0])
        ]
        max_skew = np.max(np.abs(cos_corners))
        if max_skew > 0.75:  # Reject heavily distorted/pinched corners
            continue

        # Alignment with Stage 3 (u, v) streamlines
        u_ref = u_dir[f0]
        v_ref = v_dir[f0]
        alignments = [max(abs(np.dot(d, u_ref)), abs(np.dot(d, v_ref))) for d in dirs]
        field_score = float(np.mean(alignments))

        # Anisotropic aspect reward
        current_aspect = max(lens[0], lens[1]) / (min(lens[0], lens[1]) + 1e-12)
        desired_aspect = min(5.0, target_aspect[f0])
        aspect_fit = 1.0 - min(1.0, abs(current_aspect - desired_aspect) / desired_aspect)

        # Combined weight for matching
        weight = float((1.0 - max_skew) * 3.0 + field_score * 2.5 + aspect_fit * 2.0 + dot_n * 1.0)
        G.add_edge(f0, f1, weight=weight, quad=q_idx)
        candidate_edges.append((f0, f1, weight, q_idx))

    print(f"    -> Formed {len(candidate_edges):,} valid candidate pairs in {time.time() - t0:.2f}s.")

    # 4. Maximum Weight Matching Solver
    choose_solver = solver
    if choose_solver == "auto":
        choose_solver = "blossom" if n_faces <= 30000 else "fast"

    print(f"[*] Solving maximum-weight graph matching (solver: {choose_solver})...")
    t_match_start = time.time()
    paired_faces = set()
    quad_faces = []

    if choose_solver == "blossom":
        # Fast component-wise exact Edmonds' Blossom algorithm
        matching = set()
        for comp in nx.connected_components(G):
            if len(comp) > 1:
                sub_g = G.subgraph(comp)
                sub_matching = nx.max_weight_matching(sub_g, maxcardinality=False)
                matching.update(sub_matching)

        for f0, f1 in matching:
            edge_data = G.get_edge_data(f0, f1)
            q = edge_data['quad']

            # Normal winding consistency check
            n_est = np.cross(v[q[2]] - v[q[0]], v[q[3]] - v[q[1]])
            if np.dot(n_est, face_normals[f0] + face_normals[f1]) < 0:
                q = [q[0], q[3], q[2], q[1]]

            quad_faces.append(q)
            paired_faces.add(f0)
            paired_faces.add(f1)
    else:
        # Priority greedy matching for massive meshes (>30,000 faces)
        candidate_edges.sort(key=lambda x: x[2], reverse=True)
        matched_flags = np.zeros(n_faces, dtype=bool)

        for f0, f1, weight, q in candidate_edges:
            if not matched_flags[f0] and not matched_flags[f1]:
                matched_flags[f0] = True
                matched_flags[f1] = True
                n_est = np.cross(v[q[2]] - v[q[0]], v[q[3]] - v[q[1]])
                if np.dot(n_est, face_normals[f0] + face_normals[f1]) < 0:
                    q = [q[0], q[3], q[2], q[1]]
                quad_faces.append(q)
                paired_faces.add(f0)
                paired_faces.add(f1)

    print(f"    -> Matching solved in {time.time() - t_match_start:.2f}s.")

    # Keep remaining transition triangles
    rem_tri_faces = [list(face) for i, face in enumerate(f) if i not in paired_faces]

    total_polys = len(quad_faces) + len(rem_tri_faces)
    quad_percent = (len(quad_faces) / max(1, total_polys)) * 100.0

    print(f"    -> Extracted Polygons: {total_polys:,}")
    print(f"    -> Quads: {len(quad_faces):,} ({quad_percent:.1f}%)")
    print(f"    -> Triangles: {len(rem_tri_faces):,} ({100.0 - quad_percent:.1f}%)")

    # 5. Export Quad-Dominant OBJ
    print(f"[*] Exporting quad mesh to: {output_path}")
    with open(output_path, 'w') as out_f:
        out_f.write("# Anisotropic Quad-Dominant Mesh (Stage 4)\n")
        out_f.write(f"# Quads: {len(quad_faces):,} | Triangles: {len(rem_tri_faces):,}\n")
        v_lines = [f"v {vert[0]:.6f} {vert[1]:.6f} {vert[2]:.6f}\n" for vert in v]
        out_f.writelines(v_lines)
        
        q_lines = [f"f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n" for q in quad_faces]
        out_f.writelines(q_lines)

        t_lines = [f"f {t[0]+1} {t[1]+1} {t[2]+1}\n" for t in rem_tri_faces]
        out_f.writelines(t_lines)

    print(f"[+] [STAGE 4 COMPLETE] Saved to: {output_path}")

    # Also export companion triangulated GLB for instant viewing in standard 3D viewers
    if output_path.endswith(".obj"):
        glb_path = output_path.replace(".obj", ".glb")
        try:
            # Triangulate quads for universal glb export
            tri_for_glb = []
            for q in quad_faces:
                tri_for_glb.append([q[0], q[1], q[2]])
                tri_for_glb.append([q[0], q[2], q[3]])
            for t in rem_tri_faces:
                tri_for_glb.append(t)
            glb_mesh = trimesh.Trimesh(vertices=v, faces=np.array(tri_for_glb), process=False)
            glb_mesh.export(glb_path)
            print(f"[+] Also exported GLB companion: {glb_path}")
        except Exception as e:
            print(f"    -> GLB companion note: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Anisotropic Quad-Dominant Extraction")
    parser.add_argument("input", help="Raw input mesh (.obj, .glb, .stl)")
    parser.add_argument("tensor", help="Stage 2 _tensor.npy path")
    parser.add_argument("field", help="Stage 3 _field.npy path")
    parser.add_argument("output", help="Output quad OBJ path")
    parser.add_argument("--crease", type=float, default=28.0, help="Sharp crease cutoff (deg)")
    parser.add_argument("--solver", type=str, default="auto", choices=["auto", "blossom", "fast"], help="Matching solver (auto, blossom, fast)")
    parser.add_argument("--rails", type=str, default=None, help="Stage 1 boundary rails file (.obj)")
    args = parser.parse_args()

    run_stage4_aniso_extraction(
        args.input, 
        args.tensor, 
        args.field, 
        args.output,
        sharp_crease_deg=args.crease,
        solver=args.solver,
        rails_path=args.rails
    )

