import argparse
import numpy as np
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

def run_stage4_extraction(input_path, tensor_npy, field_npy, output_obj_path,
                          target_quads=3500, max_skew=0.80):
    print(f"[*] [STAGE 4] Loading base asset: {input_path}")
    raw_mesh = trimesh.load(input_path, force='mesh')
    if isinstance(raw_mesh, trimesh.Scene):
        raw_mesh = raw_mesh.dump(concatenate=True)
    
    # Coordinate welding to eliminate UV seam fragmentation
    _, u_idx, inv = np.unique(np.round(raw_mesh.vertices, 6), axis=0, return_index=True, return_inverse=True)
    raw_mesh = trimesh.Trimesh(vertices=raw_mesh.vertices[u_idx], faces=inv[raw_mesh.faces], process=True)
    raw_mesh = clean_trimesh(raw_mesh)

    print(f"    -> Raw Input: {len(raw_mesh.vertices):,} verts | {len(raw_mesh.faces):,} faces")

    # 1. Component-Safe Decimation to Target Carrier Budget
    target_tris = max(1200, target_quads * 2)
    print(f"[*] Building clean carrier mesh (~{target_tris:,} triangles)...")
    
    # Process connected bodies independently to preserve disconnected sub-meshes
    split_bodies = raw_mesh.split(only_watertight=False)
    simplified_bodies = []
    
    total_raw_faces = max(1, len(raw_mesh.faces))
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

    print(f"    -> Carrier Ready: {len(cv_pos):,} verts | {n_c_faces:,} faces")

    # 2. Ingest Precomputed Fields from Stage 2 & 3
    print(f"[*] Projecting Stage 2 Tensor & Stage 3 Cross-Field onto carrier...")
    raw_kdtree = cKDTree(raw_mesh.triangles_center)
    _, nearest_raw_idx = raw_kdtree.query(c_centers, k=1)

    tensor_data = np.load(tensor_npy)
    field_data = np.load(field_npy)

    carrier_h1 = tensor_data[nearest_raw_idx, 6]
    carrier_h2 = tensor_data[nearest_raw_idx, 7]
    carrier_aspect = carrier_h2 / (carrier_h1 + 1e-12)

    carrier_u = field_data[nearest_raw_idx, 0, :]
    carrier_v = field_data[nearest_raw_idx, 1, :]

    # 3. Assemble Blossom Pairing Graph
    print("[*] Evaluating triangle pairs along Stage 3 flow lines...")
    adj = carrier.face_adjacency
    adj_edges = carrier.face_adjacency_edges
    unshared = carrier.face_adjacency_unshared

    G = nx.Graph()

    for (f0, f1), (e0, e1), (u0, u1) in zip(adj, adj_edges, unshared):
        dot_n = np.dot(c_normals[f0], c_normals[f1])
        if dot_n < 0.5:  # Do not pair across sharp ridges (> 60 deg)
            continue

        q_idx = [u0, e0, u1, e1]
        pts = cv_pos[q_idx]

        edges = np.roll(pts, -1, axis=0) - pts
        lens = np.linalg.norm(edges, axis=1) + 1e-12
        dirs = edges / lens[:, None]

        # Corner orthogonality check
        cos_corners = [
            -np.dot(dirs[0], dirs[1]),
            -np.dot(dirs[1], dirs[2]),
            -np.dot(dirs[2], dirs[3]),
            -np.dot(dirs[3], dirs[0])
        ]
        skew = np.max(np.abs(cos_corners))
        if skew > max_skew:
            continue

        # Alignment with Stage 3 streamlines
        u_ref = carrier_u[f0]
        v_ref = carrier_v[f0]
        alignments = [max(abs(np.dot(d, u_ref)), abs(np.dot(d, v_ref))) for d in dirs]
        field_alignment = float(np.mean(alignments))

        # Target Aspect Ratio Conformity
        current_ratio = max(lens[0], lens[1]) / (min(lens[0], lens[1]) + 1e-12)
        target_ratio = min(5.0, carrier_aspect[f0])
        aspect_score = 1.0 - min(1.0, abs(current_ratio - target_ratio) / target_ratio)

        weight = (1.0 - skew) * 3.0 + field_alignment * 3.5 + aspect_score * 2.0 + dot_n
        G.add_edge(int(f0), int(f1), weight=float(weight), quad=q_idx)

    # 4. Maximum-Weight Blossom Matching
    print(f"[*] Solving global Blossom matching on {G.number_of_edges():,} valid candidates...")
    matching = nx.max_weight_matching(G, maxcardinality=False)

    paired = set()
    quad_faces = []

    for f0, f1 in matching:
        q = G.get_edge_data(f0, f1)['quad']
        # Normal winding check
        n_est = np.cross(cv_pos[q[2]] - cv_pos[q[0]], cv_pos[q[3]] - cv_pos[q[1]])
        if np.dot(n_est, c_normals[f0] + c_normals[f1]) < 0:
            q = [q[0], q[3], q[2], q[1]]
        quad_faces.append(q)
        paired.add(f0)
        paired.add(f1)

    rem_tri = [list(c_faces[i]) for i in range(n_c_faces) if i not in paired]
    total_polys = len(quad_faces) + len(rem_tri)
    quad_pct = (len(quad_faces) / max(1, total_polys)) * 100.0

    print(f"    -> Extracted Polygons: {total_polys:,}")
    print(f"    -> Quads: {len(quad_faces):,} ({quad_pct:.1f}%)")
    print(f"    -> Remainder Triangles: {len(rem_tri):,} ({100.0 - quad_pct:.1f}%)")

    # 5. Local Surface Snapping & Feature-Preserving Tangential Smoothing
    print("[*] Applying feature-preserving tangential surface smoothing...")
    kdtree_v = cKDTree(raw_mesh.vertices)
    _, nearest_idx = kdtree_v.query(cv_pos, k=1)
    projected_verts = 0.25 * cv_pos + 0.75 * raw_mesh.vertices[nearest_idx]

    # Build 1-ring vertex neighbors for geometric regularizing
    v_adj = [set() for _ in range(len(projected_verts))]
    for q in quad_faces:
        for i in range(4):
            v_adj[q[i]].add(q[(i+1)%4])
            v_adj[q[(i+1)%4]].add(q[i])
    for t in rem_tri:
        for i in range(3):
            v_adj[t[i]].add(t[(i+1)%3])
            v_adj[t[(i+1)%3]].add(t[i])

    poly_tris = list(rem_tri)
    for q in quad_faces:
        poly_tris.append([q[0], q[1], q[2]])
        poly_tris.append([q[0], q[2], q[3]])

    smoothed_v = projected_verts.copy()
    for it in range(6):
        temp_mesh = trimesh.Trimesh(vertices=smoothed_v, faces=poly_tris, process=False)
        curr_vn = temp_mesh.vertex_normals
        disp = np.zeros_like(smoothed_v)
        for i in range(len(smoothed_v)):
            nbrs = list(v_adj[i])
            if not nbrs:
                continue
            delta = np.mean(smoothed_v[nbrs], axis=0) - smoothed_v[i]
            # Tangential projection: cancel normal displacement to preserve silhouette & volume
            norm = curr_vn[i]
            disp[i] = delta - np.dot(delta, norm) * norm
        smoothed_v += 0.35 * disp

    final_mesh = trimesh.Trimesh(vertices=smoothed_v, faces=poly_tris, process=False)
    smooth_vn = final_mesh.vertex_normals

    # 6. Export as Wavefront OBJ with Smooth Normals and True 4-Index Quad Definitions
    print(f"[*] Writing Smooth Quad OBJ to: {output_obj_path}")
    with open(output_obj_path, 'w') as f_out:
        f_out.write("# Stage 4 Quad-Dominant Asset (Smooth Shaded)\n")
        for v in smoothed_v:
            f_out.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for n in smooth_vn:
            f_out.write(f"vn {n[0]:.6f} {n[1]:.6f} {n[2]:.6f}\n")
        f_out.write("s 1\n")
        for q in quad_faces:
            f_out.write(f"f {q[0]+1}//{q[0]+1} {q[1]+1}//{q[1]+1} {q[2]+1}//{q[2]+1} {q[3]+1}//{q[3]+1}\n")
        for t in rem_tri:
            f_out.write(f"f {t[0]+1}//{t[0]+1} {t[1]+1}//{t[1]+1} {t[2]+1}//{t[2]+1}\n")

    if output_obj_path.endswith('.obj'):
        glb_out = output_obj_path.replace('.obj', '.glb')
        try:
            final_mesh.export(glb_out)
            print(f"    -> Also exported Smooth GLB companion: {glb_out}")
        except Exception:
            pass

    print(f"[+] [STAGE 4 COMPLETE] Output saved to: {output_obj_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Anisotropic Quad-Dominant Extraction")
    parser.add_argument("input", help="Raw mesh path (.glb, .obj)")
    parser.add_argument("tensor", help="Stage 2 _tensor.npy path")
    parser.add_argument("field", help="Stage 3 _field.npy path")
    parser.add_argument("output", help="Output quad OBJ path")
    parser.add_argument("--quads", type=int, default=3500, help="Target quad face count")
    args = parser.parse_args()

    run_stage4_extraction(args.input, args.tensor, args.field, args.output, target_quads=args.quads)
