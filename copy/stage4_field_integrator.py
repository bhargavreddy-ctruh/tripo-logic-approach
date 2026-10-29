import argparse
import os
import time
import numpy as np
import trimesh
import networkx as nx
from scipy.spatial import cKDTree

def run_stage4_integrated(
    input_path, 
    tensor_npy, 
    field_npy, 
    output_path,
    target_faces=4000, 
    sharp_crease_deg=28.0
):
    print(f"[*] [STAGE 4 INTEGRATOR] Loading base asset: {input_path}")
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

    orig_face_count = len(raw_mesh.faces)

    # 1. Clean Carrier Simplification
    # Simplifies the mesh to a workable carrier count before pairing into quads
    carrier_target_tris = target_faces * 2
    if orig_face_count > carrier_target_tris:
        print(f"[*] Building clean carrier mesh: {orig_face_count:,} -> ~{carrier_target_tris:,} triangles...")
        try:
            carrier = raw_mesh.simplify_quadric_decimation(face_count=carrier_target_tris)
        except Exception:
            ratio = float(carrier_target_tris) / float(orig_face_count)
            carrier = raw_mesh.simplify_quadric_decimation(percent=min(0.99, max(0.01, ratio)))
    else:
        carrier = raw_mesh.copy()

    carrier.merge_vertices()
    if hasattr(carrier, 'remove_duplicate_faces'):
        carrier.remove_duplicate_faces()
    if hasattr(carrier, 'remove_degenerate_faces'):
        carrier.remove_degenerate_faces()
    if hasattr(carrier, 'nondegenerate_faces'):
        carrier.update_faces(carrier.nondegenerate_faces())

    cv_pos = carrier.vertices
    c_faces = carrier.faces
    c_normals = carrier.face_normals
    c_centers = carrier.triangles_center
    n_c_faces = len(c_faces)

    print(f"    -> Carrier Ready: {len(cv_pos):,} vertices | {n_c_faces:,} faces")

    # 2. Ingest Stage 2 & Stage 3 Precomputed Fields
    print(f"[*] Ingesting Stage 2 Tensor ({tensor_npy}) & Stage 3 Field ({field_npy})...")
    raw_centers = raw_mesh.triangles_center
    raw_kdtree = cKDTree(raw_centers)

    tensor_data = np.load(tensor_npy)
    field_data = np.load(field_npy)

    # Nearest-neighbor field projection from raw mesh faces to carrier faces
    _, nearest_raw_idx = raw_kdtree.query(c_centers, k=1)

    carrier_h1 = tensor_data[nearest_raw_idx, 6]
    carrier_h2 = tensor_data[nearest_raw_idx, 7]
    carrier_aspect = carrier_h2 / (carrier_h1 + 1e-12)

    carrier_u = field_data[nearest_raw_idx, 0, :]
    carrier_v = field_data[nearest_raw_idx, 1, :]

    # 3. Form Feature-Protected Anisotropic Pairing Graph
    print("[*] Assembling Blossom pairing graph guided by Stage 3 flow lines...")
    cos_sharp = np.cos(np.radians(sharp_crease_deg))
    adj_faces = carrier.face_adjacency
    adj_edges = carrier.face_adjacency_edges
    unshared_v = carrier.face_adjacency_unshared

    G = nx.Graph()

    for (f0, f1), (e0, e1), (u0, u1) in zip(adj_faces, adj_edges, unshared_v):
        # Prevent merging across sharp feature creases
        dot_n = np.dot(c_normals[f0], c_normals[f1])
        if dot_n < cos_sharp:
            continue

        q_idx = [u0, e0, u1, e1]
        pts = cv_pos[q_idx]

        # Edge vectors
        edges = np.roll(pts, -1, axis=0) - pts
        lens = np.linalg.norm(edges, axis=1) + 1e-12
        dirs = edges / lens[:, None]

        # Planarity / Skew Metric
        cos_corners = [
            -np.dot(dirs[0], dirs[1]),
            -np.dot(dirs[1], dirs[2]),
            -np.dot(dirs[2], dirs[3]),
            -np.dot(dirs[3], dirs[0])
        ]
        max_skew = np.max(np.abs(cos_corners))
        if max_skew > 0.85:
            continue

        # Alignment with Stage 3 Flow Field
        u_ref = carrier_u[f0]
        v_ref = carrier_v[f0]
        alignments = [max(abs(np.dot(d, u_ref)), abs(np.dot(d, v_ref))) for d in dirs]
        field_alignment = float(np.mean(alignments))

        # Aspect Ratio Fit
        current_ratio = max(lens[0], lens[1]) / (min(lens[0], lens[1]) + 1e-12)
        target_ratio = min(5.0, carrier_aspect[f0])
        aspect_score = 1.0 - min(1.0, abs(current_ratio - target_ratio) / target_ratio)

        weight = (1.0 - max_skew) * 3.0 + field_alignment * 3.0 + aspect_score * 2.0 + dot_n
        G.add_edge(int(f0), int(f1), weight=float(weight), quad=q_idx)

    # 4. Edmonds' Maximum Weight Matching (Component-wise Blossom Algorithm)
    print(f"[*] Solving global Blossom matching on {G.number_of_edges():,} valid pairs...")
    t0 = time.time()
    matching = set()
    for comp in nx.connected_components(G):
        if len(comp) > 1:
            sub_g = G.subgraph(comp)
            sub_m = nx.max_weight_matching(sub_g, maxcardinality=False)
            matching.update(sub_m)
    print(f"    -> Blossom solved in {time.time() - t0:.2f}s.")

    paired_faces = set()
    quad_faces = []

    for f0, f1 in matching:
        q = G.get_edge_data(f0, f1)['quad']
        # Winding check
        n_est = np.cross(cv_pos[q[2]] - cv_pos[q[0]], cv_pos[q[3]] - cv_pos[q[1]])
        if np.dot(n_est, c_normals[f0] + c_normals[f1]) < 0:
            q = [q[0], q[3], q[2], q[1]]
        quad_faces.append(q)
        paired_faces.add(f0)
        paired_faces.add(f1)

    rem_tri_faces = [list(c_faces[i]) for i in range(n_c_faces) if i not in paired_faces]

    total_polys = len(quad_faces) + len(rem_tri_faces)
    quad_pct = (len(quad_faces) / max(1, total_polys)) * 100.0

    print(f"    -> Extracted Polygons: {total_polys:,}")
    print(f"    -> Quads: {len(quad_faces):,} ({quad_pct:.1f}%)")
    print(f"    -> Transition Triangles: {len(rem_tri_faces):,} ({100.0 - quad_pct:.1f}%)")

    # 5. Export Mesh
    print(f"[*] Writing quad mesh to: {output_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, 'w') as f_out:
        f_out.write("# Stage 4 Quad-Dominant Extracted Mesh\n")
        f_out.write(f"# Quads: {len(quad_faces):,} | Triangles: {len(rem_tri_faces):,}\n")
        v_lines = [f"v {p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n" for p in cv_pos]
        f_out.writelines(v_lines)
        q_lines = [f"f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n" for q in quad_faces]
        f_out.writelines(q_lines)
        t_lines = [f"f {t[0]+1} {t[1]+1} {t[2]+1}\n" for t in rem_tri_faces]
        f_out.writelines(t_lines)

    print(f"[+] [STAGE 4 COMPLETE] Saved to: {output_path}")

    # Also export companion GLB for direct preview in 3D viewers
    if output_path.endswith(".obj"):
        glb_path = output_path.replace(".obj", ".glb")
        try:
            tri_list = []
            for q in quad_faces:
                tri_list.append([q[0], q[1], q[2]])
                tri_list.append([q[0], q[2], q[3]])
            for t in rem_tri_faces:
                tri_list.append(t)
            glb_mesh = trimesh.Trimesh(vertices=cv_pos, faces=np.array(tri_list), process=False)
            glb_mesh.export(glb_path)
            print(f"[+] Also exported GLB companion: {glb_path}")
        except Exception as e:
            print(f"    -> GLB companion note: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Integrated Anisotropic Quad Extraction")
    parser.add_argument("input", help="Raw input mesh")
    parser.add_argument("tensor", help="Stage 2 _tensor.npy")
    parser.add_argument("field", help="Stage 3 _field.npy")
    parser.add_argument("output", help="Output quad OBJ")
    parser.add_argument("--faces", type=int, default=4000, help="Target face count budget")
    parser.add_argument("--crease", type=float, default=28.0, help="Crease threshold (deg)")
    args = parser.parse_args()

    run_stage4_integrated(
        args.input, 
        args.tensor, 
        args.field, 
        args.output,
        target_faces=args.faces, 
        sharp_crease_deg=args.crease
    )
