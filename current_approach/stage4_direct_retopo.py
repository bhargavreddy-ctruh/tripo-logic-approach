import argparse
import os
import sys
import numpy as np
import trimesh
import networkx as nx
from scipy.spatial import cKDTree

def run_clean_retopo(
    input_path, 
    output_obj_path, 
    tensor_path=None,
    field_path=None,
    target_quads=3500, 
    max_skew=0.94
):
    print(f"[*] [STAGE 4 CLEAN] Loading asset: {input_path}")
    mesh = trimesh.load(input_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)

    # Clean manifold geometry & weld spatial coordinate duplicates
    _, u_idx, inv = np.unique(np.round(mesh.vertices, 6), axis=0, return_index=True, return_inverse=True)
    mesh = trimesh.Trimesh(vertices=mesh.vertices[u_idx], faces=inv[mesh.faces], process=True)
    mesh.merge_vertices()
    if hasattr(mesh, 'remove_duplicate_faces'):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    elif hasattr(mesh, 'nondegenerate_faces'):
        mesh.update_faces(mesh.nondegenerate_faces())

    n_raw_verts = len(mesh.vertices)
    n_raw_faces = len(mesh.faces)
    print(f"    -> Original Mesh: {n_raw_verts:,} vertices | {n_raw_faces:,} faces")

    # Resolve required inputs from Stage 2 and Stage 3
    cand_dir = os.path.dirname(output_obj_path) or "."
    base_name = os.path.basename(input_path).split('.')[0].lower()

    if tensor_path is None:
        for cand in [
            os.path.join(cand_dir, f"{base_name}_stage2_tensor.npy"),
            os.path.join(cand_dir, f"{base_name}_tensor.npy"),
            os.path.join("vacuum_quad_engine/assets/outputs", f"{base_name}_tensor_check_tensor.npy"),
            os.path.join("vacuum_quad_engine/assets/outputs", f"{base_name}_stage2_tensor.npy"),
            "beetle_car_tensor_check_tensor.npy"
        ]:
            if os.path.exists(cand):
                tensor_path = cand
                break

    if field_path is None:
        for cand in [
            os.path.join(cand_dir, f"{base_name}_stage3_field.npy"),
            os.path.join(cand_dir, f"{base_name}_field.npy"),
            os.path.join("vacuum_quad_engine/assets/outputs", f"{base_name}_stage3_field.npy"),
            "beetle_car_stage3_field.npy"
        ]:
            if os.path.exists(cand):
                field_path = cand
                break

    tensor_data = None
    if tensor_path and os.path.exists(tensor_path):
        print(f"[*] Ingesting Stage 2 Metric Tensor: {tensor_path}")
        tensor_data = np.load(tensor_path)

    field_data = None
    if field_path and os.path.exists(field_path):
        print(f"[*] Ingesting Stage 3 Cross-Field: {field_path}")
        field_data = np.load(field_path)

    # 1. Target Coarse Triangle Budget
    target_tris = max(1000, target_quads * 2)

    # 2. Part-Preserving Surface Resampling (Zero Spikes, Zero Bounding-Box Bridges)
    print(f"[*] Resampling surface to coarse carrier ({target_tris:,} triangles)...")
    carrier = mesh.copy()
    if len(carrier.faces) > target_tris:
        try:
            carrier = carrier.simplify_quadric_decimation(face_count=target_tris)
        except Exception:
            ratio = min(0.99, max(0.01, float(target_tris) / len(carrier.faces)))
            carrier = carrier.simplify_quadric_decimation(percent=ratio)

    carrier.merge_vertices()
    if hasattr(carrier, 'remove_duplicate_faces'):
        carrier.remove_duplicate_faces()
    if hasattr(carrier, 'remove_degenerate_faces'):
        carrier.remove_degenerate_faces()
    elif hasattr(carrier, 'nondegenerate_faces'):
        carrier.update_faces(carrier.nondegenerate_faces())

    c_verts = carrier.vertices.copy()
    c_faces = carrier.faces
    c_normals = carrier.face_normals
    print(f"    -> Coarse Carrier: {len(c_verts):,} vertices | {len(c_faces):,} faces")

    # Project Stage 2 tensor and Stage 3 field onto coarse carrier
    carrier_aspect = None
    carrier_u = None
    carrier_v = None
    if tensor_data is not None or field_data is not None:
        raw_tree = cKDTree(mesh.triangles_center)
        _, nearest_f = raw_tree.query(carrier.triangles_center, k=1)
        if tensor_data is not None:
            c_h1 = tensor_data[nearest_f, 6]
            c_h2 = tensor_data[nearest_f, 7]
            carrier_aspect = c_h2 / (c_h1 + 1e-12)
        if field_data is not None:
            carrier_u = field_data[nearest_f, 0, :]
            carrier_v = field_data[nearest_f, 1, :]

    # 3. Direct Edge-Pairing Graph for Quad Formation
    adj = carrier.face_adjacency
    adj_edges = carrier.face_adjacency_edges
    unshared = carrier.face_adjacency_unshared

    G = nx.Graph()

    for idx, ((f0, f1), (e0, e1), (u0, u1)) in enumerate(zip(adj, adj_edges, unshared)):
        dot_n = np.dot(c_normals[f0], c_normals[f1])
        if dot_n < 0.5:  # Don't pair across sharp borders (> 60 degrees)
            continue

        n_avg = c_normals[f0] + c_normals[f1]
        # Convexity: edge (e0, e1) separates u0 and u1
        E = c_verts[e1] - c_verts[e0]
        V0 = c_verts[u0] - c_verts[e0]
        V1 = c_verts[u1] - c_verts[e0]
        if np.dot(np.cross(E, V0), n_avg) * np.dot(np.cross(E, V1), n_avg) >= 0:
            continue

        # Form quad indices: [u0, e0, u1, e1]
        q_idx = [u0, e0, u1, e1]
        pts = c_verts[q_idx]

        edges = np.roll(pts, -1, axis=0) - pts
        lens = np.linalg.norm(edges, axis=1) + 1e-12
        dirs = edges / lens[:, None]

        # Check corner angles (ideal square/rectangle has dot product ~ 0)
        cos_corners = [
            -np.dot(dirs[0], dirs[1]),
            -np.dot(dirs[1], dirs[2]),
            -np.dot(dirs[2], dirs[3]),
            -np.dot(dirs[3], dirs[0])
        ]
        skew = np.max(np.abs(cos_corners))
        if skew > max_skew:
            continue

        # Geometry score
        aspect_ratio = np.min(lens) / np.max(lens)
        weight = (1.0 - skew) * 3.0 + aspect_ratio * 2.0 + dot_n

        # Field alignment bonus from Stage 3
        if carrier_u is not None and carrier_v is not None:
            u_ref = carrier_u[f0]
            v_ref = carrier_v[f0]
            aligns = [max(abs(np.dot(d, u_ref)), abs(np.dot(d, v_ref))) for d in dirs]
            field_score = float(np.mean(aligns))
            weight += field_score * 3.0

        # Anisotropic stretching bonus from Stage 2
        if carrier_aspect is not None:
            current_asp = np.max(lens) / np.min(lens)
            target_asp = min(5.0, float(carrier_aspect[f0]))
            asp_fit = 1.0 - min(1.0, abs(current_asp - target_asp) / target_asp)
            weight += asp_fit * 2.0

        G.add_edge(int(f0), int(f1), weight=float(weight), quad=q_idx)

    # 4. Maximum Weight Matching
    print(f"[*] Pairing coarse triangles into quads across {G.number_of_edges():,} valid candidates...")
    matching = nx.max_weight_matching(G, maxcardinality=False)

    paired = set()
    quad_polys = []

    for f0, f1 in matching:
        q = G.get_edge_data(f0, f1)['quad']
        # Winding order consistency
        n_est = np.cross(c_verts[q[2]] - c_verts[q[0]], c_verts[q[3]] - c_verts[q[1]])
        if np.dot(n_est, c_normals[f0] + c_normals[f1]) < 0:
            q = [q[0], q[3], q[2], q[1]]
        quad_polys.append(q)
        paired.add(f0)
        paired.add(f1)

    # Remaining triangles maintain 100% surface coverage
    tri_polys = [list(c_faces[i]) for i in range(len(c_faces)) if i not in paired]

    total = len(quad_polys) + len(tri_polys)
    quad_pct = (len(quad_polys) / max(1, total)) * 100.0

    print(f"    -> Extracted Polygons: {total:,}")
    print(f"    -> Quads: {len(quad_polys):,} ({quad_pct:.1f}%)")
    print(f"    -> Remainder Triangles: {len(tri_polys):,} ({100.0 - quad_pct:.1f}%)")

    # 5. Snap Coarse Vertices to Original High-Res Mesh (Safe, localized projection)
    kdtree = cKDTree(mesh.vertices)
    _, nearest_idx = kdtree.query(c_verts, k=1)
    # Blend slightly so vertices don't collapse into clusters
    final_verts = 0.2 * c_verts + 0.8 * mesh.vertices[nearest_idx]

    # 6. Export as Wavefront OBJ with true Quad definitions
    print(f"[*] Exporting clean quad-dominant OBJ to: {output_obj_path}")
    os.makedirs(os.path.dirname(os.path.abspath(output_obj_path)), exist_ok=True)
    with open(output_obj_path, 'w') as f_out:
        f_out.write("# Clean Quad-Dominant Extraction (Stage 4)\n")
        f_out.write(f"# Quads: {len(quad_polys):,} | Triangles: {len(tri_polys):,} | Total: {total:,}\n")
        for v in final_verts:
            f_out.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for q in quad_polys:
            f_out.write(f"f {q[0]+1} {q[1]+1} {q[2]+1} {q[3]+1}\n")
        for t in tri_polys:
            f_out.write(f"f {t[0]+1} {t[1]+1} {t[2]+1}\n")

    print(f"[+] [SUCCESS] Quad mesh saved to: {output_obj_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4 Clean Quad Extraction")
    parser.add_argument("input", help="Raw input asset (.glb, .obj)")
    parser.add_argument("output", help="Output quad OBJ path")
    parser.add_argument("--tensor", type=str, default=None, help="Stage 2 _tensor.npy path")
    parser.add_argument("--field", type=str, default=None, help="Stage 3 _field.npy path")
    parser.add_argument("--quads", type=int, default=3500, help="Target quad face count")
    parser.add_argument("--max-skew", type=float, default=0.94, help="Corner skew threshold (default: 0.94)")
    
    # Handle possible 4 positional arguments: input tensor field output
    raw_args = sys.argv[1:]
    if len(raw_args) >= 4 and not raw_args[0].startswith('-') and not raw_args[1].startswith('-') and not raw_args[2].startswith('-') and not raw_args[3].startswith('-'):
        in_path = raw_args[0]
        ten_path = raw_args[1]
        fld_path = raw_args[2]
        out_path = raw_args[3]
        run_clean_retopo(in_path, out_path, tensor_path=ten_path, field_path=fld_path)
    else:
        args = parser.parse_args()
        run_clean_retopo(
            args.input, 
            args.output, 
            tensor_path=args.tensor,
            field_path=args.field,
            target_quads=args.quads,
            max_skew=args.max_skew
        )

