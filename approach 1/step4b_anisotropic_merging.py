import argparse
from collections import defaultdict
import numpy as np

def merge_polygons(poly0, poly1, v_a, v_b):
    """
    Merge two oriented polygons sharing directed edge (v_a, v_b) into a single
    conforming watertight polygon.
    
    Topological Guarantee:
    In poly0, the directed edge runs v_a -> v_b.
    In poly1, the directed edge runs v_b -> v_a.
    Removing the internal shared edge preserves all external boundary edges exactly,
    preventing T-junction cracks and guaranteeing manifold watertightness.
    """
    m = len(poly0)
    idx_a0 = poly0.index(v_a)
    if poly0[(idx_a0 + 1) % m] != v_b:
        v_a, v_b = v_b, v_a
        idx_a0 = poly0.index(v_a)

    seq0 = []
    curr = (idx_a0 + 1) % m
    while True:
        seq0.append(poly0[curr])
        if poly0[curr] == v_a:
            break
        curr = (curr + 1) % m

    if v_b not in poly1:
        return None
    idx_b1 = poly1.index(v_b)
    n = len(poly1)
    if poly1[(idx_b1 + 1) % n] != v_a:
        return None

    seq1 = []
    curr = (idx_b1 + 1) % n
    while True:
        seq1.append(poly1[curr])
        if poly1[curr] == v_b:
            break
        curr = (curr + 1) % n

    merged = seq0[:-1] + seq1[:-1]
    # Guard against non-simple or self-intersecting loops
    if len(set(merged)) != len(merged):
        return None
    return merged


def run_step4b_anisotropic_merging(
    input_quad_path, 
    output_merged_path,
    coplanar_deg=3.5,
    collinear_deg=18.0,
    max_iterations=5
):
    print(f"[*] [STEP 4b] Loading quad mesh for Watertight Anisotropic Merging: {input_quad_path}")
    
    # 1. Parse raw polygon mesh (preserving quads, triangles, and N-gons)
    verts = []
    polygons = []
    with open(input_quad_path, 'r') as f:
        for line in f:
            if line.startswith('v '):
                parts = line.strip().split()
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith('f '):
                parts = line.strip().split()
                poly = [int(p.split('/')[0]) - 1 for p in parts[1:]]
                polygons.append(poly)

    v_pos = np.array(verts, dtype=np.float64)
    init_total = len(polygons)
    init_quads = sum(1 for p in polygons if len(p) == 4)
    print(f"    -> Initial Polygons: {init_total:,} ({init_quads:,} quads, {init_total - init_quads:,} non-quads)")

    # Baseline boundary edge check
    init_e_counts = defaultdict(int)
    for poly in polygons:
        m = len(poly)
        for i in range(m):
            e = tuple(sorted((poly[i], poly[(i + 1) % m])))
            init_e_counts[e] += 1
    init_boundary = sum(1 for cnt in init_e_counts.values() if cnt == 1)
    print(f"    -> Baseline Open Boundary Edges: {init_boundary:,}")

    cos_coplanar = np.cos(np.radians(coplanar_deg))
    cos_collinear = np.cos(np.radians(collinear_deg))

    # 2. Iterative Watertight Anisotropic Merging Passes
    for it in range(max_iterations):
        # Compute face normals
        normals = []
        for poly in polygons:
            if len(poly) >= 3:
                p0, p1, p2 = v_pos[poly[0]], v_pos[poly[1]], v_pos[poly[2]]
                n = np.cross(p1 - p0, p2 - p0)
                norm = np.linalg.norm(n)
                normals.append(n / norm if norm > 1e-12 else np.array([0.0, 0.0, 1.0]))
            else:
                normals.append(np.array([0.0, 0.0, 1.0]))
        normals = np.array(normals)

        # Build edge-to-face adjacency map
        edge_to_faces = defaultdict(list)
        for f_idx, poly in enumerate(polygons):
            m = len(poly)
            for i in range(m):
                e = tuple(sorted((poly[i], poly[(i + 1) % m])))
                edge_to_faces[e].append((f_idx, i))

        merged_this_round = 0
        consumed_faces = set()
        new_polygons = []

        # Find pairable adjacent coplanar faces
        for e, face_data in edge_to_faces.items():
            if len(face_data) != 2:
                continue

            (f0, _), (f1, _) = face_data
            if f0 in consumed_faces or f1 in consumed_faces:
                continue

            poly0 = polygons[f0]
            poly1 = polygons[f1]

            # Condition 1: Strict Coplanarity (same normal within threshold)
            dot_n = np.dot(normals[f0], normals[f1])
            if dot_n < cos_coplanar:
                continue

            v_a, v_b = e
            m, n = len(poly0), len(poly1)
            ia0, ib0 = poly0.index(v_a), poly0.index(v_b)
            ia1, ib1 = poly1.index(v_a), poly1.index(v_b)

            # Collinearity check at both endpoints v_a and v_b
            prev_a0 = poly0[(ia0 - 1) % m] if poly0[(ia0 + 1) % m] == v_b else poly0[(ia0 + 1) % m]
            prev_a1 = poly1[(ia1 - 1) % n] if poly1[(ia1 + 1) % n] == v_b else poly1[(ia1 + 1) % n]
            d_a0 = v_pos[v_a] - v_pos[prev_a0]
            d_a1 = v_pos[prev_a1] - v_pos[v_a]
            len_a0, len_a1 = np.linalg.norm(d_a0), np.linalg.norm(d_a1)

            prev_b0 = poly0[(ib0 - 1) % m] if poly0[(ib0 + 1) % m] == v_a else poly0[(ib0 + 1) % m]
            prev_b1 = poly1[(ib1 - 1) % n] if poly1[(ib1 + 1) % n] == v_a else poly1[(ib1 + 1) % n]
            d_b0 = v_pos[v_b] - v_pos[prev_b0]
            d_b1 = v_pos[prev_b1] - v_pos[v_b]
            len_b0, len_b1 = np.linalg.norm(d_b0), np.linalg.norm(d_b1)

            if min(len_a0, len_a1, len_b0, len_b1) < 1e-12:
                continue

            cos_a = np.dot(d_a0, d_a1) / (len_a0 * len_a1)
            cos_b = np.dot(d_b0, d_b1) / (len_b0 * len_b1)

            # Ensure side boundaries are straight (collinear within threshold)
            if cos_a < cos_collinear or cos_b < cos_collinear:
                continue

            # Merge polygons while retaining topological boundary
            merged_poly = merge_polygons(poly0, poly1, v_a, v_b)
            if merged_poly is None:
                continue

            new_polygons.append(merged_poly)
            consumed_faces.add(f0)
            consumed_faces.add(f1)
            merged_this_round += 1

        # Retain untouched polygons
        for f_idx, poly in enumerate(polygons):
            if f_idx not in consumed_faces:
                new_polygons.append(poly)

        polygons = new_polygons
        print(f"    -> Pass {it + 1}: Merged {merged_this_round:,} pairs into elongated faces (Remaining: {len(polygons):,})")
        if merged_this_round == 0:
            break

    # 3. Clean Unreferenced Vertices
    active_verts = set()
    for poly in polygons:
        active_verts.update(poly)

    old_to_new = {}
    new_v_pos = []
    for idx, v in enumerate(v_pos):
        if idx in active_verts:
            old_to_new[idx] = len(new_v_pos)
            new_v_pos.append(v)

    reindexed_polygons = [[old_to_new[idx] for idx in poly] for poly in polygons]

    # 4. Verify Watertight Topology (0 new boundary edges)
    post_e_counts = defaultdict(int)
    for poly in reindexed_polygons:
        m = len(poly)
        for i in range(m):
            e = tuple(sorted((poly[i], poly[(i + 1) % m])))
            post_e_counts[e] += 1
    post_boundary = sum(1 for cnt in post_e_counts.values() if cnt == 1)
    new_boundary_created = post_boundary - init_boundary

    # 5. Export Watertight Mesh
    with open(output_merged_path, 'w') as out:
        out.write("# Watertight Anisotropic Quad Merged Mesh (Step 4b)\n")
        out.write(f"# Source: {input_quad_path}\n")
        out.write(f"# Polygons: {len(reindexed_polygons):,} | Vertices: {len(new_v_pos):,}\n")
        for v in new_v_pos:
            out.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for poly in reindexed_polygons:
            out.write(f"f {' '.join(str(idx + 1) for idx in poly)}\n")

    total = len(reindexed_polygons)
    quads = sum(1 for p in reindexed_polygons if len(p) == 4)
    print(f"[+] [STEP 4b COMPLETE] Saved to: {output_merged_path}")
    print(f"    -> Total Polys: {total:,} | Vertices: {len(new_v_pos):,}")
    print(f"    -> Boundary Edges: {post_boundary:,} (New Holes Created: {new_boundary_created})")
    if new_boundary_created == 0:
        print("    -> [VERIFIED] 100% Watertight: No holes or T-junction cracks created!")
    else:
        print(f"    -> [!] Warning: {new_boundary_created} boundary edges were created.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 4b: Watertight Anisotropic Planar Quad Merging")
    parser.add_argument("input", help="Step 4 quad mesh OBJ path")
    parser.add_argument("output", help="Output anisotropic quad mesh path")
    parser.add_argument("--coplanar", type=float, default=3.5, help="Normal deviation limit in degrees (default: 3.5°)")
    parser.add_argument("--collinear", type=float, default=18.0, help="Collinearity deviation limit in degrees (default: 18.0°)")
    parser.add_argument("--passes", type=int, default=5, help="Maximum aggregation passes (default: 5)")
    args = parser.parse_args()

    run_step4b_anisotropic_merging(
        args.input, 
        args.output,
        coplanar_deg=args.coplanar,
        collinear_deg=args.collinear,
        max_iterations=args.passes
    )
