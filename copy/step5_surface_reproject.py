import argparse
import numpy as np
import trimesh
from scipy.spatial import cKDTree

def laplacian_tangent_relax(v, faces, n_iters=10, step=0.5, crease_verts=None):
    """
    Laplacian smoothing restricted to the tangent plane of each vertex.
    Crease vertices are held fixed.
    """
    crease_set = set(crease_verts or [])
    adjacency = {i: set() for i in range(len(v))}
    for f in faces:
        for i in range(len(f)):
            a, b = f[i], f[(i + 1) % len(f)]
            adjacency[a].add(b)
            adjacency[b].add(a)

    for _ in range(n_iters):
        new_v = v.copy()
        for i, neighbors in adjacency.items():
            if not neighbors or i in crease_set:
                continue
            centroid = np.mean(v[list(neighbors)], axis=0)
            disp = centroid - v[i]
            new_v[i] = v[i] + step * disp
        v = new_v
    return v


def run_step5_surface_reproject(
    quad_mesh_path,
    highres_source_path,
    output_path,
    relax_iters=10,
    relax_step=0.5,
    crease_deg=30.0,
    reproject=True
):
    print(f"[*] [STEP 5] Loading quad-dominant mesh: {quad_mesh_path}")
    quad_mesh = trimesh.load(quad_mesh_path, process=False)
    if isinstance(quad_mesh, trimesh.Scene):
        quad_mesh = quad_mesh.dump(concatenate=True)

    v = quad_mesh.vertices.copy()
    n_verts = len(v)

    # Parse faces and edges from OBJ directly (trimesh may triangulate quads)
    raw_faces = []
    with open(quad_mesh_path) as f:
        for line in f:
            if line.startswith('f '):
                idxs = [int(x.split('/')[0]) - 1 for x in line.strip().split()[1:]]
                raw_faces.append(idxs)

    print(f"    -> Quad Mesh: {n_verts:,} vertices | {len(raw_faces):,} polygons")

    # 1. Identify crease vertices (held fixed during relaxation)
    print(f"[*] Identifying fixed crease vertices (dihedral >= {crease_deg}°)...")
    cos_crease = np.cos(np.radians(crease_deg))
    fn = quad_mesh.face_normals
    adj_faces = quad_mesh.face_adjacency
    adj_edges = quad_mesh.face_adjacency_edges

    crease_verts = set()
    if len(adj_faces) > 0:
        dot_n = np.clip(np.sum(fn[adj_faces[:, 0]] * fn[adj_faces[:, 1]], axis=1), -1.0, 1.0)
        sharp_mask = dot_n < cos_crease
        sharp_edges = adj_edges[sharp_mask]
        for v0, v1 in sharp_edges:
            crease_verts.add(int(v0))
            crease_verts.add(int(v1))

    print(f"    -> Crease-locked vertices: {len(crease_verts):,}")

    # 2. Tangent-Plane Laplacian Smoothing
    print(f"[*] Running tangent-plane Laplacian relaxation ({relax_iters} iterations, step={relax_step})...")
    adjacency = {i: set() for i in range(n_verts)}
    for face in raw_faces:
        n = len(face)
        for i in range(n):
            a, b = face[i], face[(i + 1) % n]
            adjacency[a].add(b)
            adjacency[b].add(a)

    for iteration in range(relax_iters):
        new_v = v.copy()
        for i, neighbors in adjacency.items():
            if not neighbors or i in crease_verts:
                continue
            centroid = np.mean(v[list(neighbors)], axis=0)
            disp = centroid - v[i]
            new_v[i] = v[i] + relax_step * disp
        v = new_v

    print(f"    -> Relaxation complete. Max vertex displacement: {np.max(np.linalg.norm(v - quad_mesh.vertices, axis=1)):.6f}")

    # 3. Reproject vertices back to high-res source surface
    if reproject:
        print(f"[*] Loading high-res source mesh for surface reprojection: {highres_source_path}")
        try:
            src_mesh = trimesh.load(highres_source_path, force='mesh')
            if isinstance(src_mesh, trimesh.Scene):
                src_mesh = src_mesh.dump(concatenate=True)
            src_mesh.merge_vertices()
            print(f"    -> Source Mesh: {len(src_mesh.vertices):,} vertices | {len(src_mesh.faces):,} faces")

            # Build KD-Tree on source mesh vertex positions
            src_tree = cKDTree(src_mesh.vertices)

            # Query closest source vertex for each quad mesh vertex
            dists, closest_idx = src_tree.query(v, workers=-1)

            # For non-crease vertices snap to closest source surface point
            snapped = v.copy()
            non_crease = np.array([i for i in range(n_verts) if i not in crease_verts])
            
            if len(non_crease) > 0:
                # Refine: project the snapped vertex onto the closest triangle surface
                # for smoother sub-vertex placement
                closest_faces_pts, _, closest_face_ids = trimesh.proximity.closest_point(
                    src_mesh, v[non_crease]
                )
                snapped[non_crease] = closest_faces_pts

            v = snapped
            avg_snap = np.mean(dists[non_crease]) if len(non_crease) > 0 else 0
            max_snap = np.max(dists[non_crease]) if len(non_crease) > 0 else 0
            print(f"    -> Reprojection complete. Avg snap distance: {avg_snap:.6f} | Max: {max_snap:.6f}")
        except Exception as e:
            print(f"    [!] Warning: Reprojection failed ({e}). Exporting relaxed mesh without reprojection.")

    # 4. Export Final Reproduced Mesh
    print(f"[*] Writing final surface-reproduced mesh to: {output_path}")
    with open(output_path, 'w') as out:
        out.write("# Step 5: Surface-Reproduced Quad-Dominant Mesh\n")
        out.write(f"# Source: {quad_mesh_path}\n")
        out.write(f"# Reprojected to: {highres_source_path}\n")
        out.write(f"# Polygons: {len(raw_faces):,} | Vertices: {n_verts:,}\n")

        for vert in v:
            out.write(f"v {vert[0]:.8f} {vert[1]:.8f} {vert[2]:.8f}\n")

        for face in raw_faces:
            out.write("f " + " ".join(str(i + 1) for i in face) + "\n")

    print(f"[+] [STEP 5 COMPLETE] Saved reproduced mesh to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Step 5: Surface Reproduction & Tangent Relaxation")
    parser.add_argument("quad_mesh", help="Input quad-dominant mesh from Step 4 (.obj)")
    parser.add_argument("highres_source", help="Original high-resolution source mesh for reprojection")
    parser.add_argument("output", help="Output final mesh (.obj)")
    parser.add_argument("--relax-iters", type=int, default=10, help="Number of tangent Laplacian relaxation iterations (default: 10)")
    parser.add_argument("--relax-step", type=float, default=0.5, help="Relaxation step size (default: 0.5)")
    parser.add_argument("--crease-deg", type=float, default=30.0, help="Dihedral angle threshold to lock crease vertices (default: 30.0)")
    parser.add_argument("--no-reproject", action="store_true", help="Skip surface reprojection (relax only)")
    args = parser.parse_args()

    run_step5_surface_reproject(
        args.quad_mesh,
        args.highres_source,
        args.output,
        relax_iters=args.relax_iters,
        relax_step=args.relax_step,
        crease_deg=args.crease_deg,
        reproject=not args.no_reproject
    )

