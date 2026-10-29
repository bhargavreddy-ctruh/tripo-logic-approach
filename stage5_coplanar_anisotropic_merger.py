import os
import sys
import math
import argparse
import bpy
import bmesh
import numpy as np

def get_opposite_edge(face, edge):
    """Returns the opposite parallel edge in a quad face."""
    if len(face.verts) != 4:
        return None
    other_verts = [v for v in face.verts if v not in edge.verts]
    for e in face.edges:
        if set(e.verts) == set(other_verts):
            return e
    return None

def run_coplanar_anisotropic_merge(input_obj, output_obj, output_glb=None, angle_thresh_deg=3.5, min_loop_len=3):
    print("="*75)
    print(">>> [STAGE 5: COPLANAR ANISOTROPIC QUAD ENGINE]")
    print("="*75)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(filepath=input_obj)

    # Isolate low-poly retopo target
    target_obj = None
    for o in bpy.context.scene.objects:
        if 'Retopo' in o.name:
            target_obj = o
            break
    if target_obj is None:
        target_obj = [o for o in bpy.context.scene.objects if o.type == 'MESH'][0]

    # Remove any extra objects (like outline rails) during retopo coarsening
    for o in list(bpy.context.scene.objects):
        if o != target_obj:
            bpy.data.objects.remove(o, do_unlink=True)

    bpy.context.view_layer.objects.active = target_obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(target_obj.data)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    initial_faces = len(bm.faces)
    initial_verts = len(bm.verts)
    print(f"[*] Input Retopo Mesh: {initial_verts:,} vertices | {initial_faces:,} faces")

    # Initial aspect ratio stats
    init_aspects = []
    for f in bm.faces:
        if len(f.verts) == 4:
            vs = [v.co for v in f.verts]
            e_lens = [(vs[(i+1)%4] - vs[i]).length for i in range(4)]
            l_max = max(e_lens)
            l_min = min(e_lens)
            if l_min > 1e-6:
                init_aspects.append(l_max / l_min)
    if init_aspects:
        print(f"    -> Initial Aspect Ratios: Median={np.median(init_aspects):.2f}, Mean={np.mean(init_aspects):.2f}, Max={np.max(init_aspects):.2f}")
        print(f"    -> Initial Quads > 2:1: {sum(1 for a in init_aspects if a > 2.0)/len(init_aspects)*100:.1f}%")

    angle_thresh = math.radians(angle_thresh_deg)

    # 1. Trace Continuous Coplanar Edge Loops
    print(f"[*] [1/4] Tracing continuous coplanar edge loops (dihedral angle <= {angle_thresh_deg}°)...")
    visited_edges = set()
    edge_loops = []

    for start_edge in bm.edges:
        if start_edge in visited_edges or start_edge.is_boundary or len(start_edge.link_faces) != 2:
            continue
        f1, f2 = start_edge.link_faces
        if len(f1.verts) != 4 or len(f2.verts) != 4:
            continue
        if f1.normal.angle(f2.normal) > angle_thresh:
            continue

        loop = [start_edge]
        visited_edges.add(start_edge)

        for face in [f1, f2]:
            curr_e = start_edge
            curr_f = face
            while True:
                opp_e = get_opposite_edge(curr_f, curr_e)
                if opp_e is None or opp_e in visited_edges or opp_e.is_boundary or len(opp_e.link_faces) != 2:
                    break
                of1, of2 = opp_e.link_faces
                if len(of1.verts) != 4 or len(of2.verts) != 4:
                    break
                if of1.normal.angle(of2.normal) > angle_thresh:
                    break
                if of1.normal.angle(f1.normal) > angle_thresh * 1.5:
                    break

                loop.append(opp_e)
                visited_edges.add(opp_e)
                curr_e = opp_e
                curr_f = of2 if curr_f == of1 else of1

        if len(loop) >= min_loop_len:
            edge_loops.append(loop)

    print(f"    -> Identified {len(edge_loops):,} coplanar edge loops (length >= {min_loop_len})")

    # 2. Select independent alternating loops (preventing adjacent loop collisions)
    print("[*] [2/4] Selecting independent non-adjacent loop set for dissolution...")
    used_faces = set()
    loops_to_dissolve = []
    edges_to_dissolve = []

    for loop in sorted(edge_loops, key=lambda l: len(l), reverse=True):
        loop_faces = set()
        for e in loop:
            loop_faces.update(e.link_faces)
        if not (loop_faces & used_faces):
            loops_to_dissolve.append(loop)
            edges_to_dissolve.extend(loop)
            used_faces.update(loop_faces)

    print(f"    -> Selected {len(loops_to_dissolve):,} independent loops ({len(edges_to_dissolve):,} edges) to dissolve.")

    if edges_to_dissolve:
        # 3. Dissolve Edge Loops
        bmesh.ops.dissolve_edges(bm, edges=edges_to_dissolve)

        # 4. Clean 2-valent collinear vertices along dissolved lines
        bm.verts.ensure_lookup_table()
        verts_to_dissolve = []
        for v in bm.verts:
            if v.is_valid and len(v.link_edges) == 2:
                e1, e2 = v.link_edges
                v_o1 = e1.other_vert(v)
                v_o2 = e2.other_vert(v)
                d1 = (v.co - v_o1.co).normalized()
                d2 = (v_o2.co - v.co).normalized()
                if d1.dot(d2) > 0.92: # Nearly collinear straight line
                    verts_to_dissolve.append(v)

        print(f"[*] [3/4] Dissolving {len(verts_to_dissolve):,} valence-2 collinear vertices...")
        bmesh.ops.dissolve_verts(bm, verts=verts_to_dissolve)

        # 5. Sanitize boundary singularities (triangulate N-gons and re-pair into quads)
        bm.faces.ensure_lookup_table()
        ngons = [f for f in bm.faces if f.is_valid and len(f.verts) > 4]
        if ngons:
            print(f"[*] [4/4] Sanitizing {len(ngons):,} boundary transition N-gons into quads + reduction tris...")
            res = bmesh.ops.triangulate(bm, faces=ngons, quad_method='BEAUTY', ngon_method='BEAUTY')
            tri_faces = [f for f in res['faces'] if f.is_valid and len(f.verts) == 3]
            if tri_faces:
                bmesh.ops.join_triangles(
                    bm,
                    faces=tri_faces,
                    angle_face_threshold=angle_thresh * 2.0,
                    angle_shape_threshold=math.radians(60.0)
                )

    bmesh.update_edit_mesh(target_obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Audit Metrics
    final_polys = len(target_obj.data.polygons)
    degrees = [len(p.vertices) for p in target_obj.data.polygons]
    quads = sum(1 for d in degrees if d == 4)
    tris = sum(1 for d in degrees if d == 3)
    ngons_final = sum(1 for d in degrees if d > 4)

    final_aspects = []
    for p in target_obj.data.polygons:
        if len(p.vertices) == 4:
            vs = [target_obj.data.vertices[i].co for i in p.vertices]
            e_lens = [(vs[(i+1)%4] - vs[i]).length for i in range(4)]
            l_max = max(e_lens)
            l_min = min(e_lens)
            if l_min > 1e-6:
                final_aspects.append(l_max / l_min)

    print("\n" + "="*75)
    print(">>> [STAGE 5 FINAL AUDIT: ANISOTROPIC TOPOLOGY REPORT]")
    print("="*75)
    print(f"[*] Total Faces: {initial_faces:,} -> {final_polys:,} (-{(1 - final_polys/initial_faces)*100:.1f}% reduction)")
    print(f"[*] Quads: {quads:,} ({quads/final_polys*100:.1f}%)")
    print(f"[*] Tris:  {tris:,} ({tris/final_polys*100:.1f}%)")
    print(f"[*] N-gons: {ngons_final} (Strictly 0)")
    if final_aspects:
        print(f"[*] Anisotropic Quad Aspect Ratios:")
        print(f"    -> Median Aspect: {np.median(final_aspects):.2f}:1")
        print(f"    -> Mean Aspect:   {np.mean(final_aspects):.2f}:1")
        print(f"    -> Max Aspect:    {np.max(final_aspects):.2f}:1")
        print(f"    -> Quads > 2:1:   {sum(1 for a in final_aspects if a > 2.0) / len(final_aspects) * 100:.1f}%")
        print(f"    -> Quads > 3:1:   {sum(1 for a in final_aspects if a > 3.0) / len(final_aspects) * 100:.1f}%")
    print("="*75)

    # Export sanitized output OBJ
    bpy.ops.wm.obj_export(filepath=output_obj, export_selected_objects=True, export_uv=False, export_normals=True)
    print(f"[+] Saved Anisotropic Quad OBJ to: {output_obj}")

    # Export GLB companion if requested
    if output_glb:
        bpy.ops.export_scene.gltf(filepath=output_glb, use_selection=True)
        print(f"[+] Saved Anisotropic Quad GLB to: {output_glb}")

if __name__ == "__main__":
    args_list = []
    if "--" in sys.argv:
        args_list = sys.argv[sys.argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description="Stage 5: Coplanar Anisotropic Quad Merger")
    parser.add_argument("input", help="Input watertight retopo OBJ")
    parser.add_argument("output_obj", help="Output anisotropic quad OBJ")
    parser.add_argument("--output_glb", default=None, help="Optional output anisotropic GLB")
    parser.add_argument("--angle", type=float, default=3.5, help="Coplanar angle threshold in degrees (default: 3.5)")
    parser.add_argument("--min_loop", type=int, default=3, help="Minimum loop length (default: 3)")
    args = parser.parse_args(args_list)

    run_coplanar_anisotropic_merge(
        args.input,
        args.output_obj,
        output_glb=args.output_glb,
        angle_thresh_deg=args.angle,
        min_loop_len=args.min_loop
    )

