import os
import sys
import math
import argparse
import bpy
import bmesh
import numpy as np

def run_anisotropic_merge(input_obj_path, output_obj_path, angle_thresh_deg=4.0, passes=2, max_aspect=6.0):
    print("="*70)
    print(">>> [ANISOTROPIC COPLANAR QUAD MERGING ENGINE v2]")
    print("="*70)

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(filepath=input_obj_path)

    # Find the low-poly retopo object
    objs = [o for o in bpy.context.scene.objects if 'Retopo' in o.name]
    if not objs:
        objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    obj = objs[0]
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.mode_set(mode='EDIT')

    bm = bmesh.from_edit_mesh(obj.data)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    initial_faces = len(bm.faces)
    initial_verts = len(bm.verts)
    print(f"[*] Input Mesh: {initial_verts:,} vertices | {initial_faces:,} faces")

    angle_thresh = math.radians(angle_thresh_deg)

    for p in range(1, passes + 1):
        bm.faces.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.verts.ensure_lookup_table()

        # 1. Find coplanar quad edges
        candidate_edges = []
        for e in bm.edges:
            if e.is_boundary or len(e.link_faces) != 2:
                continue
            f1, f2 = e.link_faces
            if len(f1.verts) != 4 or len(f2.verts) != 4:
                continue
            if f1.normal.angle(f2.normal) <= angle_thresh:
                candidate_edges.append(e)

        if not candidate_edges:
            print(f"[*] Pass {p}: No more candidate edges found. Stopping.")
            break

        # 2. Pick independent edge set
        dissolve_edges = set()
        used_faces = set()
        used_verts = set()

        for e in candidate_edges:
            f1, f2 = e.link_faces
            if f1 in used_faces or f2 in used_faces:
                continue
            v1, v2 = e.verts
            if v1 in used_verts or v2 in used_verts:
                continue

            dissolve_edges.add(e)
            used_faces.add(f1)
            used_faces.add(f2)
            used_verts.add(v1)
            used_verts.add(v2)

        print(f"[*] Pass {p}/{passes}: Merging {len(dissolve_edges):,} coplanar quad pairs...")
        if not dissolve_edges:
            break

        bmesh.ops.dissolve_edges(bm, edges=list(dissolve_edges))

        # 3. Clean collinear vertices
        bm.verts.ensure_lookup_table()
        verts_to_dissolve = []
        for v in bm.verts:
            if v.is_valid and len(v.link_edges) == 2:
                e1, e2 = v.link_edges
                v_o1 = e1.other_vert(v)
                v_o2 = e2.other_vert(v)
                d1 = (v.co - v_o1.co).normalized()
                d2 = (v_o2.co - v.co).normalized()
                if d1.dot(d2) > 0.94: # within ~20 degrees of straight line
                    verts_to_dissolve.append(v)

        bmesh.ops.dissolve_verts(bm, verts=verts_to_dissolve)

        # 4. Triangulate remaining N-gons into quads + reduction tris
        bm.faces.ensure_lookup_table()
        ngons = [f for f in bm.faces if f.is_valid and len(f.verts) > 4]
        if ngons:
            res = bmesh.ops.triangulate(bm, faces=ngons, quad_method='BEAUTY', ngon_method='BEAUTY')
            tri_faces = [f for f in res['faces'] if f.is_valid and len(f.verts) == 3]
            if tri_faces:
                bpy.ops.mesh.select_all(action='DESELECT')
                for f in tri_faces:
                    f.select = True
                bpy.ops.mesh.tris_convert_to_quads(face_threshold=angle_thresh, shape_threshold=math.radians(60.0))

    bmesh.update_edit_mesh(obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Audit final topology
    final_polys = len(obj.data.polygons)
    degrees = [len(p.vertices) for p in obj.data.polygons]
    quads = sum(1 for d in degrees if d == 4)
    tris = sum(1 for d in degrees if d == 3)
    ngons_final = sum(1 for d in degrees if d > 4)

    final_aspects = []
    for p in obj.data.polygons:
        if len(p.vertices) == 4:
            vs = [obj.data.vertices[i].co for i in p.vertices]
            e_lens = [(vs[(i+1)%4] - vs[i]).length for i in range(4)]
            l_max = max(e_lens)
            l_min = min(e_lens)
            if l_min > 1e-6:
                final_aspects.append(l_max / l_min)

    print("\n" + "="*70)
    print(">>> [ANISOTROPIC MERGE FINAL AUDIT REPORT]")
    print("="*70)
    print(f"[*] Initial Faces: {initial_faces:,} -> Final Faces: {final_polys:,} (-{(1 - final_polys/initial_faces)*100:.1f}%)")
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
    print("="*70)

    # Export output OBJ
    bpy.ops.wm.obj_export(filepath=output_obj_path, export_selected_objects=True, export_uv=False, export_normals=True)
    print(f"[+] Saved Anisotropic Quad Model to: {output_obj_path}")

if __name__ == "__main__":
    args_list = []
    if "--" in sys.argv:
        args_list = sys.argv[sys.argv.index("--") + 1:]
    parser = argparse.ArgumentParser(description="Anisotropic Coplanar Quad Merger")
    parser.add_argument("input", help="Input OBJ path")
    parser.add_argument("output", help="Output OBJ path")
    parser.add_argument("--angle", type=float, default=4.0, help="Coplanar angle threshold in degrees")
    parser.add_argument("--passes", type=int, default=2, help="Number of coarsening passes")
    args = parser.parse_args(args_list)

    run_anisotropic_merge(args.input, args.output, angle_thresh_deg=args.angle, passes=args.passes)
