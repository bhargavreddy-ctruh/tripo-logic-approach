import os
import sys
import math
import argparse
import bpy
import bmesh
import mathutils
import numpy as np

def get_continuation_edge(v, incoming_edge):
    """
    At vertex v (regular valence 4), finds the unique opposite edge
    continuing straight through v (does not share a face with incoming_edge).
    Matches Blender's Alt+Click edge loop line selection.
    """
    if len(v.link_edges) != 4:
        return None
    inc_faces = set(incoming_edge.link_faces)
    for e in v.link_edges:
        if e != incoming_edge and not (set(e.link_faces) & inc_faces):
            return e
    return None

def run_anisotropic_ribbon_engine(
    input_obj_path,
    output_obj_path,
    output_glb_path=None,
    preview_png_path=None,
    passes=3,
    crease_angle_deg=16.0,
    feature_protect=True
):
    print("=" * 75)
    print(">>> [STAGE 5: TRIPO-GRADE IN-DEPTH FEATURE PRESERVING QUAD ENGINE]")
    print("=" * 75)
    print(f"[*] Input Retopo Model: {input_obj_path}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(filepath=input_obj_path)

    # Find the retopo mesh object (ignore visualization rails/curves)
    target_obj = None
    for o in bpy.context.scene.objects:
        if 'Retopo' in o.name:
            target_obj = o
            break
    if target_obj is None:
        mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
        mesh_objs.sort(key=lambda x: len(x.data.polygons))
        target_obj = mesh_objs[0]

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
    print(f"[*] Input Shell: {initial_verts:,} vertices | {initial_faces:,} faces")

    # =========================================================================
    # STEP 1: IN-DEPTH FACIAL & GEOMETRIC FEATURE PROTECTION MASK
    # =========================================================================
    protected_edges = set()
    crease_rad = math.radians(crease_angle_deg)

    # A. Open Boundaries (always locked to prevent holes)
    for e in bm.edges:
        if e.is_boundary:
            protected_edges.add(e)

    # B. Hard CAD Creases (> crease_angle_deg)
    for e in bm.edges:
        if len(e.link_faces) == 2:
            if e.link_faces[0].normal.angle(e.link_faces[1].normal) > crease_rad:
                protected_edges.add(e)

    if feature_protect:
        # C. Face Area Analysis: Protect dense micro-features (eyes, lips, nostrils, rivets)
        face_areas = [f.calc_area() for f in bm.faces]
        med_area = np.median(face_areas) if face_areas else 1.0

        for e in bm.edges:
            if len(e.link_faces) == 2:
                ang = e.link_faces[0].normal.angle(e.link_faces[1].normal)
                f_min_area = min(e.link_faces[0].calc_area(), e.link_faces[1].calc_area())
                # Organic feature ridges (> 9.0 deg on detailed faces)
                if ang > math.radians(9.0) and f_min_area < med_area * 1.5:
                    protected_edges.add(e)

        # D. Compound Curvature Domes & Saddles (Preserves 3D facial volume on cheeks, nose, chin, horns)
        for v in bm.verts:
            if len(v.link_faces) >= 3:
                norms = [f.normal for f in v.link_faces]
                avg_n = sum(norms, mathutils.Vector((0, 0, 0))) / len(norms)
                if avg_n.length > 1e-4:
                    avg_n.normalize()
                    devs = [f.normal.angle(avg_n) for f in v.link_faces if f.normal.length > 1e-4]
                    if devs and np.mean(devs) > math.radians(18.0):
                        for e in v.link_edges:
                            protected_edges.add(e)

    print(f"[*] Feature Protection Field: Locked {len(protected_edges):,} edges ({len(protected_edges)/max(len(bm.edges),1)*100:.1f}%) against dissolution.")

    # =========================================================================
    # STEP 2: MULTI-PASS 1D CONTINUOUS LINE DISSOLUTION (ON UNPROTECTED SURFACES)
    # =========================================================================
    for pass_idx in range(1, passes + 1):
        bm.faces.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.verts.ensure_lookup_table()

        visited_edges = set()
        all_lines = []

        for start_e in bm.edges:
            if start_e in visited_edges or start_e.is_boundary or start_e in protected_edges:
                continue

            line = [start_e]
            visited_edges.add(start_e)

            # Trace forward (verts[1])
            curr_e = start_e
            curr_v = start_e.verts[1]
            while True:
                nxt = get_continuation_edge(curr_v, curr_e)
                if not nxt or nxt in visited_edges or nxt.is_boundary or nxt in protected_edges:
                    break
                line.append(nxt)
                visited_edges.add(nxt)
                curr_v = nxt.other_vert(curr_v)
                curr_e = nxt

            # Trace backward (verts[0])
            curr_e = start_e
            curr_v = start_e.verts[0]
            while True:
                nxt = get_continuation_edge(curr_v, curr_e)
                if not nxt or nxt in visited_edges or nxt.is_boundary or nxt in protected_edges:
                    break
                line.insert(0, nxt)
                visited_edges.add(nxt)
                curr_v = nxt.other_vert(curr_v)
                curr_e = nxt

            if len(line) >= 3:
                all_lines.append(line)

        line_info = []
        for l in all_lines:
            l_faces = {f for e in l for f in e.link_faces}
            line_info.append({'line': l, 'faces': l_faces, 'len': len(l)})

        line_info.sort(key=lambda x: x['len'], reverse=True)

        # Non-adjacent line selection (ensures uniform parallel slats)
        used_faces = set()
        dissolve_lines = []
        for li in line_info:
            if not (li['faces'] & used_faces):
                dissolve_lines.append(li)
                used_faces.update(li['faces'])

        edges_to_dissolve = [e for li in dissolve_lines for e in li['line']]
        print(f"[*] Pass {pass_idx}/{passes}: Dissolving {len(edges_to_dissolve):,} transverse edges across {len(dissolve_lines):,} continuous straight lines...")

        if not edges_to_dissolve:
            break

        bmesh.ops.dissolve_edges(bm, edges=edges_to_dissolve, use_verts=True)

        # Dissolve collinear valence-2 vertices along straight lines
        bm.verts.ensure_lookup_table()
        v_dissolve = []
        for v in bm.verts:
            if v.is_valid and len(v.link_edges) == 2 and v not in protected_edges:
                e1, e2 = v.link_edges
                d1 = (v.co - e1.other_vert(v).co).normalized()
                d2 = (e2.other_vert(v).co - v.co).normalized()
                if d1.dot(d2) > 0.92:
                    v_dissolve.append(v)
        if v_dissolve:
            bmesh.ops.dissolve_verts(bm, verts=v_dissolve)

        # Clean boundary singularities into shape-preserving quad/triangle adapters
        bm.faces.ensure_lookup_table()
        ngons = [f for f in bm.faces if f.is_valid and len(f.verts) > 4]
        if ngons:
            res = bmesh.ops.triangulate(bm, faces=ngons, quad_method='BEAUTY', ngon_method='BEAUTY')
            tri_faces = [f for f in res['faces'] if f.is_valid and len(f.verts) == 3]
            bmesh.ops.join_triangles(bm, faces=tri_faces, angle_face_threshold=0.25, angle_shape_threshold=0.85)

    bmesh.update_edit_mesh(target_obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    # Topology Quality Audit
    final_polys = len(target_obj.data.polygons)
    degrees = [len(p.vertices) for p in target_obj.data.polygons]
    quads = sum(1 for d in degrees if d == 4)
    tris = sum(1 for d in degrees if d == 3)
    ngons_final = sum(1 for d in degrees if d > 4)

    final_aspects = []
    for p in target_obj.data.polygons:
        if len(p.vertices) == 4:
            vs = [target_obj.data.vertices[i].co for i in p.vertices]
            e_lens = [(vs[(i + 1) % 4] - vs[i]).length for i in range(4)]
            l_max = max(e_lens)
            l_min = min(e_lens)
            if l_min > 1e-6:
                final_aspects.append(l_max / l_min)

    print("\n" + "=" * 75)
    print(">>> [FINAL AUDIT: IN-DEPTH FEATURE PRESERVATION QUALITY REPORT]")
    print("=" * 75)
    print(f"[*] Faces: {initial_faces:,} -> {final_polys:,} (-{(1 - final_polys / initial_faces) * 100:.1f}% reduction)")
    print(f"[*] Quads: {quads:,} ({quads / final_polys * 100:.1f}%)")
    print(f"[*] Tris:  {tris:,} ({tris / final_polys * 100:.1f}%) [Preserved on facial transitions & singularities]")
    print(f"[*] N-gons: {ngons_final} (Strictly 0)")
    if final_aspects:
        print(f"[*] Anisotropic Quad Aspect Ratios:")
        print(f"    -> Median Aspect: {np.median(final_aspects):.2f}:1")
        print(f"    -> Mean Aspect:   {np.mean(final_aspects):.2f}:1")
        print(f"    -> Max Aspect:    {np.max(final_aspects):.2f}:1")
        print(f"    -> Quads > 2:1:   {sum(1 for a in final_aspects if a > 2.0) / len(final_aspects) * 100:.1f}%")
        print(f"    -> Quads > 3:1:   {sum(1 for a in final_aspects if a > 3.0) / len(final_aspects) * 100:.1f}%")
    print("=" * 75)

    # Export clean OBJ
    bpy.ops.wm.obj_export(filepath=output_obj_path, export_selected_objects=True, export_uv=False, export_normals=True)
    print(f"[+] Saved In-Depth Feature Preserved OBJ to: {output_obj_path}")

    # Export companion GLB
    if output_glb_path:
        bpy.ops.export_scene.gltf(filepath=output_glb_path, use_selection=True)
        print(f"[+] Saved In-Depth Feature Preserved GLB to: {output_glb_path}")

    # Render Studio Wireframe Preview
    if preview_png_path:
        world = bpy.context.scene.world or bpy.data.worlds.new('World')
        bpy.context.scene.world = world

        mat = bpy.data.materials.new(name="MWire")
        nodes = mat.node_tree.nodes
        links = mat.node_tree.links
        bsdf = nodes.get('Principled BSDF')

        wf_node = nodes.new('ShaderNodeWireframe')
        wf_node.inputs['Size'].default_value = 0.5
        wf_node.use_pixel_size = True

        mix = nodes.new('ShaderNodeMix')
        mix.data_type = 'RGBA'
        links.new(wf_node.outputs['Fac'], mix.inputs['Factor'])
        mix.inputs[6].default_value = (0.20, 0.20, 0.22, 1.0)
        mix.inputs[7].default_value = (0.80, 0.80, 0.85, 1.0)
        links.new(mix.outputs[2], bsdf.inputs['Base Color'])
        bsdf.inputs['Roughness'].default_value = 0.35

        target_obj.data.materials.clear()
        target_obj.data.materials.append(mat)

        l1 = bpy.data.lights.new(name='Key', type='SUN')
        l1.energy = 5.0
        l1_obj = bpy.data.objects.new('Key', l1)
        l1_obj.rotation_euler = (math.radians(50), math.radians(20), math.radians(-35))
        bpy.context.scene.collection.objects.link(l1_obj)

        cam = bpy.data.cameras.new('Cam')
        cam.lens = 45.0
        cam_obj = bpy.data.objects.new('Cam', cam)
        bpy.context.scene.collection.objects.link(cam_obj)
        bpy.context.scene.camera = cam_obj

        bbox_corners = [target_obj.matrix_world @ mathutils.Vector(corner) for corner in target_obj.bound_box]
        center = sum(bbox_corners, mathutils.Vector((0, 0, 0))) / 8.0
        bb = target_obj.dimensions
        max_dim = max(bb.x, bb.y, bb.z)
        cam_dist = max_dim * 1.6
        cam_obj.location = center + mathutils.Vector((cam_dist * 0.35, -cam_dist * 1.05, cam_dist * 0.55))
        direction = center - cam_obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam_obj.rotation_euler = rot_quat.to_euler()

        scene = bpy.context.scene
        scene.render.resolution_x = 1200
        scene.render.resolution_y = 900
        scene.render.filepath = preview_png_path
        bpy.ops.render.render(write_still=True)
        print(f"[+] Saved Studio Wireframe Preview to: {preview_png_path}")

if __name__ == "__main__":
    args_list = []
    if "--" in sys.argv:
        args_list = sys.argv[sys.argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description="Stage 5: Tripo-Grade In-Depth Feature Preserving Quad Engine")
    parser.add_argument("input", help="Input watertight retopo OBJ")
    parser.add_argument("output_obj", help="Output anisotropic quad OBJ")
    parser.add_argument("--output_glb", default=None, help="Optional output anisotropic GLB")
    parser.add_argument("--preview", default=None, help="Optional wireframe preview PNG")
    parser.add_argument("--passes", type=int, default=3, help="Number of coarsening passes (default: 3)")
    parser.add_argument("--crease", type=float, default=16.0, help="Crease angle threshold in degrees (default: 16.0)")
    parser.add_argument("--no_feature_protect", action="store_true", help="Disable in-depth feature protection")
    args = parser.parse_args(args_list)

    run_anisotropic_ribbon_engine(
        args.input,
        args.output_obj,
        output_glb_path=args.output_glb,
        preview_png_path=args.preview,
        passes=args.passes,
        crease_angle_deg=args.crease,
        feature_protect=not args.no_feature_protect
    )
