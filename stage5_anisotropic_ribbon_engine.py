import os
import sys
import math
import argparse
import bpy
import bmesh
import mathutils
import numpy as np

# ============================================================================
# CAD SPLIT NORMALS - Razor-sharp edges, no mounded gradients
# ============================================================================
def compute_cad_split_normals(mesh, crease_rad=math.radians(18.0)):
    """
    Computes explicit CAD-grade split loop normals at sharp creases (> crease_rad)
    and sets them directly onto the mesh so WebGL/GLTF/OBJ renders razor-sharp
    edges without mounding or muddy gradients.
    """
    bm = bmesh.new()
    bm.from_mesh(mesh)
    bm.faces.ensure_lookup_table()
    bm.edges.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    sharp_edges = set()
    for e in bm.edges:
        if e.is_boundary:
            sharp_edges.add(e)
        elif len(e.link_faces) == 2:
            n0 = e.link_faces[0].normal
            n1 = e.link_faces[1].normal
            if n0.length > 1e-4 and n1.length > 1e-4:
                if n0.angle(n1) > crease_rad:
                    sharp_edges.add(e)
        else:
            sharp_edges.add(e)

    loop_normals = [mathutils.Vector((0, 0, 1))] * len(mesh.loops)

    for v in bm.verts:
        if not v.link_faces:
            continue
        unvisited = set(v.link_faces)
        clusters = []
        while unvisited:
            f_start = unvisited.pop()
            cluster = [f_start]
            frontier = [f_start]
            while frontier:
                curr_f = frontier.pop()
                for e in curr_f.edges:
                    if e in sharp_edges or e not in v.link_edges:
                        continue
                    for adj_f in e.link_faces:
                        if adj_f in unvisited:
                            unvisited.remove(adj_f)
                            cluster.append(adj_f)
                            frontier.append(adj_f)
            clusters.append(cluster)

        for cluster in clusters:
            avg_n = mathutils.Vector((0, 0, 0))
            for f in cluster:
                for l in f.loops:
                    if l.vert == v:
                        e_prev = l.link_loop_prev.edge
                        e_next = l.edge
                        v_prev = e_prev.other_vert(v).co - v.co
                        v_next = e_next.other_vert(v).co - v.co
                        if v_prev.length > 1e-6 and v_next.length > 1e-6:
                            ang = v_prev.angle(v_next)
                            avg_n += f.normal * ang
                        else:
                            avg_n += f.normal
            if avg_n.length > 1e-6:
                avg_n.normalize()
            else:
                avg_n = cluster[0].normal

            for f in cluster:
                for l in f.loops:
                    if l.vert == v:
                        loop_normals[l.index] = avg_n

    bm.free()
    mesh.normals_split_custom_set(loop_normals)


# ============================================================================
# FULL-RAIL TRANSVERSE LOOP TRACER
# ============================================================================
def trace_full_transverse_loop(start_edge, flat_edges_set, protected_edges_set):
    """
    Traces a COMPLETE transverse loop from start_edge in both directions.

    Handles BOTH regular (valence-4) and irregular (valence-3,5,6) vertices:
      - At valence-4: use the unique "opposite" edge (standard Alt+Click)
      - At valence-3/5/6: use best-direction continuation — find the flat edge
        that is most collinear with the incoming direction AND shares no face
        with the incoming edge. This allows loops to cross singularities.

    A valid rail must span from one protected/boundary edge to another, OR
    be a closed ring (cylinder), OR be long enough (>= 4 edges) to be useful.
    """
    def get_best_continuation(v, incoming_e):
        """
        At any valence vertex, find the best-matching continuation edge.
        For valence-4: strict opposite (topological). 
        For valence-3/5/6: best-direction (geometric — most collinear flat edge).
        """
        valence = len(v.link_edges)

        # Standard topological opposite for valence-4
        if valence == 4:
            inc_faces = set(incoming_e.link_faces)
            for e in v.link_edges:
                if e != incoming_e and not (set(e.link_faces) & inc_faces):
                    return e
            return None

        # For boundary/too-low valence: stop — genuine boundary
        if valence <= 2 or v.is_boundary:
            return None

        # For irregular valence (3, 5, 6+): best-direction geometric continuation
        # Direction we're travelling
        incoming_dir = (v.co - incoming_e.other_vert(v).co)
        if incoming_dir.length < 1e-8:
            return None
        incoming_dir = incoming_dir.normalized()

        inc_faces = set(incoming_e.link_faces)
        best_e = None
        best_dot = 0.70  # Minimum collinearity threshold (allows up to ~45° turns)

        for e in v.link_edges:
            if e == incoming_e:
                continue
            if e in protected_edges_set:
                continue
            if e not in flat_edges_set:
                continue
            # Must not share a face with incoming (avoids doubling back)
            if set(e.link_faces) & inc_faces:
                continue
            e_dir = (e.other_vert(v).co - v.co)
            if e_dir.length < 1e-8:
                continue
            e_dir = e_dir.normalized()
            dot = incoming_dir.dot(e_dir)
            if dot > best_dot:
                best_dot = dot
                best_e = e

        return best_e

    def trace_direction(start_e, start_v):
        edges = []
        curr_e = start_e
        curr_v = start_v
        visited = {start_e}
        while True:
            # Boundary vertex: loop naturally ends here
            if curr_v.is_boundary or len(curr_v.link_edges) <= 2:
                return edges, True

            cont = get_best_continuation(curr_v, curr_e)

            if cont is None:
                # No valid continuation — check if this is a natural terminus
                # (protected edge blocks it = valid boundary hit)
                is_at_boundary = (
                    any(e in protected_edges_set for e in curr_v.link_edges) or
                    len(curr_v.link_edges) < 4
                )
                return edges, is_at_boundary

            if cont in protected_edges_set:
                return edges, True   # Hit a crease/boundary rail — valid terminus

            if cont not in flat_edges_set:
                # Continuation leaves the flat zone — hit curvature boundary
                return edges, len(curr_v.link_edges) < 4 or curr_v.is_boundary

            if cont in visited:
                return edges, True   # Closed ring — valid

            visited.add(cont)
            edges.append(cont)
            curr_v = cont.other_vert(curr_v)
            curr_e = cont

    back_edges, back_hit = trace_direction(start_edge, start_edge.verts[0])
    fwd_edges, fwd_hit = trace_direction(start_edge, start_edge.verts[1])

    full_loop = list(reversed(back_edges)) + [start_edge] + fwd_edges

    is_closed_ring = (
        len(back_edges) > 0 and len(fwd_edges) > 0 and
        len(back_edges) + len(fwd_edges) > 2 and
        back_edges[-1] == fwd_edges[-1]
    )

    # STRICT BOUNDARY-TO-BOUNDARY VALIDATION (Zero-tolerance for zig-zags):
    # The loop MUST be a complete rail spanning from a true feature boundary to another
    # feature boundary on both ends, OR be a closed ring (e.g. cylindrical loop).
    # NEVER dissolve a partial loop that terminates mid-patch — partial dissolution
    # creates T-junctions, N-gons, and diagonal zig-zag triangulation lines!
    if not ((back_hit and fwd_hit) or is_closed_ring):
        return None

    if len(full_loop) < 2:
        return None

    return full_loop


# ============================================================================
# DIAMOND DART ADAPTER
# ============================================================================
def apply_diamond_dart_adapters(bm, crease_rad):
    """
    Merges coplanar symmetric triangle pairs into clean diamond dart quads
    at density transitions.
    """
    bm.faces.ensure_lookup_table()
    tris = [f for f in bm.faces if f.is_valid and len(f.verts) == 3]
    if not tris:
        return 0
    bmesh.ops.join_triangles(
        bm, faces=tris,
        angle_face_threshold=math.radians(15.0),
        angle_shape_threshold=math.radians(85.0)
    )
    return len(tris)


# ============================================================================
# MAIN ENGINE
# ============================================================================
def run_anisotropic_ribbon_engine(
    input_obj_path,
    output_obj_path,
    cad_ref_path=None,
    output_glb_path=None,
    preview_png_path=None,
    passes=4,
    crease_angle_deg=18.0,
    feature_protect=True
):
    print("=" * 75)
    print(">>> [STAGE 5: TRIPO-GRADE FULL-RAIL ANISOTROPIC RIBBON ENGINE v3]")
    print("=" * 75)
    print(f"[*] Input: {input_obj_path}")

    bpy.ops.wm.read_factory_settings(use_empty=True)
    bpy.ops.wm.obj_import(filepath=input_obj_path, forward_axis='NEGATIVE_Z', up_axis='Y')

    target_obj = None
    for o in bpy.context.scene.objects:
        if 'Retopo' in o.name:
            target_obj = o
            break
    if target_obj is None:
        mesh_objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
        mesh_objs.sort(key=lambda x: len(x.data.polygons), reverse=True)
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
    print(f"[*] Input: {initial_verts:,} verts | {initial_faces:,} faces")

    # =========================================================================
    # STEP 0: WATERTIGHT HOLE SEALING
    # =========================================================================
    diag = (target_obj.dimensions.x**2 + target_obj.dimensions.y**2 + target_obj.dimensions.z**2)**0.5

    init_b_edges = sum(1 for e in bm.edges if e.is_boundary)
    if init_b_edges > 0:
        print(f"[*] Sealing {init_b_edges:,} open boundary edges...")
        b_edges = [e for e in bm.edges if e.is_boundary]
        try:
            bmesh.ops.holes_fill(bm, edges=b_edges, sides=0)
        except Exception:
            pass
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()

        rem_b_edges = [e for e in bm.edges if e.is_valid and e.is_boundary]
        if rem_b_edges:
            visited_e = set()
            for start_e in rem_b_edges:
                if start_e in visited_e:
                    continue
                loop_edges = [start_e]
                visited_e.add(start_e)
                curr_v = start_e.verts[1]
                while True:
                    nxt = None
                    for cand in curr_v.link_edges:
                        if cand.is_boundary and cand not in visited_e:
                            nxt = cand
                            break
                    if nxt:
                        visited_e.add(nxt)
                        loop_edges.append(nxt)
                        curr_v = nxt.other_vert(curr_v)
                    else:
                        break
                v = loop_edges[0].verts[0]
                c_verts = [v]
                for e in loop_edges:
                    v = e.other_vert(v)
                    if v != c_verts[0]:
                        c_verts.append(v)
                if len(c_verts) >= 3:
                    seen_v = {}
                    split_done = False
                    for idx, vert in enumerate(c_verts):
                        if vert in seen_v:
                            sub_c = c_verts[seen_v[vert]:idx]
                            rem_c = c_verts[:seen_v[vert]] + c_verts[idx:]
                            if len(sub_c) >= 3:
                                try:
                                    bm.faces.new(sub_c)
                                except Exception:
                                    pass
                            if len(rem_c) >= 3:
                                try:
                                    bm.faces.new(rem_c)
                                except Exception:
                                    pass
                            split_done = True
                            break
                        seen_v[vert] = idx
                    if not split_done:
                        try:
                            bm.faces.new(c_verts)
                        except Exception:
                            try:
                                bm.faces.new(list(reversed(c_verts)))
                            except Exception:
                                pass

        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        ngons = [f for f in bm.faces if f.is_valid and len(f.verts) > 4]
        if ngons:
            res = bmesh.ops.triangulate(bm, faces=ngons, quad_method='BEAUTY', ngon_method='BEAUTY')
            tri_faces = [f for f in res['faces'] if f.is_valid and len(f.verts) == 3]
            bmesh.ops.join_triangles(bm, faces=tri_faces,
                                     angle_face_threshold=math.radians(12.0),
                                     angle_shape_threshold=math.radians(60.0))
        for e in list(bm.edges):
            if e.is_valid and len(e.link_faces) > 2:
                faces = list(e.link_faces)
                for f in faces[2:]:
                    if f.is_valid:
                        bm.faces.remove(f)
        bm.edges.ensure_lookup_table()
        bm.faces.ensure_lookup_table()
        bm.verts.ensure_lookup_table()
        sealed = sum(1 for e in bm.edges if e.is_boundary)
        print(f"[+] Sealed: {init_b_edges} -> {sealed} boundary edges")

    # =========================================================================
    # STEP 1: FEATURE PROTECTION MASK
    # =========================================================================
    protected_edges = set()
    crease_rad = math.radians(crease_angle_deg)

    for e in bm.edges:
        if e.is_boundary:
            protected_edges.add(e)

    for e in bm.edges:
        if len(e.link_faces) == 2:
            n0 = e.link_faces[0].normal
            n1 = e.link_faces[1].normal
            if n0.length > 1e-4 and n1.length > 1e-4:
                if n0.angle(n1) > crease_rad:
                    protected_edges.add(e)

    if feature_protect:
        face_areas = [f.calc_area() for f in bm.faces]
        med_area = np.median(face_areas) if face_areas else 1.0

        # Micro-features only: BOTH curved AND tiny - never protect flat panels
        for e in bm.edges:
            if e in protected_edges:
                continue
            if len(e.link_faces) == 2:
                n0 = e.link_faces[0].normal
                n1 = e.link_faces[1].normal
                if n0.length > 1e-4 and n1.length > 1e-4:
                    ang = n0.angle(n1)
                    f_min_area = min(e.link_faces[0].calc_area(), e.link_faces[1].calc_area())
                    if ang > math.radians(12.0) and f_min_area < med_area * 0.15:
                        protected_edges.add(e)

        # High compound curvature vertices (domes/saddles)
        for v in bm.verts:
            if len(v.link_faces) >= 4:
                norms = [f.normal for f in v.link_faces if f.normal.length > 1e-4]
                if len(norms) >= 3:
                    avg_n = sum(norms, mathutils.Vector((0, 0, 0))) / len(norms)
                    if avg_n.length > 1e-4:
                        avg_n.normalize()
                        devs = [f.normal.angle(avg_n) for f in v.link_faces
                                if f.normal.length > 1e-4]
                        if devs and np.mean(devs) > math.radians(28.0):
                            for e in v.link_edges:
                                protected_edges.add(e)

    pct = len(protected_edges) / max(len(bm.edges), 1) * 100.0
    print(f"[*] Protection: {len(protected_edges):,} edges ({pct:.1f}%) locked. Flat panels open.")

    # =========================================================================
    # STEP 2: MULTI-PASS FULL-RAIL TRANSVERSE DISSOLUTION
    # =========================================================================
    # coplanar_rad: flat edge threshold — 12° captures flat CAD panels with
    # normal estimation noise from Instant Meshes (adds ~3-7° on flat panels)
    # without crossing the 18° crease boundary
    coplanar_rad = math.radians(12.0)
    total_dissolved = 0

    for pass_idx in range(1, passes + 1):
        bm.faces.ensure_lookup_table()
        bm.edges.ensure_lookup_table()
        bm.verts.ensure_lookup_table()

        flat_edges = set()
        for e in bm.edges:
            if e in protected_edges or e.is_boundary or len(e.link_faces) != 2:
                continue
            n0 = e.link_faces[0].normal
            n1 = e.link_faces[1].normal
            if n0.length > 1e-4 and n1.length > 1e-4:
                if n0.angle(n1) < coplanar_rad:
                    flat_edges.add(e)

        print(f"[*] Pass {pass_idx}/{passes}: {len(flat_edges):,} flat candidate edges")

        visited_edges = set()
        all_loops = []

        for start_e in flat_edges:
            if start_e in visited_edges:
                continue
            loop = trace_full_transverse_loop(start_e, flat_edges, protected_edges)
            if loop is None:
                continue
            for e in loop:
                visited_edges.add(e)
            all_loops.append(loop)

        print(f"    -> Traced {len(all_loops):,} complete transverse rails")

        if not all_loops:
            print(f"    -> No valid rails. Stopping early.")
            break

        # Greedy non-adjacent selection (longest rails first)
        loop_face_sets = []
        for loop in all_loops:
            fs = set()
            for e in loop:
                for f in e.link_faces:
                    fs.add(f)
            loop_face_sets.append(fs)

        order = sorted(range(len(all_loops)), key=lambda i: len(all_loops[i]), reverse=True)
        used_faces = set()
        dissolve_loops = []
        for i in order:
            if not (loop_face_sets[i] & used_faces):
                dissolve_loops.append(all_loops[i])
                used_faces.update(loop_face_sets[i])

        # Flatten into single batch dissolve list
        edges_to_dissolve = []
        seen_e = set()
        for loop in dissolve_loops:
            for e in loop:
                if e.is_valid and e not in seen_e and e not in protected_edges:
                    edges_to_dissolve.append(e)
                    seen_e.add(e)

        if not edges_to_dissolve:
            print(f"    -> Nothing to dissolve after selection. Stopping.")
            break

        print(f"    -> Dissolving {len(edges_to_dissolve):,} edges across {len(dissolve_loops):,} rails...")
        bmesh.ops.dissolve_edges(bm, edges=edges_to_dissolve, use_verts=False)
        total_dissolved += len(edges_to_dissolve)

        # Dissolve collinear valence-2 vertices (straightens backbone)
        bm.verts.ensure_lookup_table()
        v_dissolve = []
        for v in bm.verts:
            if v.is_valid and len(v.link_edges) == 2 and not v.is_boundary:
                e1, e2 = v.link_edges
                d1 = (v.co - e1.other_vert(v).co).normalized()
                d2 = (e2.other_vert(v).co - v.co).normalized()
                if d1.dot(d2) > 0.90:
                    v_dissolve.append(v)
        if v_dissolve:
            bmesh.ops.dissolve_verts(bm, verts=v_dissolve)
            print(f"    -> Dissolved {len(v_dissolve):,} collinear verts")

        # N-gon cleanup from junctions
        bm.faces.ensure_lookup_table()
        ngons = [f for f in bm.faces if f.is_valid and len(f.verts) > 4]
        if ngons:
            res = bmesh.ops.triangulate(bm, faces=ngons, quad_method='BEAUTY', ngon_method='BEAUTY')
            tri_faces = [f for f in res['faces'] if f.is_valid and len(f.verts) == 3]
            if tri_faces:
                bmesh.ops.join_triangles(bm, faces=tri_faces,
                                         angle_face_threshold=math.radians(16.0),
                                         angle_shape_threshold=math.radians(85.0))

        # Coplanar zig-zag elimination per-pass
        bm.faces.ensure_lookup_table()
        pass_tris = [f for f in bm.faces if f.is_valid and len(f.verts) == 3]
        if pass_tris:
            bmesh.ops.join_triangles(bm, faces=pass_tris,
                                     angle_face_threshold=math.radians(16.0),
                                     angle_shape_threshold=math.radians(85.0))

        # Update protection with new crease edges
        bm.edges.ensure_lookup_table()
        for e in bm.edges:
            if e not in protected_edges and len(e.link_faces) == 2:
                n0 = e.link_faces[0].normal
                n1 = e.link_faces[1].normal
                if n0.length > 1e-4 and n1.length > 1e-4:
                    if n0.angle(n1) > crease_rad:
                        protected_edges.add(e)

    print(f"[*] Total dissolved: {total_dissolved:,} edges across {passes} passes")

    # =========================================================================
    # STEP 3: FINAL CLEANUP
    # =========================================================================
    bm.faces.ensure_lookup_table()
    ngons_left = [f for f in bm.faces if f.is_valid and len(f.verts) > 4]
    if ngons_left:
        res = bmesh.ops.triangulate(bm, faces=ngons_left, quad_method='BEAUTY', ngon_method='BEAUTY')
        tri_faces = [f for f in res['faces'] if f.is_valid and len(f.verts) == 3]
        bmesh.ops.join_triangles(bm, faces=tri_faces,
                                 angle_face_threshold=math.radians(16.0),
                                 angle_shape_threshold=math.radians(85.0))

    # Master zig-zag pass: merge coplanar triangle pairs into clean quads
    bm.faces.ensure_lookup_table()
    all_tris = [f for f in bm.faces if f.is_valid and len(f.verts) == 3]
    if all_tris:
        bmesh.ops.join_triangles(bm, faces=all_tris,
                                 angle_face_threshold=math.radians(16.0),
                                 angle_shape_threshold=math.radians(85.0))

    # Diamond dart adapters
    bm.faces.ensure_lookup_table()
    apply_diamond_dart_adapters(bm, crease_rad)

    # Loose cleanup
    loose_edges = [e for e in bm.edges if e.is_valid and len(e.link_faces) == 0]
    for e in loose_edges:
        bm.edges.remove(e)
    loose_verts = [v for v in bm.verts if v.is_valid and len(v.link_edges) == 0]
    for v in loose_verts:
        bm.verts.remove(v)

    # Final manifold enforcement
    for e in list(bm.edges):
        if e.is_valid and len(e.link_faces) > 2:
            faces = sorted(list(e.link_faces), key=lambda f: f.calc_area(), reverse=True)
            for f in faces[2:]:
                if f.is_valid:
                    bm.faces.remove(f)

    bm.edges.ensure_lookup_table()
    bm.faces.ensure_lookup_table()
    bm.verts.ensure_lookup_table()

    # Capture sharp edges
    crease_rad_sharp = math.radians(crease_angle_deg)
    sharp_indices = set()
    for e in bm.edges:
        if len(e.link_faces) == 2:
            n0 = e.link_faces[0].normal
            n1 = e.link_faces[1].normal
            if n0.length > 1e-4 and n1.length > 1e-4:
                if n0.angle(n1) > crease_rad_sharp:
                    sharp_indices.add(e.index)
        elif e.is_boundary:
            sharp_indices.add(e.index)

    bmesh.update_edit_mesh(target_obj.data)
    bpy.ops.object.mode_set(mode='OBJECT')

    # =========================================================================
    # STEP 4: AUDIT
    # =========================================================================
    final_polys = len(target_obj.data.polygons)
    degrees = [len(p.vertices) for p in target_obj.data.polygons]
    quads = sum(1 for d in degrees if d == 4)
    tris = sum(1 for d in degrees if d == 3)
    ngons_final = sum(1 for d in degrees if d > 4)

    final_aspects = []
    long_edge_dots = []

    for p in target_obj.data.polygons:
        if len(p.vertices) == 4:
            vs = [target_obj.data.vertices[i].co for i in p.vertices]
            e0 = vs[1] - vs[0]
            e1 = vs[2] - vs[1]
            e2 = vs[3] - vs[2]
            e3 = vs[0] - vs[3]
            e_lens = [e.length for e in [e0, e1, e2, e3]]
            l_max = max(e_lens)
            l_min = min(e_lens)
            if l_min > 1e-6:
                final_aspects.append(l_max / l_min)
            if l_max > 1e-6:
                idx_max = e_lens.index(l_max)
                edges_vec = [e0, e1, e2, e3]
                opp_idx = (idx_max + 2) % 4
                v_main = edges_vec[idx_max].normalized() if edges_vec[idx_max].length > 1e-6 else mathutils.Vector((0,0,0))
                v_opp = edges_vec[opp_idx].normalized() if edges_vec[opp_idx].length > 1e-6 else mathutils.Vector((0,0,0))
                if v_main.length > 0.5 and v_opp.length > 0.5:
                    long_edge_dots.append(abs(v_main.dot(v_opp)))

    print("\n" + "=" * 75)
    print(">>> [FINAL AUDIT: TRIPO-GRADE QUALITY REPORT]")
    print("=" * 75)
    print(f"[*] Faces: {initial_faces:,} -> {final_polys:,}  (-{(1-final_polys/max(initial_faces,1))*100:.1f}%)")
    print(f"[*] Quads: {quads:,} ({quads/max(final_polys,1)*100:.1f}%)")
    print(f"[*] Tris:  {tris:,} ({tris/max(final_polys,1)*100:.1f}%)")
    print(f"[*] N-gons: {ngons_final} (target: 0)")
    if final_aspects:
        print(f"[*] Aspect Ratios: median={np.median(final_aspects):.2f}  90th%={np.percentile(final_aspects,90):.2f}  max={np.max(final_aspects):.2f}")
        print(f"    -> >2:1: {sum(1 for a in final_aspects if a>2.0)/len(final_aspects)*100:.1f}%  >3:1: {sum(1 for a in final_aspects if a>3.0)/len(final_aspects)*100:.1f}%")
    if long_edge_dots:
        print(f"[*] Long-edge parallelism: median={np.median(long_edge_dots):.4f}  >=0.997: {sum(1 for d in long_edge_dots if d>=0.997)/len(long_edge_dots)*100:.1f}%")
    print("=" * 75)

    # =========================================================================
    # STEP 5: NORMALS & SHARP EDGES
    # =========================================================================
    for p in target_obj.data.polygons:
        p.use_smooth = True

    sharp_count = 0
    for e in target_obj.data.edges:
        if e.index in sharp_indices:
            e.use_edge_sharp = True
            sharp_count += 1
    print(f"[*] Marked {sharp_count:,} sharp edges.")

    compute_cad_split_normals(target_obj.data, crease_rad=math.radians(crease_angle_deg))
    print(f"[+] CAD split normals baked ({crease_angle_deg}° cutoff)")

    studio_mat = bpy.data.materials.new(name="Studio_Clay")
    studio_mat.use_nodes = True
    bsdf = studio_mat.node_tree.nodes.get('Principled BSDF')
    if bsdf:
        bsdf.inputs['Base Color'].default_value = (0.78, 0.79, 0.82, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.45
        bsdf.inputs['Metallic'].default_value = 0.0
    target_obj.data.materials.clear()
    target_obj.data.materials.append(studio_mat)

    for o in bpy.context.scene.objects:
        o.select_set(False)
    target_obj.select_set(True)
    bpy.context.view_layer.objects.active = target_obj

    # =========================================================================
    # EXPORT
    # =========================================================================
    bpy.ops.wm.obj_export(
        filepath=output_obj_path,
        forward_axis='NEGATIVE_Z', up_axis='Y',
        export_selected_objects=True,
        export_uv=False,
        export_normals=True,
        export_materials=True
    )
    print(f"[+] OBJ: {output_obj_path}")

    if output_glb_path:
        bpy.ops.export_scene.gltf(
            filepath=output_glb_path,
            use_selection=True,
            export_materials='EXPORT',
            export_normals=True
        )
        print(f"[+] GLB: {output_glb_path}")

    # Also export companion FBX to preserve 100% native quads without GLTF triangulation
    output_fbx_path = output_obj_path.rsplit('.', 1)[0] + ".fbx"
    try:
        bpy.ops.export_scene.fbx(filepath=output_fbx_path, use_selection=True)
        print(f"[+] FBX (100% Native Quads): {output_fbx_path}")
    except Exception as e:
        print(f"    -> FBX export note: {e}")

    # =========================================================================
    # WIREFRAME PREVIEW
    # =========================================================================
    if preview_png_path:
        world = bpy.context.scene.world or bpy.data.worlds.new('World')
        bpy.context.scene.world = world

        # Base Clay
        clay_mat = bpy.data.materials.new(name="Clay")
        bsdf = clay_mat.node_tree.nodes.get('Principled BSDF')
        if bsdf:
            bsdf.inputs['Base Color'].default_value = (0.75, 0.77, 0.80, 1.0)
            bsdf.inputs['Roughness'].default_value = 0.4
        target_obj.data.materials.clear()
        target_obj.data.materials.append(clay_mat)

        # Clean Quad Wireframe Overlay (renders actual polygon edges, zero triangulation diagonals)
        wire_obj = target_obj.copy()
        wire_obj.data = target_obj.data.copy()
        bpy.context.scene.collection.objects.link(wire_obj)

        wire_mat = bpy.data.materials.new(name="WireBlack")
        wbsdf = wire_mat.node_tree.nodes.get('Principled BSDF')
        if wbsdf:
            wbsdf.inputs['Base Color'].default_value = (0.05, 0.05, 0.07, 1.0)
            wbsdf.inputs['Roughness'].default_value = 0.8
        wire_obj.data.materials.clear()
        wire_obj.data.materials.append(wire_mat)

        bb = target_obj.dimensions
        max_dim = max(bb.x, bb.y, bb.z)
        mod = wire_obj.modifiers.new(name="Wire", type='WIREFRAME')
        mod.thickness = max(0.0008, max_dim * 0.0009)
        mod.use_replace = True
        mod.use_even_offset = True

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

        bbox_corners = [target_obj.matrix_world @ mathutils.Vector(c)
                        for c in target_obj.bound_box]
        center = sum(bbox_corners, mathutils.Vector((0, 0, 0))) / 8.0
        bb = target_obj.dimensions
        max_dim = max(bb.x, bb.y, bb.z)
        cam_dist = max_dim * 1.6
        cam_obj.location = center + mathutils.Vector((cam_dist*0.35, -cam_dist*1.05, cam_dist*0.55))
        direction = center - cam_obj.location
        rot_quat = direction.to_track_quat('-Z', 'Y')
        cam_obj.rotation_euler = rot_quat.to_euler()

        scene = bpy.context.scene
        scene.render.engine = 'CYCLES'
        scene.cycles.samples = 16
        scene.cycles.adaptive_threshold = 0.1
        scene.render.resolution_x = 1200
        scene.render.resolution_y = 900
        scene.render.filepath = preview_png_path
        bpy.ops.render.render(write_still=True)
        print(f"[+] Preview: {preview_png_path}")


if __name__ == "__main__":
    args_list = []
    if "--" in sys.argv:
        args_list = sys.argv[sys.argv.index("--") + 1:]

    parser = argparse.ArgumentParser(description="Stage 5: Tripo-Grade Full-Rail Anisotropic Ribbon Engine v3")
    parser.add_argument("input")
    parser.add_argument("output_obj")
    parser.add_argument("--cad_ref", default=None)
    parser.add_argument("--output_glb", default=None)
    parser.add_argument("--preview", default=None)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--crease", type=float, default=18.0)
    parser.add_argument("--no_feature_protect", action="store_true")
    args = parser.parse_args(args_list)

    run_anisotropic_ribbon_engine(
        args.input,
        args.output_obj,
        cad_ref_path=args.cad_ref,
        output_glb_path=args.output_glb,
        preview_png_path=args.preview,
        passes=args.passes,
        crease_angle_deg=args.crease,
        feature_protect=not args.no_feature_protect
    )
