import os
import sys
import argparse
import subprocess
import tempfile
import time
import numpy as np
import trimesh

def run_field_streamline_retopo(
    input_path, 
    output_path,
    target_quads=4500,
    preserve_sharp=True,
    sharp_angle=28.0
):
    print(f"[*] [STAGE 4] Executing Field-Integrated Streamline Retopology on: {input_path}")
    print(f"    -> Target Face Budget: ~{target_quads:,} quads")

    # Locate Quadriflow binary or Blender
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    quadriflow_bin = os.path.join(workspace_dir, "bin", "quadriflow")
    
    # 1. Load and prepare input mesh
    print(f"[*] Loading and preparing manifold geometry...")
    mesh = trimesh.load(input_path, force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = mesh.dump(concatenate=True)
    mesh.merge_vertices()
    if hasattr(mesh, 'remove_duplicate_faces'):
        mesh.remove_duplicate_faces()
    if hasattr(mesh, 'remove_degenerate_faces'):
        mesh.remove_degenerate_faces()
    if hasattr(mesh, 'nondegenerate_faces'):
        mesh.update_faces(mesh.nondegenerate_faces())

    orig_verts = len(mesh.vertices)
    orig_faces = len(mesh.faces)
    print(f"    -> Working Mesh: {orig_verts:,} vertices | {orig_faces:,} faces")

    # 2. Execute via Standalone Quadriflow C++ Solver if available
    if os.path.exists(quadriflow_bin) and os.access(quadriflow_bin, os.X_OK):
        print(f"[*] Using native Quadriflow Streamline Solver: {quadriflow_bin}")
        
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_input = os.path.join(tmp_dir, "input_clean.obj")
            tmp_output = os.path.join(tmp_dir, "output_retopo.obj")
            
            # Export clean manifold OBJ
            mesh.export(tmp_input)

            # Determine nominal face parameter for quadriflow
            if orig_faces < 20000 and target_quads > 2000:
                base_f = int(target_quads * 2.5)
            else:
                base_f = int(target_quads)

            # Robust candidate search list
            candidates = [
                base_f,
                int(base_f * 0.95),
                int(base_f * 1.05),
                int(base_f * 0.90),
                int(base_f * 0.85),
                int(base_f * 0.75),
                target_quads
            ]

            t0 = time.time()
            success = False

            for f_val in candidates:
                for sharp_flag in ([True, False] if preserve_sharp else [False]):
                    if os.path.exists(tmp_output):
                        os.remove(tmp_output)

                    cmd = [quadriflow_bin, "-i", tmp_input, "-o", tmp_output, "-f", str(f_val)]
                    if sharp_flag:
                        cmd.append("-sharp")

                    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                    if os.path.exists(tmp_output) and os.path.getsize(tmp_output) > 100:
                        success = True
                        print(f"    -> Streamlines integrated successfully with f={f_val} (sharp={sharp_flag}) in {time.time() - t0:.2f}s.")
                        break
                if success:
                    break

            if not success or not os.path.exists(tmp_output) or os.path.getsize(tmp_output) == 0:
                raise RuntimeError(f"Quadriflow failed across all candidate budgets: {res.stderr or res.stdout}")

            # Copy to target output
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            with open(tmp_output, 'r') as src, open(output_path, 'w') as dst:
                dst.write(f"# Anisotropic Streamline Quad Mesh (Stage 4)\n")
                dst.write(f"# Generated via Field-Integrated Quadriflow Solver\n")
                for line in src:
                    dst.write(line)

    else:
        # Fallback to Blender headless engine if Quadriflow binary is absent
        blender_bin = "/Applications/Blender.app/Contents/MacOS/Blender"
        if not os.path.exists(blender_bin):
            blender_bin = "blender"
            
        print(f"[*] Delegating to Blender Headless Engine: {blender_bin}")
        blender_script = f"""
import bpy
bpy.ops.wm.read_factory_settings(use_empty=True)
bpy.ops.wm.obj_import(filepath={repr(input_path)})
mesh_objs = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
bpy.context.view_layer.objects.active = mesh_objs[0]
for obj in mesh_objs: obj.select_set(True)
if len(mesh_objs) > 1: bpy.ops.object.join()
bpy.ops.object.quadriflow_remesh(
    use_mesh_symmetry=False,
    use_preserve_sharp={preserve_sharp},
    use_preserve_boundary=True,
    preserve_attributes=False,
    smooth_normals=True,
    mode='FACES',
    target_faces={target_quads},
    seed=42
)
bpy.ops.wm.obj_export(filepath={repr(output_path)}, export_selected_objects=True)
"""
        with tempfile.NamedTemporaryFile('w', suffix='.py', delete=False) as f_script:
            f_script.write(blender_script)
            script_path = f_script.name

        try:
            cmd = [blender_bin, "--background", "--python", script_path]
            subprocess.run(cmd, check=True)
        finally:
            if os.path.exists(script_path):
                os.remove(script_path)

    # 3. Analyze resulting Quad-Dominant Mesh
    with open(output_path, 'r') as f:
        poly_lines = [line.strip().split() for line in f if line.startswith('f ')]

    quad_count = sum(1 for p in poly_lines if len(p) == 5)
    tri_count = sum(1 for p in poly_lines if len(p) == 4)
    total_polys = len(poly_lines)
    quad_pct = (quad_count / max(1, total_polys)) * 100.0

    print(f"[+] [STAGE 4 COMPLETE] Newly generated quad mesh saved to: {output_path}")
    print(f"    -> Output Polygons: {total_polys:,} (Original was {orig_faces:,})")
    print(f"    -> Quads: {quad_count:,} ({quad_pct:.1f}%) | Triangles: {tri_count:,} ({100.0 - quad_pct:.1f}%)")

    # 4. Also export companion GLB for direct previewing in 3D viewers
    if output_path.endswith(".obj"):
        glb_path = output_path.replace(".obj", ".glb")
        try:
            # Parse faces for GLB triangulation
            parsed_verts = []
            parsed_tris = []
            with open(output_path, 'r') as f_obj:
                for line in f_obj:
                    if line.startswith('v '):
                        parts = line.strip().split()[1:4]
                        parsed_verts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    elif line.startswith('f '):
                        indices = [int(p.split('/')[0]) - 1 for p in line.strip().split()[1:]]
                        if len(indices) == 4:
                            parsed_tris.append([indices[0], indices[1], indices[2]])
                            parsed_tris.append([indices[0], indices[2], indices[3]])
                        elif len(indices) == 3:
                            parsed_tris.append(indices)
            
            glb_mesh = trimesh.Trimesh(
                vertices=np.array(parsed_verts, dtype=np.float64),
                faces=np.array(parsed_tris, dtype=np.int64),
                process=False
            )
            glb_mesh.export(glb_path)
            print(f"[+] Also exported GLB companion: {glb_path}")
        except Exception as e:
            print(f"    -> GLB companion note: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 4: Field-Integrated Streamline Retopology")
    parser.add_argument("input", help="Raw input asset (.obj, .glb, .stl)")
    parser.add_argument("output", help="Output quad OBJ path")
    parser.add_argument("--quads", type=int, default=4500, help="Target quad face count (default: 4500)")
    parser.add_argument("--crease", type=float, default=28.0, help="Sharp crease cutoff (deg)")
    args = parser.parse_args()

    run_field_streamline_retopo(
        args.input, 
        args.output,
        target_quads=args.quads,
        sharp_angle=args.crease
    )
