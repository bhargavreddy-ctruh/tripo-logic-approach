import bpy
import math
import mathutils

def render_model(obj_path, out_png, mode='shaded', color=(0.88, 0.88, 0.90, 1.0), wireframe=False):
    bpy.ops.wm.read_factory_settings(use_empty=True)
    
    # Import
    if obj_path.endswith('.obj'):
        bpy.ops.wm.obj_import(filepath=obj_path)
    elif obj_path.endswith('.glb') or obj_path.endswith('.gltf'):
        bpy.ops.import_scene.gltf(filepath=obj_path)

    objs = [o for o in bpy.context.scene.objects if o.type == 'MESH']
    if not objs:
        print(f"Error: No mesh found in {obj_path}")
        return

    # Join if multiple
    bpy.context.view_layer.objects.active = objs[0]
    for o in objs:
        o.select_set(True)
    if len(objs) > 1:
        bpy.ops.object.join()
    obj = bpy.context.active_object
    bpy.ops.object.shade_smooth()

    if wireframe:
        # Add wireframe modifier
        wf = obj.modifiers.new(name="Wireframe", type='WIREFRAME')
        wf.thickness = 0.002
        wf.use_replace = False

    # Studio World
    world = bpy.context.scene.world or bpy.data.worlds.new('World')
    bpy.context.scene.world = world
    world.use_nodes = True
    world.node_tree.nodes['Background'].inputs['Color'].default_value = (0.05, 0.05, 0.07, 1.0)

    # Material
    mat = bpy.data.materials.new(name="M_Render")
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Roughness'].default_value = 0.35
    if mode == 'gold':
        bsdf.inputs['Base Color'].default_value = (1.0, 0.72, 0.12, 1.0)
        bsdf.inputs['Metallic'].default_value = 0.8
        bsdf.inputs['Roughness'].default_value = 0.25
    elif mode == 'tensor':
        bsdf.inputs['Base Color'].default_value = (0.15, 0.85, 0.95, 1.0)
        bsdf.inputs['Roughness'].default_value = 0.3
    
    if len(obj.data.materials) == 0:
        obj.data.materials.append(mat)
    else:
        obj.data.materials[0] = mat

    # 3-Point Lighting
    l1 = bpy.data.lights.new(name='Key', type='SUN')
    l1.energy = 5.0
    l1_obj = bpy.data.objects.new('Key', l1)
    l1_obj.rotation_euler = (math.radians(50), math.radians(20), math.radians(-35))
    bpy.context.scene.collection.objects.link(l1_obj)

    l2 = bpy.data.lights.new(name='Fill', type='SUN')
    l2.energy = 2.5
    l2_obj = bpy.data.objects.new('Fill', l2)
    l2_obj.rotation_euler = (math.radians(40), math.radians(-30), math.radians(50))
    bpy.context.scene.collection.objects.link(l2_obj)

    l3 = bpy.data.lights.new(name='Rim', type='SUN')
    l3.energy = 3.5
    l3_obj = bpy.data.objects.new('Rim', l3)
    l3_obj.rotation_euler = (math.radians(-60), math.radians(10), math.radians(160))
    bpy.context.scene.collection.objects.link(l3_obj)

    # Camera framing
    cam = bpy.data.cameras.new('Cam')
    cam.lens = 45.0
    cam_obj = bpy.data.objects.new('Cam', cam)
    bpy.context.scene.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    center = sum(bbox_corners, mathutils.Vector((0, 0, 0))) / 8.0
    bb = obj.dimensions
    max_dim = max(bb.x, bb.y, bb.z)

    # Elevated 3/4 perspective
    cam_dist = max_dim * 1.55
    cam_obj.location = center + mathutils.Vector((cam_dist * 0.35, -cam_dist * 1.05, cam_dist * 0.65))
    direction = center - cam_obj.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    cam_obj.rotation_euler = rot_quat.to_euler()

    scene = bpy.context.scene
    scene.render.resolution_x = 1200
    scene.render.resolution_y = 900
    scene.render.filepath = out_png
    bpy.ops.render.render(write_still=True)
    print(f"[+] Rendered: {out_png}")

if __name__ == "__main__":
    doc_dir = "/Users/bhargavreddy/.gemini/antigravity/brain/dd2c49ef-17fb-42ef-8bc0-db83734e3c03/doc_images"
    
    # 1. Step 1 Rails
    render_model(
        "/Users/bhargavreddy/tripo-logic-approach/vacuum_quad_engine/assets/outputs/sofa_rails_tuned.obj",
        f"{doc_dir}/step1_sofa_rails_render.png",
        mode='gold'
    )
    
    # 2. Step 2 Tensor
    render_model(
        "/Users/bhargavreddy/tripo-logic-approach/vacuum_quad_engine/assets/outputs/sofa_stage2.obj",
        f"{doc_dir}/step2_sofa_tensor_render.png",
        mode='tensor'
    )

    # 3. Step 3 Single Mesh
    render_model(
        "/Users/bhargavreddy/tripo-logic-approach/vacuum_quad_engine/assets/outputs/sofa_single_mesh.obj",
        f"{doc_dir}/step3_sofa_single_mesh_render.png",
        mode='shaded'
    )

    # 4. Wireframe Inspection
    render_model(
        "/Users/bhargavreddy/tripo-logic-approach/vacuum_quad_engine/assets/outputs/sofa_single_mesh.obj",
        f"{doc_dir}/sofa_final_wireframe.png",
        mode='shaded',
        wireframe=True
    )

