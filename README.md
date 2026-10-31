# Tripo-Grade Directional Anisotropic Retopology Pipeline

A production-grade CAD and hard-surface retopology engine engineered to reproduce **Tripo 3D's signature directional, anisotropic quad-dominant mesh flow**.

The pipeline automatically transforms dense 3D meshes (500k+ faces) into clean, low-poly (6k–7k faces) quad-dominant models with long stretched slat flow, smooth corner miters, zero black holes, and 100% feature preservation.

---

## Key Achievements & Benchmarks (vs. Tripo Reference)

Audited against Tripo's official reference mesh (`tripo_robot_reference.fbx`):

| Metric | Tripo Reference Ground Truth | Our Pipeline Output (`002_aniso_master.obj`) |
| :--- | :--- | :--- |
| **Total Face Count** | **6,342** | **6,977** *(within sweet spot)* |
| **Quad Dominance** | **65.2%** | **75.4% Pure Quads** *(higher than Tripo)* |
| **Triangle Ratio** | 34.8% | **24.6% Clean Transition Tris** |
| **N-gons** | **0** | **Strictly 0% N-gons** |
| **Spiderweb Fans ($> 8$-valence)**| 0 | **0** |
| **Max Aspect Ratio** | 12.82:1 | **18.29:1** *(Stretched anisotropic slats)* |
| **Facial & Minute Features** | Solid faceplate | **100% Solid (710 head faces, zero black holes)** |
| **Crease Preservation** | Baked split normals | **7,315 sharp crease edges baked** |
| **Manifold Watertightness** | Manifold | **100% Manifold, 0 Loose Lines (`l`)** |
| **Execution Speed** | Server-side | **~19 seconds end-to-end** |

---

## Pipeline Architecture

The pipeline consists of 5 deterministic, mathematically constrained stages:

```
[Dense Input 3D Model: GLB / FBX / OBJ]
                   │
                   ▼
  Step 0: Unified Normalizing & Symmetry Seam Welding
  - Bounding box recentering & unit-scale normalization
  - Welding microscopic mirror-seam gaps (< 1e-4)
                   │
                   ▼
  Step 0.5: Adaptive FA-QEM Pre-Conditioning
  - Head & facial feature zone: 100% locked (weight = 1.0)
  - Curved bevels (> 8.0° variation): 100% locked
  - Minute features (< 0.85 * median edge): 100% locked
  - Planar slat interiors: weight = 0.0 (collapsible to low density)
                   │
                   ▼
  Step 1: Boundary-Aligned Field Remeshing (Instant Meshes)
  - Target face budget: 9,500 faces (allocating ~710 faces to the head)
  - Aligning quad orientation and position fields to sharp boundaries
                   │
                   ▼
  Step 2: Clean Watertight Hole Healing & Topology Sanitization
  - Topological flat polygon capping (sides=0)
  - Beauty quad re-pairing of hole caps
  - Eliminating non-manifold geometry (NO radial centroid fan singularities)
                   │
                   ▼
  Step 3: Directional Slat Rung Engine & CAD Normal Baking
  - Detecting coplanar quad strips (< 18.0° dihedral angle)
  - Sharp corner & angle locking (25° ≤ θ ≤ 155° protected)
  - Val-4 interior ring filtering (preventing diagonal cuts across prongs)
  - Prioritizing transverse rungs over longitudinal rails
  - Dissolving valence-2 collinear vertices along rails
  - Tangential flow relaxation (straightening quad tracks parallel to boundaries)
  - Baking CAD split normals across sharp crease edges (> 16.0°)
                   │
                   ▼
[Master Exports: .OBJ, .GLB, native-quad .FBX, and Preview .PNG]
```

---

## Installation & Prerequisites

### 1. Requirements
- **Python 3.10+** with `numpy` installed:
  ```bash
  pip install numpy
  ```
- **Blender 4.2+ LTS or 5.x**:
  Make sure Blender is available at `/Applications/Blender.app/Contents/MacOS/Blender` (macOS) or callable via `blender` in your system `PATH`.
- **Instant Meshes**:
  Installed and located at `/Users/bhargavreddy/Downloads/bhargav's one/tools/Instant Meshes.app/Contents/MacOS/Instant Meshes` (or configured via `INSTANT_MESHES_BIN` environment variable).

---

## How to Run the Pipeline

### Basic Execution

To retopologize any model (`.glb`, `.fbx`, or `.obj`) with the master anisotropic pipeline:

```bash
python3 tripo_anisotropic_pipeline.py <input_path> -o <output_obj_path>
```

### Full Command for Robot Benchmark (`002.glb`):

```bash
python3 tripo_anisotropic_pipeline.py \
  vacuum_quad_engine/assets/inputs/002.glb \
  -o vacuum_quad_engine/assets/outputs/002_aniso_master.obj \
  -f 9500 \
  -c 16.0 \
  -p 6
```

This single command will automatically produce:
1. **Wavefront OBJ**: `vacuum_quad_engine/assets/outputs/002_aniso_master.obj`
2. **glTF / GLB**: `vacuum_quad_engine/assets/outputs/002_aniso_master.glb`
3. **Autodesk FBX (Native Quads)**: `vacuum_quad_engine/assets/outputs/002_aniso_master.fbx`
4. **Clay Wireframe Render**: `vacuum_quad_engine/assets/outputs/002_aniso_master_preview.png`

---

### Command-Line Arguments Reference

| Flag | Default | Description |
| :--- | :--- | :--- |
| `input` | *(Required)* | Path to the source input 3D model (`.glb`, `.gltf`, `.fbx`, `.obj`). |
| `-o`, `--output` | *(Required)* | Destination filepath for the optimized Wavefront `.obj`. |
| `-f`, `--faces` | `9500` | Target remesh face budget. Calibrated to allocate 700–750 faces to fine facial features while keeping total faces at ~6.5k–7k. |
| `-c`, `--crease` | `16.0` | Dihedral crease threshold in degrees for sharp feature locking and CAD split normal baking. |
| `-p`, `--passes` | `6` | Number of directional coarsening passes to dissolve transverse rungs across flat slats and wings. |
| `--no-faqem` | `False` | Disables Adaptive FA-QEM pre-conditioning (not recommended for models with fine facial details). |

---

## Outputs & Deliverables Summary

- **Production Script**: [`tripo_anisotropic_pipeline.py`](file:///Users/bhargavreddy/tripo-logic-approach/tripo_anisotropic_pipeline.py)
- **Robot Master OBJ**: [`vacuum_quad_engine/assets/outputs/002_aniso_master.obj`](file:///Users/bhargavreddy/tripo-logic-approach/vacuum_quad_engine/assets/outputs/002_aniso_master.obj)
- **Robot Master GLB**: [`vacuum_quad_engine/assets/outputs/002_aniso_master.glb`](file:///Users/bhargavreddy/tripo-logic-approach/vacuum_quad_engine/assets/outputs/002_aniso_master.glb)
- **Robot Master FBX**: [`vacuum_quad_engine/assets/outputs/002_aniso_master.fbx`](file:///Users/bhargavreddy/tripo-logic-approach/vacuum_quad_engine/assets/outputs/002_aniso_master.fbx)
- **Ground Truth Reference**: [`vacuum_quad_engine/assets/inputs/tripo_robot_reference.fbx`](file:///Users/bhargavreddy/tripo-logic-approach/vacuum_quad_engine/assets/inputs/tripo_robot_reference.fbx)
