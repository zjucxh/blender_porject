# Blender SMPL Cloth Simulation

This project provides a Blender-based pipeline for simulating cloth (e.g., T-shirt) on SMPL body models using motion capture data. It automates the process of loading SMPL models, applying pose and shape parameters, running cloth simulation, and exporting the results.

## Features

- **SMPL Model Loader:** Automatically imports SMPL FBX models and sets up the armature and shape keys.
- **Pose & Shape Application:** Applies pose and shape parameters to the SMPL model for each animation frame.
- **Cloth Simulation:** Imports a garment mesh (T-shirt), sets up Blender's cloth physics, and bakes the simulation.
- **Batch Processing:** Processes multiple `.npz` motion files in a directory.
- **Export:** Exports simulated body and garment meshes as `.ply` files for each frame.

## Requirements

- **Blender** (tested with 3.x)
- **Python** (used inside Blender, with `bpy`, `numpy`, `cv2`, `scipy`)
- **SMPL Model FBX** files in `assets/model/`
- **Garment OBJ** file in `assets/meshes/tshirt_snug.obj`
- **Motion Data**: `.npz` files containing SMPL pose, shape, and translation data

## Usage

### 1. Prepare Assets

- Place the SMPL FBX model (e.g., `basicModel_m_lbs_10_207_0_v1.0.2.fbx`) in `assets/model/`.
- Place the garment OBJ (e.g., `tshirt_snug.obj`) in `assets/meshes/`.
- Place your motion `.npz` files in a directory (e.g., `dataset/CMU_SAMPLED/`).

### 2. Run the Simulation

Open Blender and run the script:

```bash
blender --background --python clothsim.py
```

The script will:
- Load each `.npz` file in the specified directory.
- Simulate the SMPL body and cloth for up to 120 frames per sequence.
- Export the simulated meshes to `dataset/CMU_SIMULATION/<seq_name>/`.

### 3. Output

For each sequence, the following will be saved:
- `animation.npz`: The pose, shape, and translation data used for simulation.
- `tshirt_XXXX.ply`: Simulated garment mesh for each frame.
- `body_XXXX.ply`: Simulated body mesh for each frame.

## Main Classes and Functions

- **SMPLModel**: Handles loading, pose/shape application, and simulation.
  - `apply_shape_pose(beta, pose, frame)`: Applies shape and pose to the SMPL model at a given frame.
  - `simulate(pose_data, output_path)`: Runs the full simulation pipeline for a given motion file.
- **bpy_export_ply(obj, frame, export_path)**: Exports the selected object as a `.ply` mesh at a specific frame.

## Notes

- The simulation uses Blender's cloth physics with preset parameters for cotton-like behavior.
- The script resets Blender to its default state after each simulation to free memory.
- The pipeline expects the SMPL model and garment mesh to have specific names (`m_avg`, `tshirt`).

## Troubleshooting

- If exported meshes are corrupted, ensure the correct object is selected and active before export.
- Make sure all dependencies (`numpy`, `cv2`, `scipy`) are available in Blender's Python environment.

## License

This project is for research and educational purposes. Please ensure you have the rights to use the SMPL model and motion data.
