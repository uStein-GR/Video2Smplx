"""
pipeline_zero_filter_render.py
================================
3-stage in-memory pipeline for SMPL-X parameter files:

  Stage 1 — Zero-out translation
    Sets 'transl' to zero for every frame.

  Stage 2 — Savitzky-Golay smoothing
    Smooths pose parameters across time to remove jitter.

  Stage 3 — Render
    Feeds the processed parameters into an SMPL-X model and writes an MP4.

No intermediate .pkl files are written to disk — all data passes between
stages in memory. Only the final MP4 is saved.

Usage
-----
Edit the CONFIGURATION block at the bottom, then run:
    python pipeline_zero_filter_render.py
"""

import os
import glob
import pickle
import numpy as np
import torch
import trimesh
import pyrender
from tqdm import tqdm
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R
import smplx


# Keys smoothed in Stage 2 (matches original filter.py)
KEYS_TO_SMOOTH = [
    'global_orient',
    'body_pose',
    'left_hand_pose',
    'right_hand_pose',
    'jaw_pose',
    'transl',
]

# Keys rebuilt into smplx_param_vector after each stage
PARAM_VECTOR_KEYS = [
    'global_orient', 'body_pose', 'left_hand_pose', 'right_hand_pose',
    'jaw_pose', 'betas', 'expression', 'transl',
]


# =============================================================================
# STAGE 1 — Zero-out translation  (in-memory)
# =============================================================================

def stage_zero_transl(all_data: list) -> list:
    """
    Receives all_data: list of pickle structures (one per frame).
    Sets 'transl' to zero in-place and rebuilds smplx_param_vector.
    Returns the modified list.
    """
    print("\n" + "=" * 55)
    print("STAGE 1 — Zero-out translation")
    print("=" * 55)

    for data in tqdm(all_data, desc="  Zeroing transl"):
        for person_data in data:
            if not isinstance(person_data, dict):
                continue
            if 'transl' in person_data:
                person_data['transl'] = np.zeros_like(person_data['transl'])
            if all(k in person_data for k in PARAM_VECTOR_KEYS):
                person_data['smplx_param_vector'] = np.concatenate(
                    [person_data[k] for k in PARAM_VECTOR_KEYS], axis=-1
                )

    print("  Stage 1 complete.")
    return all_data


# =============================================================================
# STAGE 2 — Savitzky-Golay smoothing  (in-memory)
# =============================================================================

def stage_smooth(
    all_data: list,
    window_length: int = 15,
    polyorder: int = 3,
) -> list:
    """
    Receives all_data: list of pickle structures (one per frame).
    Applies Savitzky-Golay filter to KEYS_TO_SMOOTH across the time axis.
    Returns the modified list.
    """
    print("\n" + "=" * 55)
    print("STAGE 2 — Savitzky-Golay smoothing")
    print("=" * 55)

    n_frames = len(all_data)
    print(f"  Frames: {n_frames}  |  window={window_length}, polyorder={polyorder}")

    if n_frames < window_length:
        raise ValueError(
            f"Not enough frames ({n_frames}) for window_length={window_length}. "
            "Reduce smooth_window_length."
        )

    # --- Collect time-series per key ---
    timeseries = {k: [] for k in KEYS_TO_SMOOTH}
    for data in all_data:
        params = data[0]
        for key in KEYS_TO_SMOOTH:
            if key in params:
                timeseries[key].append(params[key])

    # --- Smooth each key ---
    smoothed = {}
    for key, frames_list in timeseries.items():
        if len(frames_list) != n_frames:
            print(f"  [WARNING] '{key}' missing in some frames — skipping.")
            continue
        arr    = np.array(frames_list)               # (T, ...)
        flat   = arr.reshape(n_frames, -1)            # (T, D)
        s_flat = savgol_filter(flat, window_length, polyorder, axis=0)
        smoothed[key] = s_flat.reshape(arr.shape)
        print(f"  Smoothed '{key}'  shape={arr.shape}")

    # --- Write smoothed values back into all_data ---
    for i, data in enumerate(all_data):
        p = data[0]
        for key, s_arr in smoothed.items():
            p[key] = s_arr[i]
        if all(k in p for k in PARAM_VECTOR_KEYS):
            p['smplx_param_vector'] = np.concatenate(
                [p[k] for k in PARAM_VECTOR_KEYS], axis=-1
            )

    print("  Stage 2 complete.")
    return all_data


# =============================================================================
# STAGE 3 — Render to video  (in-memory)
# =============================================================================

BODY_KEYS = [
    'betas', 'body_pose', 'left_hand_pose', 'right_hand_pose',
    'jaw_pose', 'expression', 'transl',
]


def stage_render(
    all_data: list,
    smplx_model_path: str,
    output_video_path: str,
    gender: str = 'neutral',
    video_fps: int = 30,
    viewport_size: int = 800,
) -> None:
    """
    Receives all_data: list of pickle structures (one per frame).
    Renders each frame with pyrender and writes an MP4 via cv2.VideoWriter.
    """
    print("\n" + "=" * 55)
    print("STAGE 3 — Rendering video")
    print("=" * 55)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"  Device: {device}")

    # --- Load SMPL-X model ---
    print("  Loading SMPL-X model...")
    model = smplx.SMPLX(
        model_path=smplx_model_path,
        gender=gender,
        use_pca=False,
        flat_hand_mean=True,
        num_betas=10,
        num_expression_coeffs=50,
    ).to(device)
    print("  Model loaded.")

    n_frames = len(all_data)
    print(f"  Frames to render: {n_frames}")

    # --- Auto-position camera from first valid frame ---
    cam_x, cam_y, cam_z = 0.0, 0.0, 2.5
    for data in all_data:
        if data and 'transl' in data[0]:
            t = data[0]['transl']
            cam_x = float(t[0])
            cam_y = float(t[1])
            cam_z = float(t[2]) + 2.5
            print(f"  Camera: x={cam_x:.2f}  y={cam_y:.2f}  z={cam_z:.2f}")
            break

    # --- Orientation correction (flip model upright) ---
    model_correction = R.from_euler('x', np.pi)

    # --- Build pyrender scene ---
    scene = pyrender.Scene(
        bg_color=[0.1, 0.1, 0.3, 1.0],
        ambient_light=[0.3, 0.3, 0.3],
    )
    camera_pose = np.array([
        [1.0, 0.0, 0.0, cam_x],
        [0.0, 1.0, 0.0, cam_y],
        [0.0, 0.0, 1.0, cam_z],
        [0.0, 0.0, 0.0, 1.0],
    ])
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0),
              pose=camera_pose)
    scene.add(pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0),
              pose=camera_pose)

    renderer = pyrender.OffscreenRenderer(
        viewport_width=viewport_size, viewport_height=viewport_size
    )

    # --- Video writer ---
    output_video_path = os.path.abspath(output_video_path)

    # Guard: a previous broken run may have created a FOLDER with this name
    if os.path.isdir(output_video_path):
        import shutil
        print(f"  [WARNING] Folder found at output path — removing: {output_video_path}")
        shutil.rmtree(output_video_path)

    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)

    # cv2.VideoWriter avoids all imageio plugin resolution issues
    import cv2
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video_path, fourcc, video_fps,
                             (viewport_size, viewport_size))
    if not writer.isOpened():
        raise RuntimeError(f"cv2.VideoWriter could not open: {output_video_path}")
    print(f"  Writing to: {output_video_path}")

    # cv2 expects BGR; pyrender gives RGB
    blank_rgb = np.full((viewport_size, viewport_size, 3), (26, 26, 76), dtype=np.uint8)
    blank_bgr = blank_rgb[:, :, ::-1]

    for frame_idx, frame_data in enumerate(tqdm(all_data, desc="  Rendering")):
        try:
            if not frame_data:
                writer.write(blank_bgr)
                continue

            person = frame_data[0]

            # Build body_params
            body_params = {}
            for key in BODY_KEYS:
                if key in person:
                    body_params[key] = (
                        torch.tensor(person[key], dtype=torch.float32)
                        .unsqueeze(0).to(device)
                    )

            # Apply orientation correction to global_orient
            orig_mat = R.from_rotvec(
                np.array(person['global_orient'], dtype=np.float32).flatten()
            ).as_matrix()
            corrected_mat = model_correction.as_matrix() @ orig_mat
            body_params['global_orient'] = torch.tensor(
                R.from_matrix(corrected_mat).as_rotvec(),
                dtype=torch.float32,
            ).unsqueeze(0).to(device)

            with torch.no_grad():
                out      = model(return_verts=True, **body_params)
                vertices = out.vertices.detach().cpu().numpy().squeeze()

            mesh = trimesh.Trimesh(vertices=vertices, faces=model.faces)
            for node in list(scene.mesh_nodes):
                scene.remove_node(node)
            scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=True))

            color, _ = renderer.render(scene)  # RGB uint8
            writer.write(color[:, :, ::-1])    # convert RGB -> BGR for cv2

        except Exception as e:
            print(f"\n  [ERROR] frame {frame_idx}: {e}")
            writer.write(blank_bgr)

    writer.release()
    renderer.delete()
    print(f"\n  Stage 3 complete. Video saved: {output_video_path}")


# =============================================================================
# PIPELINE ENTRY POINT
# =============================================================================

VIDEO_NAME = 'smplest_wilor_emoca.mp4'

def run_pipeline(
    input_pkl_folder: str,
    smplx_model_path: str,
    output_dir: str,
    smooth_window_length: int = 15,
    smooth_polyorder: int = 3,
    gender: str = 'neutral',
    video_fps: int = 30,
    viewport_size: int = 800,
):
    """
    Loads all .pkl files once, passes them through 3 in-memory stages,
    saves the processed .pkl files to output_dir/params/,
    and writes the final MP4 to output_dir/smplest_wilor_emoca.mp4.
    """
    output_dir        = os.path.abspath(output_dir)
    params_out_dir    = os.path.join(output_dir, 'params')
    output_video_path = os.path.join(output_dir, VIDEO_NAME)

    os.makedirs(params_out_dir, exist_ok=True)
    os.makedirs(output_dir,     exist_ok=True)

    print("\n" + "#" * 55)
    print("  SMPL-X PIPELINE: zero → smooth → render")
    print("#" * 55)
    print(f"  Input      : {input_pkl_folder}")
    print(f"  Params out : {params_out_dir}")
    print(f"  Video out  : {output_video_path}")

    # --- Load all frames once ---
    file_paths = sorted(glob.glob(os.path.join(input_pkl_folder, '*.pkl')))
    if not file_paths:
        raise FileNotFoundError(f"No .pkl files found in: {input_pkl_folder}")
    print(f"\n  Loading {len(file_paths)} files...")

    all_data = []
    for fp in tqdm(file_paths, desc="  Loading"):
        with open(fp, 'rb') as f:
            all_data.append(pickle.load(f))

    # --- Run stages ---
    all_data = stage_zero_transl(all_data)
    all_data = stage_smooth(all_data, smooth_window_length, smooth_polyorder)

    # --- Save processed parameters ---
    print("\n" + "=" * 55)
    print("SAVING processed parameters")
    print("=" * 55)
    for i, data in enumerate(tqdm(all_data, desc="  Saving params")):
        out_name = os.path.basename(file_paths[i])
        with open(os.path.join(params_out_dir, out_name), 'wb') as f:
            pickle.dump(data, f)
    print(f"  Saved {len(all_data)} files to: {params_out_dir}")

    # --- Render ---
    stage_render(all_data, smplx_model_path, output_video_path,
                 gender, video_fps, viewport_size)

    print("\n" + "#" * 55)
    print("  PIPELINE COMPLETE")
    print("#" * 55)


# =============================================================================
# ENTRY POINT — supports both direct editing (legacy) and CLI via pipeline.py
# =============================================================================

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Zero-out translation, smooth, and render SMPL-X .pkl files to MP4.'
    )
    parser.add_argument('--input_pkl_folder',     required=False, default=None,
                        help='Folder of fused .pkl files (one per frame).')
    parser.add_argument('--smplx_model_path',     required=False, default=None,
                        help='Path to SMPLX_NEUTRAL.npz.')
    parser.add_argument('--output_dir',           required=False, default=None,
                        help='Output directory (params/ subfolder + MP4 saved here).')
    parser.add_argument('--smooth_window_length', type=int,   default=15)
    parser.add_argument('--smooth_polyorder',     type=int,   default=3)
    parser.add_argument('--gender',               default='neutral')
    parser.add_argument('--video_fps',            type=int,   default=30)
    parser.add_argument('--viewport_size',        type=int,   default=800)
    args = parser.parse_args()

    # If CLI args are provided use them; otherwise fall back to the hardcoded
    # defaults below so the script still works when run directly without args.
    _input  = args.input_pkl_folder or r""
    _model  = args.smplx_model_path or r""
    _outdir = args.output_dir       or r""

    run_pipeline(
        input_pkl_folder     = _input,
        smplx_model_path     = _model,
        output_dir           = _outdir,
        smooth_window_length = args.smooth_window_length,
        smooth_polyorder     = args.smooth_polyorder,
        gender               = args.gender,
        video_fps            = args.video_fps,
        viewport_size        = args.viewport_size,
    )