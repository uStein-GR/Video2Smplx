#!/usr/bin/env python
"""
pipeline.py  —  End-to-end full-body reconstruction pipeline
=============================================================
Orchestrates SMPLest-X + WiLoR + EMOCA in their own conda environments,
fuses the outputs, then smooths and renders a final MP4.

Each sub-system runs inside its own conda environment via `conda run`.
No packages are shared between environments — no dependency conflicts.

Directory layout (auto-created)
--------------------------------
  MoE/
    demo/
      input/                   ← extracted frames  (PERSISTENT — never deleted)
      result_params_unified/   ← WiLoR output  (../demo/result_params_unified from WiLoR cwd)
        params/
  SMPLest-X-Inference/
    demo/
      <name>.mp4               ← copy of input video placed here for SMPLest-X
      output_params/<name>/    ← SMPLest-X per-frame .pkl output
  EMOCA-Inference/
    demo/
      input/                   ← frames copied here for EMOCA
      output/                  ← EMOCA per-frame .pkl output

Quick-start
-----------
    python pipeline.py \\
        --video  demo/P.mp4 \\
        --output output/ \\
        --smplestx_env ubuntu \\
        --wilor_env    ubuntu \\
        --emoca_env    work38

Full example with all options
------------------------------
    python pipeline.py \\
        --video  demo/P.mp4 \\
        --output output/ \\
        --name   my_run \\
        --smplestx_env ubuntu  --wilor_env ubuntu  --emoca_env work38 \\
        --smplestx_ckpt smplest_x_h \\
        --emoca_model   EMOCA_v2_lr_mse_20 \\
        --fps 30  --viewport 800 \\
        --smooth_window 15  --smooth_poly 3

Skip flags (resume a partial run)
----------------------------------
    --skip_extract   --skip_smplestx  --skip_wilor
    --skip_emoca     --skip_fuse      --skip_render
"""

import argparse
import os
import pickle
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Project root and sub-project directories
# ---------------------------------------------------------------------------

ROOT         = Path(__file__).resolve().parent
SMPLESTX_DIR = ROOT / 'SMPLest-X-Inference'
WILOR_DIR    = ROOT / 'WiLoR-Inference'
EMOCA_DIR    = ROOT / 'EMOCA-Inference'

# Shared frame folder — WiLoR reads via "../demo/input" from its cwd
SHARED_FRAMES_DIR = ROOT / 'demo' / 'input'

DEFAULT_SMPLESTX_CKPT = 'smplest_x_h'
DEFAULT_EMOCA_MODEL   = 'EMOCA_v2_lr_mse_20'
DEFAULT_SMPLX_MODEL   = str(
    SMPLESTX_DIR / 'human_models' / 'human_model_files' / 'smplx' / 'SMPLX_NEUTRAL.npz'
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _banner(title: str):
    bar = '=' * 62
    print(f'\n{bar}\n  {title}\n{bar}')


def _extract_id(filename: str):
    """Return the last integer found in a filename, or None."""
    m = re.findall(r'\d+', filename)
    return int(m[-1]) if m else None


def conda_run(env: str, script: Path, script_args: list,
              cwd: Path = None, label: str = ''):
    """
    Run `python <script> <args>` inside a conda environment.
    Prints the full command before executing so every step is transparent.
    Raises subprocess.CalledProcessError on failure.
    """
    cmd = (
        ['conda', 'run', '--no-capture-output', '-n', env, 'python', str(script)]
        + [str(a) for a in script_args]
    )
    print(f'\n{"─" * 62}')
    print(f'  [{label}]  env={env}')
    print(f'  cwd : {cwd or ROOT}')
    print(f'  $ {" ".join(cmd)}')
    print(f'{"─" * 62}')
    subprocess.run(cmd, cwd=str(cwd or ROOT), check=True)


# ---------------------------------------------------------------------------
# Stage 0 — Extract frames from video
# ---------------------------------------------------------------------------

def stage_extract(video: Path, fps: int) -> int:
    """
    Extract frames from the input video using ffmpeg.
    Frames are saved as 000001.jpg, 000002.jpg, … inside MoE/demo/input/.
    This folder is NEVER deleted by the pipeline.
    Returns the total frame count.
    """
    _banner('STAGE 0 — Frame extraction')
    SHARED_FRAMES_DIR.mkdir(parents=True, exist_ok=True)

    subprocess.run([
        'ffmpeg', '-y',
        '-i', str(video),
        '-vf', f'fps={fps}',
        '-qscale', '0',
        str(SHARED_FRAMES_DIR / '%06d.jpg'),
    ], check=True)

    n = len(list(SHARED_FRAMES_DIR.glob('*.jpg')))
    print(f'\n  Done. {n} frames → {SHARED_FRAMES_DIR}')
    return n


# ---------------------------------------------------------------------------
# Stage 1 — SMPLest-X full-body inference
# ---------------------------------------------------------------------------

def stage_smplestx(video: Path, name: str, ckpt: str, env: str) -> Path:
    """
    Run SMPLest-X on the input video.

    SMPLest-X expects the video at:
        SMPLest-X-Inference/demo/<name>.mp4
    We copy the video there, then call:
        python main/inference.py --num_gpus 1 --video <name>.mp4
                                 --ckpt_name <ckpt> --save_params

    SMPLest-X auto-extracts frames internally; the pipeline preserves its
    own shared frame folder (MoE/demo/input/) independently.

    Output .pkl files end up at:
        SMPLest-X-Inference/demo/output_params/<name>/
    """
    _banner('STAGE 1 — SMPLest-X body inference')

    # Place video where SMPLest-X expects it
    smplestx_demo = SMPLESTX_DIR / 'demo'
    smplestx_demo.mkdir(parents=True, exist_ok=True)
    video_dst = smplestx_demo / video.name
    if video_dst.resolve() != video.resolve():
        shutil.copy2(str(video), str(video_dst))
        print(f'  Copied video → {video_dst}')

    # Remove any leftover symlink/dir from a previous run at the frames location
    # inference.py does os.makedirs(..., exist_ok=True) which fails on a symlink
    leftover = SMPLESTX_DIR / 'demo' / 'input_frames' / name
    if leftover.is_symlink():
        leftover.unlink()
        print(f'  Removed leftover symlink: {leftover}')
    elif leftover.is_dir():
        shutil.rmtree(str(leftover))
        print(f'  Removed leftover dir: {leftover}')

    conda_run(
        env,
        SMPLESTX_DIR / 'main' / 'inference.py',
        [
            '--num_gpus',  '1',
            '--video',     video.name,
            '--ckpt_name', ckpt,
            '--save_params',
        ],
        cwd=SMPLESTX_DIR,
        label='SMPLest-X',
    )

    out = SMPLESTX_DIR / 'demo' / 'output_params' / name
    n = len(list(out.glob('*.pkl'))) if out.exists() else 0
    print(f'\n  Done. {n} .pkl files → {out}')
    return out


# ---------------------------------------------------------------------------
# Stage 2 — WiLoR hand inference
# ---------------------------------------------------------------------------

def stage_wilor(env: str) -> Path:
    """
    Run WiLoR on the shared frames using absolute paths.
    No frame copying — reads directly from MoE/demo/input/.

    Output .pkl files are in:
        MoE/demo/result_params_unified/params/
    """
    _banner('STAGE 2 — WiLoR hand inference')

    wilor_result_dir = ROOT / 'demo' / 'result_params_unified'
    wilor_result_dir.mkdir(parents=True, exist_ok=True)

    conda_run(
        env,
        WILOR_DIR / 'demo_params_unified.py',
        [
            '--img_folder', str(SHARED_FRAMES_DIR),
            '--out_folder', str(wilor_result_dir),
            '--save_params',
        ],
        cwd=WILOR_DIR,
        label='WiLoR',
    )

    out = ROOT / 'demo' / 'result_params_unified' / 'params'
    n = len(list(out.glob('*.pkl'))) if out.exists() else 0
    print(f'\n  Done. {n} .pkl files → {out}')
    return out


# ---------------------------------------------------------------------------
# Stage 3 — EMOCA face inference
# ---------------------------------------------------------------------------

def stage_emoca(model_name: str, env: str) -> Path:
    """
    Run EMOCA on the shared frames using absolute paths.
    No frame copying — reads directly from MoE/demo/input/.

    Output .pkl files (one per frame with keys 'exp', 'jaw_pose') are in:
        EMOCA-Inference/demo/output/
    """
    _banner('STAGE 3 — EMOCA face inference')

    emoca_out = EMOCA_DIR / 'demo' / 'output'
    emoca_out.mkdir(parents=True, exist_ok=True)

    conda_run(
        env,
        EMOCA_DIR / 'gdl_apps' / 'EMOCA' / 'demos' / 'visualize3.py',
        [
            '--input_folder',  str(SHARED_FRAMES_DIR),
            '--output_folder', str(emoca_out),
            '--model_name',    model_name,
            '--device',        'cuda',
        ],
        cwd=EMOCA_DIR,
        label='EMOCA',
    )

    out = EMOCA_DIR / 'demo' / 'output'
    n = len(list(out.glob('*.pkl'))) if out.exists() else 0
    print(f'\n  Done. {n} .pkl files → {out}')
    return out


# ---------------------------------------------------------------------------
# Stage 4 — Fuse SMPLest-X + WiLoR + EMOCA
# ---------------------------------------------------------------------------

def stage_fuse(smplestx_dir: Path, wilor_dir: Path,
               emoca_dir: Path, out_dir: Path) -> Path:
    """
    For each SMPLest-X frame:
      - Replace left/right_hand_pose with WiLoR values (if available).
      - Replace expression (50-dim) and jaw_pose (3-dim) with EMOCA (if available).
    Falls back to the original SMPLest-X values when a specialist has no data.

    WiLoR files: <stem>_params.pkl  →  frame id = last int in stem
    EMOCA files: <stem>_params.pkl  →  frame id = last int in stem
      (visualize3.py names output by image stem, same as SMPLest-X)
    """
    _banner('STAGE 4 — Parameter fusion (SMPLest-X + WiLoR + EMOCA)')
    out_dir.mkdir(parents=True, exist_ok=True)

    # Index WiLoR: frame_id → path
    wilor_map = {}
    if wilor_dir.exists():
        for f in os.listdir(wilor_dir):
            if f.endswith('.pkl'):
                fid = _extract_id(f)
                if fid is not None:
                    wilor_map[fid] = wilor_dir / f

    # Index EMOCA: frame_id → path
    # TestData (used by visualize3.py) names images as frame_{N*100} internally,
    # so EMOCA saves "frame_00100_params.pkl" for frame 1, "frame_00200_params.pkl"
    # for frame 2, etc.  Divide the extracted integer by 100 to get the real frame id.
    emoca_map = {}
    if emoca_dir.exists():
        for f in os.listdir(emoca_dir):
            if f.endswith('.pkl'):
                raw = _extract_id(f)
                if raw is not None:
                    emoca_map[raw // 100] = emoca_dir / f

    files = sorted(smplestx_dir.glob('*.pkl')) if smplestx_dir.exists() else []
    if not files:
        print(f'  [ERROR] No SMPLest-X .pkl files found in: {smplestx_dir}')
        return out_dir

    wm = wskip = em = eskip = errs = 0

    for fp in tqdm(files, desc='  Fusing frames'):
        dst = out_dir / fp.name
        try:
            with open(fp, 'rb') as f:
                data = pickle.load(f)

            if not data:
                shutil.copy(fp, dst)
                errs += 1
                continue

            person = data[0]
            fid    = _extract_id(fp.name)

            # ── WiLoR: replace hand poses ──────────────────────────────────
            if fid in wilor_map:
                with open(wilor_map[fid], 'rb') as f:
                    w = pickle.load(f)
                if w.get('right_hand_pose') is not None:
                    person['right_hand_pose'] = w['right_hand_pose']
                if w.get('left_hand_pose') is not None:
                    person['left_hand_pose'] = w['left_hand_pose']
                wm += 1
            else:
                wskip += 1

            # ── EMOCA: replace expression and jaw pose ─────────────────────
            if fid in emoca_map:
                with open(emoca_map[fid], 'rb') as f:
                    e = pickle.load(f)
                if 'exp' in e:
                    person['expression'] = e['exp'].flatten()
                if 'jaw_pose' in e:
                    person['jaw_pose'] = e['jaw_pose'].flatten()
                em += 1
            else:
                eskip += 1

            data[0] = person
            with open(dst, 'wb') as f:
                pickle.dump(data, f)

        except Exception as ex:
            print(f'\n  [ERROR] {fp.name}: {ex}')
            shutil.copy(fp, dst)
            errs += 1

    print(f'\n  WiLoR  — matched: {wm}  skipped (no data): {wskip}')
    print(f'  EMOCA  — matched: {em}  skipped (no data): {eskip}')
    print(f'  Errors: {errs}   Total frames: {len(files)}')
    return out_dir


# ---------------------------------------------------------------------------
# Stage 5 — Zero translation + Savitzky-Golay smooth + Render to MP4
# ---------------------------------------------------------------------------

def stage_render(fused_dir: Path, out_dir: Path, smplx_model: Path,
                 fps: int, viewport: int, win: int, poly: int,
                 env: str) -> Path:
    """
    Calls zero_filter_render.py (in the render env) which:
      1. Zeros out translation (centers avatar).
      2. Applies Savitzky-Golay smoothing to remove temporal jitter.
      3. Renders each frame with pyrender and saves smplest_wilor_emoca.mp4.
    """
    _banner('STAGE 5 — Zero / Smooth / Render')
    out_dir.mkdir(parents=True, exist_ok=True)

    conda_run(
        env,
        ROOT / 'zero_filter_render.py',
        [
            '--input_pkl_folder',     str(fused_dir),
            '--smplx_model_path',     str(smplx_model),
            '--output_dir',           str(out_dir),
            '--video_fps',            str(fps),
            '--viewport_size',        str(viewport),
            '--smooth_window_length', str(win),
            '--smooth_polyorder',     str(poly),
        ],
        cwd=ROOT,
        label='Render',
    )

    video = out_dir / 'smplest_wilor_emoca.mp4'
    if video.exists():
        print(f'\n  Done. Video → {video}')
    return video


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    io = p.add_argument_group('I/O')
    io.add_argument('--video',  required=True,
                    help='Path to the input video file.')
    io.add_argument('--output', required=True,
                    help='Root output directory for fused params and final video.')
    io.add_argument('--name',   default=None,
                    help='Run name used for SMPLest-X output folder '
                         '(default: video filename stem).')

    env = p.add_argument_group('Conda environments')
    env.add_argument('--smplestx_env', default='ubuntu',  metavar='ENV',
                     help='Conda env for SMPLest-X (default: ubuntu).')
    env.add_argument('--wilor_env',    default='ubuntu',  metavar='ENV',
                     help='Conda env for WiLoR (default: ubuntu).')
    env.add_argument('--emoca_env',    default='work38d', metavar='ENV',
                     help='Conda env for EMOCA (default: work38d).')
    env.add_argument('--render_env',   default=None,      metavar='ENV',
                     help='Conda env for rendering '
                          '(needs smplx, pyrender, scipy, torch). '
                          'Defaults to --smplestx_env.')

    mdl = p.add_argument_group('Model options')
    mdl.add_argument('--smplestx_ckpt', default=DEFAULT_SMPLESTX_CKPT, metavar='NAME',
                     help='SMPLest-X checkpoint folder name inside pretrained_models/.')
    mdl.add_argument('--emoca_model',   default=DEFAULT_EMOCA_MODEL,   metavar='NAME',
                     help='EMOCA model folder name inside assets/EMOCA/models/.')
    mdl.add_argument('--smplx_model',   default=DEFAULT_SMPLX_MODEL,   metavar='PATH',
                     help='Path to SMPLX_NEUTRAL.npz used for rendering.')

    vid = p.add_argument_group('Video / render settings')
    vid.add_argument('--fps',           type=int,   default=30,
                     help='FPS for frame extraction and output video.')
    vid.add_argument('--viewport',      type=int,   default=800,
                     help='Render viewport size in pixels (square).')
    vid.add_argument('--smooth_window', type=int,   default=15,
                     help='Savitzky-Golay window length (must be odd).')
    vid.add_argument('--smooth_poly',   type=int,   default=3,
                     help='Savitzky-Golay polynomial order (< window).')

    skip = p.add_argument_group('Skip flags  (resume a partial run)')
    skip.add_argument('--skip_extract',  action='store_true',
                      help='Skip frame extraction — reuse existing frames in MoE/demo/input/.')
    skip.add_argument('--skip_smplestx', action='store_true',
                      help='Skip SMPLest-X inference.')
    skip.add_argument('--skip_wilor',    action='store_true',
                      help='Skip WiLoR inference.')
    skip.add_argument('--skip_emoca',    action='store_true',
                      help='Skip EMOCA inference.')
    skip.add_argument('--skip_fuse',     action='store_true',
                      help='Skip parameter fusion.')
    skip.add_argument('--skip_render',   action='store_true',
                      help='Skip rendering.')

    return p


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args       = build_parser().parse_args()
    video      = Path(args.video).resolve()
    name       = args.name or video.stem
    out_root   = Path(args.output).resolve()
    render_env = args.render_env or args.smplestx_env

    if not video.exists():
        sys.exit(f'[ERROR] Video not found: {video}')

    # ── derived paths ────────────────────────────────────────────────────────
    smplestx_params = SMPLESTX_DIR / 'demo' / 'output_params' / name
    wilor_out       = ROOT / 'demo' / 'result_params_unified' / 'params'
    emoca_out       = EMOCA_DIR / 'demo' / 'output'
    fused_dir       = out_root / 'fused_params'
    render_out      = out_root / 'rendered'

    # ── summary ─────────────────────────────────────────────────────────────
    print('\n' + '#' * 62)
    print('  FULL-BODY RECONSTRUCTION PIPELINE')
    print('#' * 62)
    print(f'  video      : {video}')
    print(f'  name       : {name}')
    print(f'  output     : {out_root}')
    print(f'  frames     : {SHARED_FRAMES_DIR}  (persistent)')
    print(f'  envs       : smplestx={args.smplestx_env}  wilor={args.wilor_env}'
          f'  emoca={args.emoca_env}  render={render_env}')
    print('#' * 62)

    # ── Stage 0: extract frames ──────────────────────────────────────────────
    if not args.skip_extract:
        stage_extract(video, args.fps)
    else:
        n = len(list(SHARED_FRAMES_DIR.glob('*.jpg'))) if SHARED_FRAMES_DIR.exists() else 0
        print(f'\n[SKIP] Stage 0  —  {n} frames already at {SHARED_FRAMES_DIR}')

    # ── Stage 1: SMPLest-X ───────────────────────────────────────────────────
    if not args.skip_smplestx:
        stage_smplestx(video, name, args.smplestx_ckpt, args.smplestx_env)
    else:
        print('\n[SKIP] Stage 1  —  SMPLest-X')

    # ── Stage 2: WiLoR ───────────────────────────────────────────────────────
    if not args.skip_wilor:
        stage_wilor(args.wilor_env)
    else:
        print('\n[SKIP] Stage 2  —  WiLoR')

    # ── Stage 3: EMOCA ───────────────────────────────────────────────────────
    if not args.skip_emoca:
        stage_emoca(args.emoca_model, args.emoca_env)
    else:
        print('\n[SKIP] Stage 3  —  EMOCA')

    # ── Stage 4: fuse ────────────────────────────────────────────────────────
    if not args.skip_fuse:
        stage_fuse(smplestx_params, wilor_out, emoca_out, fused_dir)
    else:
        print('\n[SKIP] Stage 4  —  Fusion')

    # ── Stage 5: render ──────────────────────────────────────────────────────
    if not args.skip_render:
        final_video = stage_render(
            fused_dir, render_out, Path(args.smplx_model),
            args.fps, args.viewport, args.smooth_window, args.smooth_poly,
            render_env,
        )
    else:
        final_video = render_out / 'smplest_wilor_emoca.mp4'
        print('\n[SKIP] Stage 5  —  Render')

    # ── done ─────────────────────────────────────────────────────────────────
    print('\n' + '#' * 62)
    print('  PIPELINE COMPLETE')
    print(f'  Extracted frames : {SHARED_FRAMES_DIR}')
    print(f'  Fused params     : {fused_dir}')
    print(f'  Rendered video   : {final_video}')
    print('#' * 62)
    print()
    print('  NOTE: Extracted frames are kept at:')
    print(f'    {SHARED_FRAMES_DIR}')
    print('  Delete manually when no longer needed.')


if __name__ == '__main__':
    main()
