import os
import pickle
import shutil
import re
import numpy as np
from tqdm import tqdm


def extract_id(filename):
    """
    Extracts the last integer found in a filename.
    Examples:
      '000001_params.pkl'       -> 1
      'frame_000100_params.pkl' -> 100
    """
    matches = re.findall(r'\d+', filename)
    if matches:
        return int(matches[-1])
    return None


def merge_all():
    """
    3-way fusion: SMPLest-X (body) + WiLoR (hands) + EMOCA (face)

    Pipeline assumptions
    --------------------
    WiLoR  (unified format from demo_params_unified.py):
      - One .pkl per frame, named frame_{N}_params.pkl  (e.g. frame_0001_params.pkl)
      - Already axis-angle, already left-hand reflected (x, -y, -z)
      - Keys: 'right_hand_pose', 'left_hand_pose'  (None if hand not detected)

    EMOCA:
      - One .pkl per frame, named frame_{ID*100}_params.pkl  (e.g. frame_000100_params.pkl)
      - Indexed by dividing the embedded integer by 100 -> real frame number
      - Keys: 'exp' (50-dim), 'jaw_pose' (3-dim axis-angle)

    SMPLest-X base files:
      - Named {base_name}_params.pkl  (e.g. 000001_params.pkl)
      - Stored as a list; person params at index [0]
    """

    # -------------------------------------------------------------------------
    # CONFIGURATION — edit these paths
    # -------------------------------------------------------------------------
    smplestx_params_dir = r""
    wilor_params_dir    = r""
    emoca_params_dir    = r""
    output_dir          = r""
    # -------------------------------------------------------------------------

    os.makedirs(output_dir, exist_ok=True)
    print(f"INFO: Output will be saved to: {output_dir}\n")

    # ------------------------------------------------------------------
    # 1. Index WiLoR files  →  real_frame_id (int) : filepath
    #    WiLoR files are named frame_0001_params.pkl (no ×100 offset),
    #    so extract_id gives the real frame number directly.
    #    e.g. "frame_0001_params.pkl" -> 1
    # ------------------------------------------------------------------
    wilor_map = {}
    for f in os.listdir(wilor_params_dir):
        if f.endswith('.pkl'):
            fid = extract_id(f)
            if fid is not None:
                wilor_map[fid] = os.path.join(wilor_params_dir, f)
    print(f"INFO: Indexed {len(wilor_map)} WiLoR parameter files.")

    # ------------------------------------------------------------------
    # 2. Index EMOCA files  →  real_frame_id (int) : filepath
    #    EMOCA saves frame N as frame_{N*100}_params.pkl
    #    So we divide the extracted integer by 100 to recover N.
    # ------------------------------------------------------------------
    emoca_map = {}
    for f in os.listdir(emoca_params_dir):
        if f.endswith('.pkl'):
            raw_id = extract_id(f)
            if raw_id is not None:
                real_id = raw_id // 100          # e.g. 100 -> 1, 200 -> 2
                emoca_map[real_id] = os.path.join(emoca_params_dir, f)
    print(f"INFO: Indexed {len(emoca_map)} EMOCA parameter files.")

    # ------------------------------------------------------------------
    # 3. Process SMPLest-X base files
    # ------------------------------------------------------------------
    smplestx_files = sorted([f for f in os.listdir(smplestx_params_dir) if f.endswith('.pkl')])

    if not smplestx_files:
        print(f"ERROR: No .pkl files found in: {smplestx_params_dir}")
        return

    print(f"INFO: Found {len(smplestx_files)} SMPLest-X parameter files to process.\n")

    wilor_match   = 0
    wilor_skip    = 0
    emoca_match   = 0
    emoca_skip    = 0
    error_count   = 0

    for smplestx_filename in tqdm(smplestx_files, desc="Fusing Parameters"):
        base_name         = smplestx_filename.replace('_params.pkl', '')
        smplestx_filepath = os.path.join(smplestx_params_dir, smplestx_filename)
        output_filepath   = os.path.join(output_dir, smplestx_filename)

        try:
            # --- Load SMPLest-X ---
            with open(smplestx_filepath, 'rb') as f:
                smplestx_data = pickle.load(f)

            if not smplestx_data:
                shutil.copy(smplestx_filepath, output_filepath)
                error_count += 1
                continue

            person_params = smplestx_data[0]

            # ==============================================================
            # STEP A — Replace hand poses with WiLoR (unified format)
            # ==============================================================
            base_id = extract_id(smplestx_filename)   # e.g. "000001_params.pkl" -> 1

            if base_id in wilor_map:
                with open(wilor_map[base_id], 'rb') as f:
                    wilor_data = pickle.load(f)

                if wilor_data.get('right_hand_pose') is not None:
                    person_params['right_hand_pose'] = wilor_data['right_hand_pose']

                if wilor_data.get('left_hand_pose') is not None:
                    person_params['left_hand_pose'] = wilor_data['left_hand_pose']

                wilor_match += 1
            else:
                wilor_skip += 1

            # ==============================================================
            # STEP B — Replace expression & jaw_pose with EMOCA
            # ==============================================================
            if base_id in emoca_map:
                with open(emoca_map[base_id], 'rb') as f:
                    emoca_data = pickle.load(f)

                # Expression — 50-dim, write as-is
                if 'exp' in emoca_data:
                    person_params['expression'] = emoca_data['exp'].flatten()

                # Jaw pose — 3-dim axis-angle
                if 'jaw_pose' in emoca_data:
                    person_params['jaw_pose'] = emoca_data['jaw_pose'].flatten()

                emoca_match += 1
            else:
                emoca_skip += 1

            # --- Save fused result ---
            smplestx_data[0] = person_params
            with open(output_filepath, 'wb') as f:
                pickle.dump(smplestx_data, f)

        except Exception as e:
            print(f"\nERROR processing {smplestx_filename}: {e} — copying original as fallback.")
            shutil.copy(smplestx_filepath, output_filepath)
            error_count += 1

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 45)
    print("FUSION SUMMARY")
    print("-" * 45)
    print(f"  WiLoR  matched & fused : {wilor_match}")
    print(f"  WiLoR  no data (kept)  : {wilor_skip}")
    print(f"  EMOCA  matched & fused : {emoca_match}")
    print(f"  EMOCA  no data (kept)  : {emoca_skip}")
    print(f"  Errors / empty         : {error_count}")
    print(f"  Total processed        : {len(smplestx_files)}")
    print("=" * 45)


if __name__ == '__main__':
    merge_all()
    print("\n--- 3-way parameter fusion complete! ---")