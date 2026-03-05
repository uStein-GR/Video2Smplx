from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import pickle

# Required for rotation matrix to axis-angle conversion
try:
    from scipy.spatial.transform import Rotation
except ImportError:
    print("\n" + "="*50)
    print("ERROR: SciPy library not found for parameter conversion.")
    print("Please install it: pip install scipy")
    print("="*50 + "\n")

from wilor.models import WiLoR, load_wilor
from wilor.utils import recursive_to
from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.utils.renderer import cam_crop_to_full
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser(description='WiLoR parameter extraction (unified output)')
    parser.add_argument('--img_folder', type=str, default='images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='out_demo', help='Output folder to save parameter .pkl files')
    parser.add_argument('--rescale_factor', type=float, default=2.0, help='Factor for padding the bbox')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png', '*.jpeg'], help='List of file extensions to consider')

    args = parser.parse_args()

    # Validate scipy is available
    try:
        Rotation.from_rotvec([0, 0, 0])
    except NameError:
        print("ERROR: Cannot run because SciPy is not installed. Please run: pip install scipy")
        exit()

    # Load models
    model, model_cfg = load_wilor(checkpoint_path='./pretrained_models/wilor_final.ckpt', cfg_path='./pretrained_models/model_config.yaml')
    detector = YOLO('./pretrained_models/detector.pt')
    
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    detector = detector.to(device)
    model.eval()

    # Create output directory
    os.makedirs(args.out_folder, exist_ok=True)
    print(f"Unified parameters will be saved in: {args.out_folder}")

    img_paths = [img for end in args.file_type for img in Path(args.img_folder).glob(end)]

    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))
        detections = detector(img_cv2, conf=0.3, verbose=False)[0]
        bboxes = []
        is_right = []
        for det in detections:
            Bbox = det.boxes.data.cpu().detach().squeeze().numpy()
            is_right.append(det.boxes.cls.cpu().detach().squeeze().item())
            bboxes.append(Bbox[:4].tolist())
        
        if len(bboxes) == 0:
            continue

        boxes = np.stack(bboxes)
        right = np.stack(is_right)
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=False, num_workers=0)

        frame_params = {
            'right_hand_pose': None,
            'left_hand_pose': None,
            'right_hand_betas': None,
            'left_hand_betas': None,
            'right_hand_global_orient': None,
            'left_hand_global_orient': None,
        }

        for batch in dataloader:
            batch = recursive_to(batch, device)

            with torch.no_grad():
                out = model(batch)

            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                is_right_hand = batch['right'][n].cpu().numpy()

                # Extract raw rotation matrices from model output
                wrist_rotmat = out['pred_mano_params']['global_orient'][n].detach().cpu().numpy()
                finger_rotmat = out['pred_mano_params']['hand_pose'][n].detach().cpu().numpy()
                betas = out['pred_mano_params']['betas'][n].detach().cpu().numpy()

                # Convert rotation matrices to axis-angle
                wrist_aa = Rotation.from_matrix(wrist_rotmat.squeeze(0)).as_rotvec()
                fingers_aa = Rotation.from_matrix(finger_rotmat).as_rotvec()

                if is_right_hand == 1.0:
                    # Right hand: no correction needed
                    frame_params['right_hand_pose'] = fingers_aa.flatten()         # (45,)
                    frame_params['right_hand_betas'] = betas                       # (10,)
                    frame_params['right_hand_global_orient'] = wrist_aa            # (3,)
                else:
                    # Left hand: apply reflection to correct WiLoR's mirroring
                    reflection_vector = np.array([1, -1, -1])  # (x, -y, -z)
                    wrist_aa = wrist_aa * reflection_vector
                    fingers_aa = fingers_aa * reflection_vector
                    
                    frame_params['left_hand_pose'] = fingers_aa.flatten()          # (45,)
                    frame_params['left_hand_betas'] = betas                        # (10,)
                    frame_params['left_hand_global_orient'] = wrist_aa             # (3,)

        if frame_params['right_hand_pose'] is not None or frame_params['left_hand_pose'] is not None:
            pkl_path = os.path.join(args.out_folder, f'{img_fn}_params.pkl')
            with open(pkl_path, 'wb') as f:
                pickle.dump(frame_params, f)
            
            # Print detection summary
            hands_detected = []
            if frame_params['right_hand_pose'] is not None:
                hands_detected.append('Right')
            if frame_params['left_hand_pose'] is not None:
                hands_detected.append('Left')
            print(f"Saved {' + '.join(hands_detected)} hand params → {pkl_path}")

if __name__ == '__main__':
    main()