import sys
import os
os.environ['PYOPENGL_PLATFORM'] = 'win32'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import os.path as osp
import argparse
import numpy as np
import torchvision.transforms as transforms
import torch.backends.cudnn as cudnn
import torch
import cv2
import datetime
from tqdm import tqdm
from pathlib import Path
from human_models.human_models import SMPLX
from ultralytics import YOLO
from main.base import Tester
from main.config import Config
from utils.data_utils import load_img, process_bbox, generate_patch_image
from utils.visualization_utils import render_mesh
from utils.inference_utils import non_max_suppression
import pickle # Pickle module for saving data
import shutil
from visualizer import visualize_hypothesis # Visualization fn for SMPL-X parameters

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_gpus', type=int, dest='num_gpus')
    parser.add_argument('--file_name', type=str, default='test')
    parser.add_argument('--ckpt_name', type=str, default='model_dump')
    parser.add_argument('--start', type=str, default=1)
    parser.add_argument('--end', type=str, default=1)
    parser.add_argument('--video', type=str, default=None, help='Video file in demo/ folder (e.g. test_vdo.mp4). Extracts frames automatically.')
    parser.add_argument('--multi_person', action='store_true')
    parser.add_argument('--save_params', action='store_true', help='Set to true to save SMPL-X parameters')
    parser.add_argument('--params_only', action='store_true', help='If set, skip rendering and only save parameters (requires --save_params)')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # --- Extract video frames if --video is provided ---
    _frames_to_cleanup = None
    if args.video:
        video_path = osp.join('demo', args.video)
        args.file_name = osp.splitext(args.video)[0]
        _frames_to_cleanup = osp.join('demo', 'input_frames', args.file_name)
        os.makedirs(_frames_to_cleanup, exist_ok=True)
        print(f"Extracting frames from {video_path} ...")
        cap = cv2.VideoCapture(video_path)
        frame_idx = 1
        while True:
            ret, frm = cap.read()
            if not ret:
                break
            cv2.imwrite(osp.join(_frames_to_cleanup, f'{frame_idx:06d}.jpg'), frm)
            frame_idx += 1
        cap.release()
        args.end = frame_idx - 1
        print(f"Extracted {args.end} frames.")
    # ---

    cudnn.benchmark = True

    # init config
    time_str = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    root_dir = Path(__file__).resolve().parent.parent
    config_path = osp.join('./pretrained_models', args.ckpt_name, 'config_base.py')
    cfg = Config.load_config(config_path)
    checkpoint_path = osp.join('./pretrained_models', args.ckpt_name, f'{args.ckpt_name}.pth.tar')
    img_folder = osp.join(root_dir, 'demo', 'input_frames', args.file_name)
    output_folder = osp.join(root_dir, 'demo', 'output_frames', args.file_name)
    if not args.params_only:
        os.makedirs(output_folder, exist_ok=True)
    exp_name = f'inference_{args.file_name}_{args.ckpt_name}_{time_str}'

    new_config = {
        "model": {
            "pretrained_model_path": checkpoint_path,
        },
        "log":{
            'exp_name':  exp_name,
            'log_dir': osp.join(root_dir, 'outputs', exp_name, 'log'),  
            }
    }
    cfg.update_config(new_config)
    cfg.prepare_log()
    
    # init human models
    smpl_x = SMPLX(cfg.model.human_model_path)

    # init tester
    demoer = Tester(cfg)
    demoer.logger.info(f"Using 1 GPU.")
    demoer.logger.info(f'Inference [{args.file_name}] with [{cfg.model.pretrained_model_path}].')
    demoer._make_model()

    # init detector
    bbox_model = getattr(cfg.inference.detection, "model_path", 
                        './pretrained_models/yolov8x.pt')
    detector = YOLO(bbox_model)

    start = int(args.start)
    end = int(args.end) + 1

    # --- Create a directory to save parameters ---
    params_output_folder = osp.join(root_dir, 'demo', 'output_params', args.file_name)
    os.makedirs(params_output_folder, exist_ok=True)
    # ---------------------------------------------

    for frame in tqdm(range(start, end)):
        
        # prepare input image
        img_path =osp.join(img_folder, f'{int(frame):06d}.jpg')

        transform = transforms.ToTensor()
        original_img = load_img(img_path)
        vis_img = original_img.copy()
        original_img_height, original_img_width = original_img.shape[:2]
        
        # detection, xyxy
        yolo_bbox = detector.predict(original_img, 
                                device='cuda', 
                                classes=0, # Changed from 00 to 0 for person class
                                conf=cfg.inference.detection.conf, 
                                save=cfg.inference.detection.save, 
                                verbose=cfg.inference.detection.verbose
                                    )[0].boxes.xyxy.detach().cpu().numpy()

        if len(yolo_bbox)<1:
            num_bbox = 0
            # --- If no person detected, still save empty parameters if requested ---
            if args.save_params:
                param_filename = osp.join(params_output_folder, f'{int(frame):06d}_params.pkl')
                with open(param_filename, 'wb') as f:
                    pickle.dump([], f) # Save an empty list to indicate no detection for this frame
                print(f"No person detected. Saved empty parameters for frame {frame}.")
            # ---------------------------------------------------------------------
        elif not args.multi_person:
            # Select the largest bbox if multi_person is false
            largest_bbox_idx = np.argmax((yolo_bbox[:, 2] - yolo_bbox[:, 0]) * (yolo_bbox[:, 3] - yolo_bbox[:, 1]))
            yolo_bbox = [yolo_bbox[largest_bbox_idx]] # Make sure it's a list for the loop
            num_bbox = 1
        else:
            # keep bbox by NMS with iou_thr
            yolo_bbox = non_max_suppression(yolo_bbox, cfg.inference.detection.iou_thr)
            num_bbox = len(yolo_bbox)

        frame_all_person_params = [] # Initialize list to store params for all detected persons in this frame

        # loop all detected bboxes
        for bbox_id in range(num_bbox):
            current_yolo_bbox = yolo_bbox[bbox_id]
            yolo_bbox_xywh = np.zeros((4))
            yolo_bbox_xywh[0] = current_yolo_bbox[0]
            yolo_bbox_xywh[1] = current_yolo_bbox[1]
            yolo_bbox_xywh[2] = abs(current_yolo_bbox[2] - current_yolo_bbox[0])
            yolo_bbox_xywh[3] = abs(current_yolo_bbox[3] - current_yolo_bbox[1])
            
            # xywh
            bbox = process_bbox(bbox=yolo_bbox_xywh, 
                                img_width=original_img_width, 
                                img_height=original_img_height, 
                                input_img_shape=cfg.model.input_img_shape, 
                                ratio=getattr(cfg.data, "bbox_ratio", 1.25))                
            img, _, _ = generate_patch_image(cvimg=original_img,  
                                                bbox=bbox, 
                                                scale=1.0, 
                                                rot=0.0, 
                                                do_flip=False, 
                                                out_shape=cfg.model.input_img_shape)
                
            img = transform(img.astype(np.float32))/255
            img = img.cuda()[None,:,:,:]
            inputs = {'img': img}
            targets = {}
            meta_info = {}

            # mesh recovery
            with torch.no_grad():
                out = demoer.model(inputs, targets, meta_info, 'test')

            # --- Extract and concatenate SMPL-X Parameters into a single vector ---
            if args.save_params:
                
                # print(f"\nDisplaying visualization for Frame {frame}, Person {bbox_id+1}...")
                # visualize_hypothesis(out, smpl_x.layer['neutral'])

                global_orient = out['smplx_root_pose'].detach().cpu().numpy()[0]
                body_pose = out['smplx_body_pose'].detach().cpu().numpy()[0]
                left_hand_pose = out['smplx_lhand_pose'].detach().cpu().numpy()[0]
                right_hand_pose = out['smplx_rhand_pose'].detach().cpu().numpy()[0]
                jaw_pose = out['smplx_jaw_pose'].detach().cpu().numpy()[0]
                betas = out['smplx_shape'].detach().cpu().numpy()[0]
                expression = out['smplx_expr'].detach().cpu().numpy()[0]
                transl = out['cam_trans'].detach().cpu().numpy()[0]

                # Concatenate them in the order specified in SMPLest_X.py global_orient (3) + body_pose (63) + left_hand_pose (45) + right_hand_pose (45) + jaw_pose (3) + betas (10) + expression (10) + transl (3) = 182
                smplx_param_vector = np.concatenate([
                    global_orient,
                    body_pose,
                    left_hand_pose,
                    right_hand_pose,
                    jaw_pose,
                    betas,
                    expression,
                    transl
                ], axis=-1)
                
                print(f"\n--- SMPL-X Parameters for Frame {frame}, Person {bbox_id+1} ---")
                print(f"Shape of smplx_param_vector: {smplx_param_vector.shape}")
                print(f"First 10 values of smplx_param_vector: {smplx_param_vector[:10]}")
                print(f"Last 10 values of smplx_param_vector (transl, expression, betas): {smplx_param_vector[-10:]}") # Added clarity
                print("--------------------------------------------------")

                # Store the extracted parameters
                person_params = {
                    'smplx_param_vector': smplx_param_vector,
                    # You might also want to store the individual components if needed for separate processing
                    'global_orient': global_orient,
                    'body_pose': body_pose,
                    'left_hand_pose': left_hand_pose,
                    'right_hand_pose': right_hand_pose,
                    'jaw_pose': jaw_pose,
                    'betas': betas,
                    'expression': expression,
                    'transl': transl,
                }
                frame_all_person_params.append(person_params)
            # -----------------------------------------------------------

            mesh = out['smplx_mesh_cam'].detach().cpu().numpy()[0]

            if not args.params_only:
                # render mesh
                focal = [cfg.model.focal[0] / cfg.model.input_body_shape[1] * bbox[2], 
                         cfg.model.focal[1] / cfg.model.input_body_shape[0] * bbox[3]]
                princpt = [cfg.model.princpt[0] / cfg.model.input_body_shape[1] * bbox[2] + bbox[0], 
                           cfg.model.princpt[1] / cfg.model.input_body_shape[0] * bbox[3] + bbox[1]]
                
                # draw the bbox on img
                vis_img = cv2.rectangle(vis_img, (int(current_yolo_bbox[0]), int(current_yolo_bbox[1])), 
                                        (int(current_yolo_bbox[2]), int(current_yolo_bbox[3])), (0, 255, 0), 1)
                # draw mesh
                vis_img = render_mesh(vis_img, mesh, smpl_x.face, {'focal': focal, 'princpt': princpt}, mesh_as_vertices=True)

        # --- Save parameters for the current frame (after looping all bboxes) ---
        if args.save_params: # Save even if num_bbox is 0 (will be empty list)
            param_filename = osp.join(params_output_folder, f'{int(frame):06d}_params.pkl')
            with open(param_filename, 'wb') as f:
                pickle.dump(frame_all_person_params, f) # Save the list of params for all persons in this frame
            if num_bbox > 0:
                print(f"Saved SMPL-X parameters for {num_bbox} person(s) in frame {frame} to {param_filename}")
        # ---------------------------------------------------------------------

        # save rendered image
        if not args.params_only:
            frame_name = os.path.basename(img_path)
            cv2.imwrite(os.path.join(output_folder, frame_name), vis_img[:, :, ::-1])

    # --- Cleanup extracted frames ---
    if _frames_to_cleanup:
        shutil.rmtree(_frames_to_cleanup, ignore_errors=True)
        print(f"Cleaned up extracted frames: {_frames_to_cleanup}")
        
        input_frames_dir = osp.dirname(_frames_to_cleanup)  # demo/input_frames/
        if osp.exists(input_frames_dir) and not os.listdir(input_frames_dir):
            os.rmdir(input_frames_dir)
            print(f"Removed empty folder: {input_frames_dir}")
    # ---


if __name__ == "__main__":
    main()