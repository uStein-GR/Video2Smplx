import argparse
from pathlib import Path
from tqdm import auto
import torch
import pickle
import gdl
from gdl_apps.EMOCA.utils.load import load_model
from gdl.datasets.ImageTestDataset import TestData
from gdl_apps.EMOCA.utils.io import test

# ... (Keep your save_emoca_parameters function exactly as it is) ...
def save_emoca_parameters(vals, output_path):
    # [Insert your existing save_emoca_parameters code here]
    # For brevity, I assume the function definition you provided exists here
    params = {}
    if 'expcode' in vals: params['exp'] = vals['expcode'].cpu().numpy()
    elif 'exp' in vals: params['exp'] = vals['exp'].cpu().numpy()
    if 'posecode' in vals:
        full_pose = vals['posecode'].cpu().numpy()
        params['pose'] = full_pose
        params['global_orient'] = full_pose[:, :3]
        params['jaw_pose'] = full_pose[:, 3:]
    elif 'pose' in vals:
        full_pose = vals['pose'].cpu().numpy()
        params['pose'] = full_pose
        params['global_orient'] = full_pose[:, :3]
        params['jaw_pose'] = full_pose[:, 3:]
    if 'shapecode' in vals: params['shape'] = vals['shapecode'].cpu().numpy()
    elif 'sha' in vals: params['shape'] = vals['sha'].cpu().numpy()
    if 'cam' in vals: params['cam'] = vals['cam'].cpu().numpy()
    if 'lightcode' in vals: params['light'] = vals['lightcode'].cpu().numpy()
    if 'texcode' in vals: params['tex'] = vals['texcode'].cpu().numpy()

    with open(output_path, 'wb') as f:
        pickle.dump(params, f)
    print(f"Saved parameters to {output_path}")
    return params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', type=str, required=True, help="Folder containing input images/frames")
    parser.add_argument('--output_folder', type=str, default="emoca_output", help="Output folder to save the results.")
    parser.add_argument('--model_name', type=str, default='EMOCA_v2_lr_mse_20', help='Name of the model to use.')
    parser.add_argument('--path_to_models', type=str, default=str(Path(gdl.__file__).parents[1] / "assets/EMOCA/models"))
    # Even if this is True, I have removed the logic in the loop below to ensure only PKL is saved
    parser.add_argument('--save_images', type=bool, default=False, help="If true, output images will be saved") 
    parser.add_argument('--device', type=str, default='cuda', help="Device to run on: 'cuda' or 'cpu'")
    
    args = parser.parse_args()

    path_to_models = args.path_to_models
    input_folder = args.input_folder
    model_name = args.model_name
    
    # OUTPUT MODIFICATION: 
    # Use the output_folder directly. Do not append model_name if you want a clean specific folder.
    # If you still want the model name subfolder, keep the next line. If you want EXACTLY the path in your arg, comment it out.
    # output_folder = Path(args.output_folder) / model_name 
    output_folder = Path(args.output_folder) # <--- Modified to use exact path provided in arguments
    
    output_folder.mkdir(parents=True, exist_ok=True)

    # 1) Load the model
    print(f"Loading model: {model_name}...")
    emoca, conf = load_model(path_to_models, model_name, 'detail')
    
    if args.device == 'cuda' and torch.cuda.is_available():
        emoca.cuda()
    else:
        emoca.cpu()
    
    emoca.eval()

    # 2) Create a dataset
    print(f"Processing images from: {input_folder}")
    dataset = TestData(input_folder, face_detector="fan", max_detection=20)

    # 3) Run the model
    for i in auto.tqdm(range(len(dataset))):
        batch = dataset[i]
        vals, visdict = test(emoca, batch)
        
        current_bs = batch["image"].shape[0]

        for j in range(current_bs):
            name = batch["image_name"][j]
            
            # --- MODIFIED SECTION ---
            # Old code created: sample_output_folder = output_folder / name
            
            # New code: Save directly to output_folder with the filename
            param_output_path = output_folder / f"{name}_params.pkl"
            
            # Helper logic:
            # Note: Your save_emoca_parameters saves the *entire batch* (vals) into the PKL. 
            # If your batch size > 1, this file will contain data for all faces in the batch.
            # Assuming you run this on video frames where batch is likely handled or acceptable.
            save_emoca_parameters(vals, param_output_path)

            # Images are NO LONGER saved, regardless of the argument flag, 
            # to ensure you get only the .pkl files.

    print(f"Done. Results saved to {output_folder}")

if __name__ == '__main__':
    main()