# visualizer.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from human_models.human_models import SMPLX

# (SKELETON_CONNECTIONS and plot_skeleton function remain the same)
SKELETON_CONNECTIONS = [
    # Legs
    (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 14),
    (5, 15), (5, 16), (6, 17), (6, 18), (6, 19),

    # Spine and Head
    (0, 7), (7, 20), (7, 21), (7, 24), (22, 23), (20, 22), (21, 23),

    # Arms
    (7, 8), (7, 9), (8, 10), (9, 11), (10, 12), (11, 13),

    # Left Hand
    (12, 29), (12, 33), (12, 37), (12, 41), (12, 25),
    (25, 26), (26, 27), (27, 28), (29, 30), (30, 31),
    (31, 32), (33, 34), (34, 35), (35, 36), (37, 38),
    (38, 39), (39, 40), (41, 42), (42, 43), (43, 44),

    # Right Hand
    (13, 49), (13, 53), (13, 57), (13, 61), (13, 45),
    (45, 46), (46, 47), (47, 48), (49, 50), (50, 51),
    (51, 52), (53, 54), (54, 55), (55, 56), (57, 58),
    (58, 59), (59, 60), (61, 62), (62, 63), (63, 64),

    (67,68), # face eyeballs
    (69,78), (70,77), (71,76), (72,75), (73,74), # face eyebrow
    (83,87), (84,86), # face below nose
    (88,97), (89,96), (90,95), (91,94), (92,99), (93,98), # face eyes
    (100,106), (101,105), (102,104), (107,111), (108,110), # face mouth
    (112,116), (113,115), (117,119), # face lip
    (120,136), (121,135), (122,134), (123,133), (124,132), (125,131), (126,130), (127,129) # face contours
]

def plot_skeleton(ax, joints_3d, connections, color, label, draw_lines=True):
    """Helper function to plot a 3D skeleton, with an option to disable lines."""
    if draw_lines:
        for start_joint_idx, end_joint_idx in connections:
            # Check if indices are valid for the given joint array
            if start_joint_idx < len(joints_3d) and end_joint_idx < len(joints_3d):
                x_coords = [joints_3d[start_joint_idx, 0], joints_3d[end_joint_idx, 0]]
                y_coords = [joints_3d[start_joint_idx, 1], joints_3d[end_joint_idx, 1]]
                z_coords = [joints_3d[start_joint_idx, 2], joints_3d[end_joint_idx, 2]]
                ax.plot(x_coords, y_coords, z_coords, c=color)

    # Always draw the joint points
    ax.scatter(joints_3d[:, 0], joints_3d[:, 1], joints_3d[:, 2], c=color, marker='o', label=label)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    ax.set_box_aspect([np.ptp(joints_3d[:, 0]), np.ptp(joints_3d[:, 1]), np.ptp(joints_3d[:, 2])])

def visualize_hypothesis(model_output, smplx_layer_neutral):
    """
    Visualizes the model's output to validate the data flow hypothesis.
    Compares the direct output, a recalculation with translation, and without.
    """
    # Ensure the SMPLX layer is on the same device as the model output tensors.
    device = model_output['smplx_root_pose'].device
    smplx_layer_neutral = smplx_layer_neutral.to(device)

    # 1. Get the 3D joints directly from the model's output
    joints_from_output = model_output['smplx_joint_cam'].detach().cpu().numpy()[0]

    # --- Prepare parameters for recalculation ---
    batch_size = model_output['smplx_root_pose'].shape[0]
    common_params = {
        'global_orient': model_output['smplx_root_pose'],
        'body_pose': model_output['smplx_body_pose'],
        'left_hand_pose': model_output['smplx_lhand_pose'],
        'right_hand_pose': model_output['smplx_rhand_pose'],
        'jaw_pose': model_output['smplx_jaw_pose'],
        'betas': model_output['smplx_shape'],
        'expression': model_output['smplx_expr'],
        'leye_pose': torch.zeros((batch_size, 3), device=device),
        'reye_pose': torch.zeros((batch_size, 3), device=device)
    }

    # 2. Recalculate WITH translation (for the final position)
    with torch.no_grad():
        output_with_transl = smplx_layer_neutral(
            transl=model_output['cam_trans'], **common_params)
        joints_with_transl = output_with_transl.joints.detach().cpu().numpy()[0]

    # 3. Recalculate WITHOUT translation (to match the direct output)
    with torch.no_grad():
        output_without_transl = smplx_layer_neutral(
            transl=torch.zeros_like(model_output['cam_trans']), **common_params)
        joints_without_transl = output_without_transl.joints.detach().cpu().numpy()[0]


    # --- Plot all three for comparison ---
    fig = plt.figure(figsize=(18, 6))
    num_joints_in_output = joints_from_output.shape[0]

    # Plot 1: Direct output with correct lines
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.set_title("Direct Output ('smplx_joint_cam')")
    plot_skeleton(ax1, joints_from_output, SKELETON_CONNECTIONS, 'blue', 'Direct Output', draw_lines=True)

    # Plot 2: Recalculated WITHOUT lines to show matching point cloud
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.set_title("Recalculated (No Translation)")
    plot_skeleton(ax2, joints_without_transl[:num_joints_in_output], SKELETON_CONNECTIONS, 'green', 'Recalculated (No Transl)', draw_lines=False)

    # Plot 3: Recalculated WITH lines to show matching point cloud
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.set_title("Recalculated (With Translation)")
    plot_skeleton(ax3, joints_with_transl[:num_joints_in_output], SKELETON_CONNECTIONS, 'red', 'Recalculated (With Transl)', draw_lines=False)

    plt.tight_layout()
    plt.show()

    difference = np.mean(np.abs(joints_from_output - joints_without_transl[:num_joints_in_output]))
    print(f"\nMean difference between Direct Output and Recalculated (No Translation): {difference:.8f}")
