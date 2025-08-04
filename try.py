import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
import time

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

model = VGGT()
model.load_state_dict(torch.load('./ckpt/model.pt'))
model = model.to(device)

# Load and preprocess example images (replace with your own image paths)
image_names = ["../data/cambridge/GreatCourt/seq1/frame00001.png", 
                "../data/cambridge/GreatCourt/seq1/frame00002.png", 
                "../data/cambridge/GreatCourt/seq1/frame00003.png",
                "../data/cambridge/GreatCourt/seq1/frame00004.png"
                ]  
images = load_and_preprocess_images(image_names).to(device)
images = images.unsqueeze(0)   # (1, 3, )

torch.cuda.synchronize()
start_time = time.time()
with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        print(images.shape)
        # Predict attributes including cameras, depth maps, and point maps.
        # print(predictions['pose_enc'])

        # to get the featuyres from the transformer only
        predictions = model(images, return_feats=True)
        
        print(predictions[0][23].shape, predictions[1])    # list, int of shapes and value: ([1, 4, 782, 2048]), 5 idx

torch.cuda.synchronize()
total_time = time.time() - start_time

print(f"Total inference time (s): {total_time:.4f}")



# # code to get similarity transformation between two sequences

# import numpy as np
# from scipy.optimize import minimize

# from utils.rotation_orthogonalization import orthogonalize_rotation_matrix


# def estimate_similarity_transform(poses1, poses2):
#     scale_init, T_init = estimate_similarity_transform_linear(poses1, poses2)
#     R_init = T_init[:3, :3]
#     t_init = T_init[:3, 3]

#     q_opt = optimize_rotation(poses1, poses2, R_init)
#     R_opt = quaternion_to_rotation_matrix(q_opt)

    
#     scale_opt = optimize_scale(poses1, poses2, R_opt)

    
#     t_opt = optimize_translation(poses1, poses2, R_opt, scale_opt)

    
#     T_opt = np.eye(4)
#     T_opt[:3, :3] = R_opt
#     T_opt[:3, 3] = t_opt

#     return scale_opt, T_opt