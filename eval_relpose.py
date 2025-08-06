import os
import random
import numpy as np
import torch
from tqdm import tqdm
import argparse
from evaluation.datasets.megadepth_valid import MegaDepth_valid
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import closed_form_inverse_se3
from vggt.utils.rotation import mat_to_quat
import cv2
import matplotlib.pyplot as plt 

def todevice(batch, device, callback=None, non_blocking=False):
    ''' Transfer some variables to another device (i.e. GPU, CPU:torch, CPU:numpy).

    batch: list, tuple, dict of tensors or other things
    device: pytorch device or 'numpy'
    callback: function that would be called on every sub-elements.
    '''
    if callback:
        batch = callback(batch)

    if isinstance(batch, dict):
        return {k: todevice(v, device) for k, v in batch.items()}

    if isinstance(batch, (tuple, list)):
        return type(batch)(todevice(x, device) for x in batch)

    x = batch
    if device == 'numpy':
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
    elif x is not None:
        if isinstance(x, np.ndarray):
            x = torch.from_numpy(x)
        if torch.is_tensor(x):
            x = x.to(device, non_blocking=non_blocking)
    return x


to_device = todevice


def get_data_loader(dataset, batch_size, num_workers=8, shuffle=True, drop_last=True, pin_mem=True):
    import torch
    from vggt.utils.misc import get_world_size, get_rank

    # pytorch dataset
    if isinstance(dataset, str):
        dataset = eval(dataset)

    world_size = get_world_size()
    rank = get_rank()

    try:
        sampler = dataset.make_sampler(batch_size, shuffle=shuffle, world_size=world_size,
                                       rank=rank, drop_last=drop_last)
    except (AttributeError, NotImplementedError):
        # not avail for this dataset
        if torch.distributed.is_initialized():
            sampler = torch.utils.data.DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=shuffle, drop_last=drop_last
            )
        elif shuffle:
            sampler = torch.utils.data.RandomSampler(dataset)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

    data_loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_mem,
        drop_last=drop_last,
    )

    return data_loader

def to_numpy(x): return todevice(x, 'numpy')

def reloc_get_rot_err(rot_a, rot_b):
    rot_err = rot_a.T.dot(rot_b)
    rot_err = cv2.Rodrigues(rot_err)[0]
    rot_err = np.reshape(rot_err, (1,3))
    rot_err = np.reshape(np.linalg.norm(rot_err, axis = 1), -1) / np.pi * 180.
    return rot_err[0]

def reloc_get_transl_ang_err(dir_a, dir_b):
    dot_product = np.sum(dir_a * dir_b)
    cos_angle = dot_product / (np.linalg.norm(dir_a) * np.linalg.norm(dir_b))
    angle = np.arccos(cos_angle)
    err = np.degrees(angle)
    return err

torch.backends.cuda.matmul.allow_tf32 = True  # for gpu >= Ampere and pytorch >= 1.12
torch.set_float32_matmul_precision('highest')


def get_args_parser():
    parser = argparse.ArgumentParser(description='evaluation code for relative camera pose estimation')
    
    # test set
    parser.add_argument('--test_dataset', type=str, 
        # default="ScanNet1500(resolution=(224,224), seed=777)")
        default="MegaDepth_valid(resolution=(518,518), seed=777)")
    parser.add_argument('--batch_size', type=int,
        default=1)
    parser.add_argument('--data_root', type=str, default='./data/megadepth1500', 
        help='MegaDepth1500 dataset root')
    parser.add_argument('--num_workers', type=int,
        default=10)
    parser.add_argument('--amp', type=int, default=1,
                                choices=[0, 1], help="Use Automatic Mixed Precision for pretraining")
    return parser


def build_dataset(dataset, batch_size, num_workers, test=False):
    split = ['Train', 'Test'][test]
    print('Building {} data loader for {}'.format(split, dataset))
    loader = get_data_loader(dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             pin_mem=True,
                             shuffle=not (test),
                             drop_last=not (test))
    print('Dataset length: ', len(loader))
    return loader


def load_vggt_model(device):
    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print("Model loaded successfully!")
    return model


def inference_vggt_pair_paths(model, img1_path, img2_path, device, dtype):
    """Run VGGT on a pair of images and return predicted extrinsics."""
    images = load_and_preprocess_images([img1_path, img2_path]).to(device)
    
    # print(images.shape)
    # print("max: ", images[0].max(), images[1].max())
    # print("min: ", images[0].min(), images[1].min())
    # for i in range(images.shape[0]):
    #     plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
    #     plt.show()
    #     plt.savefig(f"./vggt_preprocess_{i}.png")
    #     plt.close()
    
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    with torch.cuda.amp.autocast(dtype=torch.float64):
        extrinsics, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    return extrinsics[0]  # shape (2, 3, 4)

def inference_vggt_pair(model, images, device, dtype):
    """Run VGGT on a pair of images and return predicted extrinsics."""
    images = images.to(device)
        
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    with torch.cuda.amp.autocast(dtype=torch.float64):
        extrinsics, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    return extrinsics[0]  # shape (2, 3, 4)

def calculate_auc_np(r_error, t_error, max_threshold=30):
    """
    Calculate the Area Under the Curve (AUC) for the given error arrays using NumPy.

    Args:
        r_error: numpy array representing R error values (Degree)
        t_error: numpy array representing T error values (Degree)
        max_threshold: Maximum threshold value for binning the histogram

    Returns:
        AUC value and the normalized histogram
    """
    error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
    max_errors = np.max(error_matrix, axis=1)
    bins = np.arange(max_threshold + 1)
    histogram, _ = np.histogram(max_errors, bins=bins)
    num_pairs = float(len(max_errors))
    normalized_histogram = histogram.astype(float) / num_pairs
    return np.mean(np.cumsum(normalized_histogram)), normalized_histogram


def rotation_angle(rot_gt, rot_pred, batch_size=None, eps=1e-15):
    """
    Calculate rotation angle error between ground truth and predicted rotations.

    Args:
        rot_gt: Ground truth rotation matrices
        rot_pred: Predicted rotation matrices
        batch_size: Batch size for reshaping the result
        eps: Small value to avoid numerical issues

    Returns:
        Rotation angle error in degrees
    """
    q_pred = mat_to_quat(rot_pred)
    q_gt = mat_to_quat(rot_gt)

    loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
    err_q = torch.arccos(1 - 2 * loss_q)

    rel_rangle_deg = err_q * 180 / np.pi

    if batch_size is not None:
        rel_rangle_deg = rel_rangle_deg.reshape(batch_size, -1)

    return rel_rangle_deg


def translation_angle(tvec_gt, tvec_pred, batch_size=None, ambiguity=True):
    """
    Calculate translation angle error between ground truth and predicted translations.

    Args:
        tvec_gt: Ground truth translation vectors
        tvec_pred: Predicted translation vectors
        batch_size: Batch size for reshaping the result
        ambiguity: Whether to handle direction ambiguity

    Returns:
        Translation angle error in degrees
    """
    rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
    rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi

    if ambiguity:
        rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())

    if batch_size is not None:
        rel_tangle_deg = rel_tangle_deg.reshape(batch_size, -1)

    return rel_tangle_deg


def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
    """
    Normalize the translation vectors and compute the angle between them.

    Args:
        t_gt: Ground truth translation vectors
        t: Predicted translation vectors
        eps: Small value to avoid division by zero
        default_err: Default error value for invalid cases

    Returns:
        Angular error between translation vectors in radians
    """
    t_norm = torch.norm(t, dim=1, keepdim=True)
    t = t / (t_norm + eps)

    t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
    t_gt = t_gt / (t_gt_norm + eps)

    loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
    err_t = torch.acos(torch.sqrt(1 - loss_t))

    err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
    return err_t


def build_pair_index(N, B=1):
    """
    Build indices for all possible pairs of frames.

    Args:
        N: Number of frames
        B: Batch size

    Returns:
        i1, i2: Indices for all possible pairs
    """
    i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
    i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
    return i1, i2

def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
    """
    Compute rotation and translation errors between predicted and ground truth poses.
    This function assumes the input poses are world-to-camera (w2c) transformations.

    Args:
        pred_se3: Predicted SE(3) transformations (w2c), shape (N, 4, 4)
        gt_se3: Ground truth SE(3) transformations (w2c), shape (N, 4, 4)
        num_frames: Number of frames (N)

    Returns:
        Rotation and translation angle errors in degrees
    """
    pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)    # (0, 1)

    relative_pose_gt = gt_se3[pair_idx_i1].bmm(
        closed_form_inverse_se3(gt_se3[pair_idx_i2])
    )
    relative_pose_pred = pred_se3[pair_idx_i1].bmm(
        closed_form_inverse_se3(pred_se3[pair_idx_i2])
    )

    rel_rangle_deg = rotation_angle(
        relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3]
    )
    rel_tangle_deg = translation_angle(
        relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3]
    )

    return rel_rangle_deg, rel_tangle_deg

def test(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

    # Load VGGT model
    vggt_model = load_vggt_model(device)

    # Load MegaDepth test loader
    # data_loader_test = build_dataset(args.resolution, args.batch_size, args.num_workers)
    
    data_loader_test = {dataset.split('(')[0]: build_dataset(dataset, args.batch_size, args.num_workers, test=True)
                        for dataset in args.test_dataset.split('+')}

    rError, tError = [], []

    for test_name, testset in data_loader_test.items():
        with torch.no_grad():
            for batch in tqdm(testset):
                # batch is a list of length 2 (view1, view2)
                view1, view2 = batch    # (1, 4, 4)

                img1, img2 = view1['img'], view2['img'] 
                img1 += 1
                img2 += 1
                img1 = img1/2
                img2 = img2/2
                # print(img1.shape, img2.shape)
                # print("max: ", img1.max(), img2.max())
                # print("min: ", img1.min(), img2.min())


                images = torch.concatenate([img1, img2], dim = 0)
                
                # for i in range(images.shape[0]):
                #     plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
                #     plt.show()
                #     plt.savefig(f"./vggt_{i}.png")
                #     plt.close()

                gt_view1 = closed_form_inverse_se3(view1['camera_pose'])
                gt_view2 = closed_form_inverse_se3(view2['camera_pose'])

                gt_se3 = torch.cat([gt_view1, gt_view2], dim = 0).to(device)  # (2, 4, 4) in world2cam

                # VGGT inference from image paths
                # img1_path = os.path.join(args.data_root, view1["instance"][0])
                # img2_path = os.path.join(args.data_root, view2["instance"][0])
                # pred_extrinsics = inference_vggt_pair_paths(vggt_model, img1_path, img2_path, device, dtype)     # world2cam in opencv format [R|t]
                
                # VGGT inference
                pred_extrinsics = inference_vggt_pair(vggt_model, images, device, dtype)     # world2cam in opencv format [R|t]

                # Convert to SE3 4x4 matrices
                add_row = torch.tensor([0, 0, 0, 1], device=device).expand(pred_extrinsics.size(0), 1, 4)
                pred_se3 = torch.cat((pred_extrinsics, add_row), dim=1)  # (2, 4, 4)
                
                num_frames = pred_se3.shape[0]     # 2
                rel_rangle_deg, rel_tangle_deg = se3_to_relative_pose_error(pred_se3, gt_se3, num_frames)
                
                Racc_5 = (rel_rangle_deg < 5).float().mean().item()
                Tacc_5 = (rel_tangle_deg < 5).float().mean().item()

                rel_rangle_deg, rel_tangle_deg = rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy()

                if rel_rangle_deg is not None and rel_tangle_deg is not None:
                    rError.extend(rel_rangle_deg)
                    tError.extend(rel_tangle_deg)

    if not rError:
        print(f"No valid sequences found, skipping")
        return

    print('In total {} pairs'.format(len(rError)))

    rError = np.array(rError)
    tError = np.array(tError)

    Auc_20, _ = calculate_auc_np(rError, tError, max_threshold=20)
    Auc_10, _ = calculate_auc_np(rError, tError, max_threshold=10)
    Auc_5, _ = calculate_auc_np(rError, tError, max_threshold=5)
    
    print("AUC@5", Auc_5)
    print("AUC@10", Auc_10)
    print("AUC@20", Auc_20)   
    


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    test(args)

# # ---------------- VGGT Pose Error Functions ---------------- #
# def build_pair_index(N, B=1):
#     i1_, i2_ = torch.combinations(torch.arange(N), 2, with_replacement=False).unbind(-1)
#     i1, i2 = [(i[None] + torch.arange(B)[:, None] * N).reshape(-1) for i in [i1_, i2_]]
#     return i1, i2

# def compare_translation_by_angle(t_gt, t, eps=1e-15, default_err=1e6):
#     t_norm = torch.norm(t, dim=1, keepdim=True)
#     t = t / (t_norm + eps)
#     t_gt_norm = torch.norm(t_gt, dim=1, keepdim=True)
#     t_gt = t_gt / (t_gt_norm + eps)
#     loss_t = torch.clamp_min(1.0 - torch.sum(t * t_gt, dim=1) ** 2, eps)
#     err_t = torch.acos(torch.sqrt(1 - loss_t))
#     err_t[torch.isnan(err_t) | torch.isinf(err_t)] = default_err
#     return err_t

# def rotation_angle(rot_gt, rot_pred):
#     eps = 1e-15
#     q_pred = mat_to_quat(rot_pred)
#     q_gt = mat_to_quat(rot_gt)
#     loss_q = (1 - (q_pred * q_gt).sum(dim=1) ** 2).clamp(min=eps)
#     err_q = torch.arccos(1 - 2 * loss_q)
#     return err_q * 180 / np.pi

# def translation_angle(tvec_gt, tvec_pred, ambiguity=True):
#     rel_tangle_deg = compare_translation_by_angle(tvec_gt, tvec_pred)
#     rel_tangle_deg = rel_tangle_deg * 180.0 / np.pi
#     if ambiguity:
#         rel_tangle_deg = torch.min(rel_tangle_deg, (180 - rel_tangle_deg).abs())
#     return rel_tangle_deg

# def se3_to_relative_pose_error(pred_se3, gt_se3, num_frames):
#     pair_idx_i1, pair_idx_i2 = build_pair_index(num_frames)
#     relative_pose_gt = gt_se3[pair_idx_i1].bmm(closed_form_inverse_se3(gt_se3[pair_idx_i2]))
#     relative_pose_pred = pred_se3[pair_idx_i1].bmm(closed_form_inverse_se3(pred_se3[pair_idx_i2]))
#     rel_rangle_deg = rotation_angle(relative_pose_gt[:, :3, :3], relative_pose_pred[:, :3, :3])
#     rel_tangle_deg = translation_angle(relative_pose_gt[:, :3, 3], relative_pose_pred[:, :3, 3])
#     return rel_rangle_deg.cpu().numpy(), rel_tangle_deg.cpu().numpy()

# def calculate_auc_np(r_error, t_error, max_threshold=30):
#     error_matrix = np.concatenate((r_error[:, None], t_error[:, None]), axis=1)
#     max_errors = np.max(error_matrix, axis=1)
#     bins = np.arange(max_threshold + 1)
#     histogram, _ = np.histogram(max_errors, bins=bins)
#     num_pairs = float(len(max_errors))
#     normalized_histogram = histogram.astype(float) / num_pairs
#     return np.mean(np.cumsum(normalized_histogram))

# # ---------------- MegaDepth Loader Adaptation ---------------- #
# def get_megadepth_sequences(root, num_frames=10):
#     dataset = MegaDepth_valid(resolution=(512,384), seed=777)
#     scenes = {}
#     for idx in range(len(dataset)):
#         pair_line = dataset.scenes[idx].strip().split(" ")
#         for img_path in pair_line:
#             scene_id = "/".join(img_path.split("/")[:2])  # adjust if needed for scene grouping
#             if scene_id not in scenes:
#                 scenes[scene_id] = set()
#             scenes[scene_id].add(img_path)
#     # Convert sets to lists
#     for k in scenes:
#         scenes[k] = list(scenes[k])
#     return scenes

# # ---------------- Main Evaluation ---------------- #
# def main():
#     DATA_ROOT = "./data/megadepth1500"
#     NUM_FRAMES = 10
#     SEED = 0
#     random.seed(SEED)
#     np.random.seed(SEED)
#     torch.manual_seed(SEED)

#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16

#     # Load model
#     model = VGGT()
#     _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
#     model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
#     model.eval()
#     model = model.to(device)
#     print("Model loaded successfully!")

#     scenes = get_megadepth_sequences(DATA_ROOT, num_frames=NUM_FRAMES)

#     all_rerrs = []
#     all_terrs = []

#     for scene_id, img_list in tqdm(scenes.items(), desc="Scenes"):
#         if len(img_list) < NUM_FRAMES:
#             continue

#         # Random sample NUM_FRAMES images
#         sampled_imgs = random.sample(img_list, NUM_FRAMES)

#         # Load GT extrinsics from metadata
#         metadata = dict(np.load(os.path.join(DATA_ROOT, "megadepth_meta_test.npz"), allow_pickle=True))
#         gt_extri = []
#         img_paths = []
#         for img_rel_path in sampled_imgs:
#             img_abs_path = os.path.join(DATA_ROOT, img_rel_path)
#             img_paths.append(img_abs_path)
#             pose_c2w = np.linalg.inv(np.float32(metadata[img_rel_path].item()["pose"]))  # already in OpenCV
#             gt_extri.append(pose_c2w[:3, :])  # 3x4
#         gt_extri = np.stack(gt_extri, axis=0)

#         # Load & preprocess images
#         images = load_and_preprocess_images(img_paths).to(device)

#         import matplotlib.pyplot as plt
#         for i in range(images.shape[0]):
#             plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
#             plt.show()
#             plt.savefig(f"./sample_{i}.png")
#             plt.close()
#         print(images.shape)

#         # VGGT inference
#         with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype):
#             predictions = model(images)
#         with torch.cuda.amp.autocast(dtype=torch.float64):
#             pred_extrinsic, _ = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
#             pred_extrinsic = pred_extrinsic[0]

#         # Add row to make 4x4
#         add_row = torch.tensor([0, 0, 0, 1], device=device).expand(NUM_FRAMES, 1, 4)
#         pred_se3 = torch.cat((pred_extrinsic, add_row), dim=1)
#         gt_se3 = torch.cat((torch.tensor(gt_extri, device=device), add_row), dim=1)

#         # Compute errors
#         r_err, t_err = se3_to_relative_pose_error(pred_se3, gt_se3, NUM_FRAMES)
#         all_rerrs.extend(r_err)
#         all_terrs.extend(t_err)

#     # Convert to numpy
#     all_rerrs = np.array(all_rerrs)
#     all_terrs = np.array(all_terrs)

#     # Reloc3r-style AUC
#     auc5 = calculate_auc_np(all_rerrs, all_terrs, max_threshold=5)
#     auc10 = calculate_auc_np(all_rerrs, all_terrs, max_threshold=10)
#     auc20 = calculate_auc_np(all_rerrs, all_terrs, max_threshold=20)

#     # VGGT-style AUC
#     auc3 = calculate_auc_np(all_rerrs, all_terrs, max_threshold=3)
#     auc15 = calculate_auc_np(all_rerrs, all_terrs, max_threshold=15)
#     auc30 = calculate_auc_np(all_rerrs, all_terrs, max_threshold=30)

#     print("=" * 50)
#     print(f"Reloc3r-style AUC: AUC@5={auc5:.4f}, AUC@10={auc10:.4f}, AUC@20={auc20:.4f}")
#     print(f"VGGT-style AUC:    AUC@3={auc3:.4f}, AUC@5={auc5:.4f}, AUC@15={auc15:.4f}, AUC@30={auc30:.4f}")
#     print("=" * 50)

# if __name__ == "__main__":
#     main()
