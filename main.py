import os, argparse, importlib
import torch
import time
import trimesh
import numpy as np
from MeshAnything.models.meshanything import MeshAnything
import datetime
from accelerate import Accelerator
from accelerate.utils import set_seed
from accelerate.utils import DistributedDataParallelKwargs
from safetensors import safe_open
from mesh_to_pc import process_mesh_to_pc
from huggingface_hub import hf_hub_download

class Dataset:
    def __init__(self, input_type, input_list, mc=False):
        super().__init__()
        self.data = []
        if input_type == 'pc_normal':
            for input_path in input_list:
                # load npy
                cur_data = np.load(input_path)
                # sample 4096
                assert cur_data.shape[0] >= 4096, "input pc_normal should have at least 4096 points"
                idx = np.random.choice(cur_data.shape[0], 4096, replace=False)
                cur_data = cur_data[idx]
                self.data.append({'pc_normal': cur_data, 'uid': input_path.split('/')[-1].split('.')[0]})

        elif input_type == 'mesh':
            mesh_list = []
            for input_path in input_list:
                # load ply
                cur_data = trimesh.load(input_path)
                mesh_list.append(cur_data)
            if mc:
                print("First Marching Cubes and then sample point cloud, need several minutes...")
            pc_list, _ = process_mesh_to_pc(mesh_list, marching_cubes=mc)
            for input_path, cur_data in zip(input_list, pc_list):
                self.data.append({'pc_normal': cur_data, 'uid': input_path.split('/')[-1].split('.')[0]})
        print(f"dataset total data samples: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data_dict = {}
        data_dict['pc_normal'] = self.data[idx]['pc_normal']
        # normalize pc coor
        pc_coor = data_dict['pc_normal'][:, :3]
        normals = data_dict['pc_normal'][:, 3:]
        bounds = np.array([pc_coor.min(axis=0), pc_coor.max(axis=0)])
        pc_coor = pc_coor - (bounds[0] + bounds[1])[None, :] / 2
        pc_coor = pc_coor / np.abs(pc_coor).max() * 0.9995
        assert (np.linalg.norm(normals, axis=-1) > 0.99).all(), "normals should be unit vectors, something wrong"
        data_dict['pc_normal'] = np.concatenate([pc_coor, normals], axis=-1, dtype=np.float16)
        data_dict['uid'] = self.data[idx]['uid']

        return data_dict

def get_args():
    parser = argparse.ArgumentParser("MeshAnything", add_help=False)

    parser.add_argument('--llm', default="facebook/opt-350m", type=str)
    parser.add_argument('--input_dir', default=None, type=str)
    parser.add_argument('--input_path', default=None, type=str)

    parser.add_argument('--out_dir', default="inference_out", type=str)
    parser.add_argument('--pretrained_weights', default="MeshAnything_350m.pth", type=str)

    parser.add_argument(
        '--input_type',
        choices=['mesh','pc_normal'],
        default='pc',
        help="Type of the asset to process (default: pc)"
    )

    parser.add_argument("--codebook_size", default=8192, type=int)
    parser.add_argument("--codebook_dim", default=1024, type=int)

    parser.add_argument("--n_max_triangles", default=800, type=int)

    parser.add_argument("--batchsize_per_gpu", default=1, type=int)
    parser.add_argument("--seed", default=0, type=int)

    parser.add_argument("--mc", default=False, action="store_true")
    parser.add_argument("--sampling", default=False, action="store_true")

    args = parser.parse_args()
    return args

def load_model(args):
    model = MeshAnything(args)
    print("load model over!!!")

    ckpt_path = hf_hub_download(
        repo_id="Yiwen-ntu/MeshAnything",
        filename="MeshAnything_350m.pth",
    )
    tensors = {}
    with safe_open(ckpt_path, framework="pt", device=0) as f:
        for k in f.keys():
            tensors[k] = f.get_tensor(k)

    model.load_state_dict(tensors, strict=True)
    print("load weights over!!!")
    return model
if __name__ == "__main__":
    args = get_args()

    cur_time = datetime.datetime.now().strftime("%d_%H-%M-%S")
    checkpoint_dir = os.path.join(args.out_dir, cur_time)
    os.makedirs(checkpoint_dir, exist_ok=True)
    kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        mixed_precision="fp16",
        project_dir=checkpoint_dir,
        kwargs_handlers=[kwargs]
    )

    model = load_model(args)
    # create dataset
    if args.input_dir is not None:
        input_list = sorted(os.listdir(args.input_dir))
        # only ply, obj or npy
        if args.input_type == 'pc_normal':
            input_list = [os.path.join(args.input_dir, x) for x in input_list if x.endswith('.npy')]
        else:
            input_list = [os.path.join(args.input_dir, x) for x in input_list if x.endswith('.ply') or x.endswith('.obj') or x.endswith('.npy')]
        set_seed(args.seed)
        dataset = Dataset(args.input_type, input_list, args.mc)
    elif args.input_path is not None:
        set_seed(args.seed)
        dataset = Dataset(args.input_type, [args.input_path], args.mc)
    else:
        raise ValueError("input_dir or input_path must be provided.")

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batchsize_per_gpu,
        drop_last = False,
        shuffle = False,
    )

    if accelerator.state.num_processes > 1:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    dataloader, model = accelerator.prepare(dataloader, model)
    begin_time = time.time()
    print("Generation Start!!!")
    with accelerator.autocast():
        for curr_iter, batch_data_label in enumerate(dataloader):
            curr_time = time.time()
            outputs = model(batch_data_label['pc_normal'], sampling=args.sampling)
            batch_size = outputs.shape[0]
            device = outputs.device

            for batch_id in range(batch_size):
                recon_mesh = outputs[batch_id]
                recon_mesh = recon_mesh[~torch.isnan(recon_mesh[:, 0, 0])]  # nvalid_face x 3 x 3
                vertices = recon_mesh.reshape(-1, 3).cpu()
                vertices_index = np.arange(len(vertices))  # 0, 1, ..., 3 x face
                triangles = vertices_index.reshape(-1, 3)

                scene_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, force="mesh",
                                             merge_primitives=True)
                scene_mesh.merge_vertices()
                scene_mesh.update_faces(scene_mesh.unique_faces())
                scene_mesh.fix_normals()
                save_path = os.path.join(checkpoint_dir, f'{batch_data_label["uid"][batch_id]}_gen.obj')
                num_faces = len(scene_mesh.faces)
                brown_color = np.array([255, 165, 0, 255], dtype=np.uint8)
                face_colors = np.tile(brown_color, (num_faces, 1))

                scene_mesh.visual.face_colors = face_colors
                scene_mesh.export(save_path)
                print(f"{save_path} Over!!")
    end_time = time.time()
    print(f"Total time: {end_time - begin_time}")