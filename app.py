import os
import torch
import trimesh
from accelerate.utils import set_seed
from accelerate import Accelerator
import numpy as np
import gradio as gr
from main import get_args, load_model
from mesh_to_pc import process_mesh_to_pc
import time
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from PIL import Image
import io

args = get_args()
model = load_model(args)

device = torch.device('cuda')
accelerator = Accelerator(
    mixed_precision="fp16",
)
model = accelerator.prepare(model)
model.eval()
print("Model loaded to device")

def wireframe_render(mesh):
    views = [
        (90, 20), (270, 20)
    ]
    mesh.vertices = mesh.vertices[:, [0, 2, 1]]

    bounding_box = mesh.bounds
    center = mesh.centroid
    scale = np.ptp(bounding_box, axis=0).max()

    fig = plt.figure(figsize=(10, 10))

    # Function to render and return each view as an image
    def render_view(mesh, azimuth, elevation):
        ax = fig.add_subplot(111, projection='3d')
        ax.set_axis_off()

        # Extract vertices and faces for plotting
        vertices = mesh.vertices
        faces = mesh.faces

        # Plot faces
        ax.add_collection3d(Poly3DCollection(
            vertices[faces],
            facecolors=(0.8, 0.5, 0.2, 1.0),  # Brownish yellow
            edgecolors='k',
            linewidths=0.5,
        ))

        # Set limits and center the view on the object
        ax.set_xlim(center[0] - scale / 2, center[0] + scale / 2)
        ax.set_ylim(center[1] - scale / 2, center[1] + scale / 2)
        ax.set_zlim(center[2] - scale / 2, center[2] + scale / 2)

        # Set view angle
        ax.view_init(elev=elevation, azim=azimuth)

        # Save the figure to a buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, dpi=300)
        plt.clf()
        buf.seek(0)

        return Image.open(buf)

    # Render each view and store in a list
    images = [render_view(mesh, az, el) for az, el in views]

    # Combine images horizontally
    widths, heights = zip(*(i.size for i in images))
    total_width = sum(widths)
    max_height = max(heights)

    combined_image = Image.new('RGBA', (total_width, max_height))

    x_offset = 0
    for img in images:
        combined_image.paste(img, (x_offset, 0))
        x_offset += img.width

    # Save the combined image
    save_path = f"combined_mesh_view_{int(time.time())}.png"
    combined_image.save(save_path)

    plt.close(fig)
    return save_path

@torch.no_grad()
def do_inference(input_3d, sample_seed=0, do_sampling=False, do_marching_cubes=False):
    set_seed(sample_seed)
    print("Seed value:", sample_seed)

    input_mesh = trimesh.load(input_3d)
    pc_list, mesh_list = process_mesh_to_pc([input_mesh], marching_cubes = do_marching_cubes)
    pc_normal = pc_list[0] # 4096, 6
    mesh = mesh_list[0]
    vertices = mesh.vertices

    pc_coor = pc_normal[:, :3]
    normals = pc_normal[:, 3:]

    bounds = np.array([vertices.min(axis=0), vertices.max(axis=0)])
    # scale mesh and pc
    vertices = vertices - (bounds[0] + bounds[1])[None, :] / 2
    vertices = vertices / (bounds[1] - bounds[0]).max()
    mesh.vertices = vertices
    pc_coor = pc_coor - (bounds[0] + bounds[1])[None, :] / 2
    pc_coor = pc_coor / (bounds[1] - bounds[0]).max()

    mesh.merge_vertices()
    mesh.update_faces(mesh.unique_faces())
    mesh.fix_normals()
    if mesh.visual.vertex_colors is not None:
        orange_color = np.array([255, 165, 0, 255], dtype=np.uint8)

        mesh.visual.vertex_colors = np.tile(orange_color, (mesh.vertices.shape[0], 1))
    else:
        orange_color = np.array([255, 165, 0, 255], dtype=np.uint8)
        mesh.visual.vertex_colors = np.tile(orange_color, (mesh.vertices.shape[0], 1))
    input_save_name = f"processed_input_{int(time.time())}.obj"
    mesh.export(input_save_name)
    input_render_res = wireframe_render(mesh)

    pc_coor = pc_coor / np.abs(pc_coor).max() * 0.9995 # input should be from -1 to 1

    assert (np.linalg.norm(normals, axis=-1) > 0.99).all(), "normals should be unit vectors, something wrong"
    normalized_pc_normal = np.concatenate([pc_coor, normals], axis=-1, dtype=np.float16)

    input = torch.tensor(normalized_pc_normal, dtype=torch.float16, device=device)[None]
    print("Data loaded")

    # with accelerator.autocast():
    with accelerator.autocast():
        outputs = model(input, do_sampling)
    print("Model inference done")
    recon_mesh = outputs[0]

    recon_mesh = recon_mesh[~torch.isnan(recon_mesh[:, 0, 0])]  # nvalid_face x 3 x 3
    vertices = recon_mesh.reshape(-1, 3).cpu()
    vertices_index = np.arange(len(vertices))  # 0, 1, ..., 3 x face
    triangles = vertices_index.reshape(-1, 3)

    artist_mesh = trimesh.Trimesh(vertices=vertices, faces=triangles, force="mesh",
                                 merge_primitives=True)
    artist_mesh.merge_vertices()
    artist_mesh.update_faces(artist_mesh.unique_faces())
    artist_mesh.fix_normals()

    if artist_mesh.visual.vertex_colors is not None:
        orange_color = np.array([255, 165, 0, 255], dtype=np.uint8)

        artist_mesh.visual.vertex_colors = np.tile(orange_color, (artist_mesh.vertices.shape[0], 1))
    else:
        orange_color = np.array([255, 165, 0, 255], dtype=np.uint8)
        artist_mesh.visual.vertex_colors = np.tile(orange_color, (artist_mesh.vertices.shape[0], 1))

    num_faces = len(artist_mesh.faces)

    brown_color = np.array([165, 42, 42, 255], dtype=np.uint8)
    face_colors = np.tile(brown_color, (num_faces, 1))

    artist_mesh.visual.face_colors = face_colors
    # add time stamp to avoid cache
    save_name = f"output_{int(time.time())}.obj"
    artist_mesh.export(save_name)
    output_render = wireframe_render(artist_mesh)
    return input_save_name, input_render_res, save_name, output_render


_HEADER_ = '''
<h2><b>Official ? Gradio Demo</b></h2><h2><a href='https://github.com/buaacyw/MeshAnything' target='_blank'><b>MeshAnything: Artist-Created Mesh Generation with Autoregressive Transformers</b></a></h2>

**MeshAnything** converts any 3D representation into meshes created by human artists, i.e., Artist-Created Meshes (AMs).

Code: <a href='https://github.com/buaacyw/MeshAnything' target='_blank'>GitHub</a>. Arxiv Paper: <a href='https://arxiv.org/abs/2406.10163' target='_blank'>ArXiv</a>.

??????**Important Notes:**
- Gradio doesn't support interactive wireframe rendering currently. For interactive mesh visualization, please use download the obj file and open it with MeshLab or https://3dviewer.net/.
- The input mesh will be normalized to a unit bounding box. The up vector of the input mesh should be +Y for better results. Click **Preprocess with Marching Cubes** if the input mesh is a manually created mesh.
- Limited by computational resources, MeshAnything is trained on meshes with fewer than 800 faces and cannot generate meshes with more than 800 faces. The shape of the input mesh should be sharp enough; otherwise, it will be challenging to represent it with only 800 faces. Thus, feed-forward image-to-3D methods may often produce bad results due to insufficient shape quality.
- For point cloud input, please refer to our github repo <a href='https://github.com/buaacyw/MeshAnything' target='_blank'>GitHub</a>.
'''


_CITE_ = r"""
If MeshAnything is helpful, please help to ? the <a href='https://github.com/buaacyw/MeshAnything' target='_blank'>Github Repo</a>. Thanks!
---
? **License**

S-Lab-1.0 LICENSE. Please refer to the [LICENSE file](https://github.com/buaacyw/GaussianEditor/blob/master/LICENSE.txt) for details.

? **Contact**

If you have any questions, feel free to open a discussion or contact us at <b>yiwen002@e.ntu.edu.sg</b>.

"""
output_model_obj = gr.Model3D(
    label="Generated Mesh (OBJ Format)",
    clear_color=[1, 1, 1, 1],
)
preprocess_model_obj = gr.Model3D(
    label="Processed Input Mesh (OBJ Format)",
    clear_color=[1, 1, 1, 1],
)
input_image_render = gr.Image(
    label="Wireframe Render of Processed Input Mesh",
)
output_image_render = gr.Image(
    label="Wireframe Render of Generated Mesh",
)
with (gr.Blocks() as demo):
    gr.Markdown(_HEADER_)
    with gr.Row(variant="panel"):
        with gr.Column():
            with gr.Row():
                input_3d = gr.Model3D(
                    label="Input Mesh",
                    clear_color=[1,1,1,1],
                )

            with gr.Row():
                with gr.Group():
                    do_marching_cubes = gr.Checkbox(label="Preprocess with Marching Cubes", value=False)
                    do_sampling = gr.Checkbox(label="Random Sampling", value=False)
                    sample_seed = gr.Number(value=0, label="Seed Value", precision=0)

            with gr.Row():
                submit = gr.Button("Generate", elem_id="generate", variant="primary")

            with gr.Row(variant="panel"):
                mesh_examples = gr.Examples(
                    examples=[
                        os.path.join("examples", img_name) for img_name in sorted(os.listdir("examples"))
                    ],
                    inputs=input_3d,
                    outputs=[preprocess_model_obj, input_image_render, output_model_obj, output_image_render],
                    fn=do_inference,
                    cache_examples = False,
                    examples_per_page=10
                )
        with gr.Column():
            with gr.Row():
                input_image_render.render()
            with gr.Row():
                with gr.Tab("OBJ"):
                    preprocess_model_obj.render()
            with gr.Row():
                output_image_render.render()
            with gr.Row():
                with gr.Tab("OBJ"):
                    output_model_obj.render()
            with gr.Row():
                gr.Markdown('''Try click random sampling and different <b>Seed Value</b> if the result is unsatisfying''')

    gr.Markdown(_CITE_)

    mv_images = gr.State()

    submit.click(
        fn=do_inference,
        inputs=[input_3d, sample_seed, do_sampling, do_marching_cubes],
        outputs=[preprocess_model_obj, input_image_render, output_model_obj, output_image_render],
    )

demo.launch(share=True)