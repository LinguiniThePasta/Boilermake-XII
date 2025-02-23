from pathlib import Path

import modal

MODEL_TYPE = "facebook/sam2-hiera-large"
SAM2_GIT_SHA = "c2ec8e14a185632b0a5d8b161928ceb50197eddc"  # pin commit! research code is fragile

image = (
    modal.Image.debian_slim(python_version="3.10")
    .apt_install("git", "wget", "python3-opencv", "ffmpeg")
    .pip_install(
        "torch~=2.4.1",
        "torchvision==0.19.1",
        "opencv-python==4.10.0.84",
        "pycocotools~=2.0.8",
        "matplotlib~=3.9.2",
        "onnxruntime==1.19.2",
        "onnx==1.17.0",
        "huggingface_hub==0.25.2",
        "ffmpeg-python==0.2.0",
        f"git+https://github.com/facebookresearch/sam2.git@{SAM2_GIT_SHA}",
    )
)
app = modal.App("sam2-app", image=image)

video_vol = modal.Volume.from_name("sam2-inputs", create_if_missing=True)
cache_vol = modal.Volume.from_name("hf-hub-cache", create_if_missing=True)
cache_dir = "/cache"


@app.cls(
    image=image.env({"HF_HUB_CACHE": cache_dir}),
    volumes={"/root/videos": video_vol, cache_dir: cache_vol},
    gpu="A100",
)
class Model:
    @modal.enter()
    def initialize_model(self):
        """Download and initialize model."""
        from sam2.sam2_video_predictor import SAM2VideoPredictor

        self.video_predictor = SAM2VideoPredictor.from_pretrained(MODEL_TYPE)

    @modal.method()
    def generate_video_masks(
        self, video="/root/videos/input.mp4", point_coords=None
    ):
        """Generate masks for a video."""
        import ffmpeg
        import numpy as np
        import torch
        from PIL import Image

        frames_dir = convert_video_to_frames(video)

        # scan all the JPEG files in this directory
        frame_names = [
            p
            for p in frames_dir.iterdir()
            if p.suffix in [".jpg", ".jpeg", ".JPG", ".JPEG"]
        ]
        frame_names.sort(key=lambda p: int(p.stem))

        # We are hardcoding the input point and label here
        # In a real-world scenario, you would want to display the video
        # and allow the user to click on the video to select the point
        if point_coords is None:
            width, height = Image.open(frame_names[0]).size
            point_coords = [[width // 2, height // 2]]

        points = np.array(point_coords, dtype=np.float32)
        # for labels, `1` means positive click and `0` means negative click
        labels = np.array([1] * len(points), np.int32)

        # run the model on GPU
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16),
        ):
            self.inference_state = self.video_predictor.init_state(
                video_path=str(frames_dir)
            )

            # add new prompts and instantly get the output on the same frame
            (
                frame_idx,
                object_ids,
                masks,
            ) = self.video_predictor.add_new_points_or_box(
                inference_state=self.inference_state,
                frame_idx=0,
                obj_id=1,
                points=points,
                labels=labels,
            )

            print(
                f"frame_idx: {frame_idx}, object_ids: {object_ids}, masks: {masks}"
            )

            # run propagation throughout the video and collect the results in a dict
            video_segments = {}  # video_segments contains the per-frame segmentation results
            for (
                out_frame_idx,
                out_obj_ids,
                out_mask_logits,
            ) in self.video_predictor.propagate_in_video(self.inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }

        out_dir = Path("/root/mask_frames")
        out_dir.mkdir(exist_ok=True)

        vis_frame_stride = 5  # visualize every 5th frame
        save_segmented_frames(
            video_segments,
            frames_dir,
            out_dir,
            frame_names,
            stride=vis_frame_stride,
        )

        ffmpeg.input(
            f"{out_dir}/frame_*.png",
            pattern_type="glob",
            framerate=30 / vis_frame_stride,
        ).filter(
            "scale",
            "trunc(iw/2)*2",
            "trunc(ih/2)*2",  # round to even dimensions to encode for "dumb players", https://trac.ffmpeg.org/wiki/Encode/H.264#Encodingfordumbplayers
        ).output(
            str(out_dir / "out.mp4"), format="mp4", pix_fmt="yuv420p"
        ).run()

        return (out_dir / "out.mp4").read_bytes()
def convert_video_to_frames(self, input_video="/root/videos/input.mp4"):
    import ffmpeg

    input_video = Path(input_video)
    output_dir = (  # output on local filesystem, not on the remote Volume
        input_video.parent.parent / input_video.stem / "video_frames"
    )
    output_dir.mkdir(exist_ok=True, parents=True)

    ffmpeg.input(input_video).output(
        f"{output_dir}/%05d.jpg", qscale=2, start_number=0
    ).run()

    return output_dir


def show_mask(mask, ax, obj_id=None, random_color=False):
    import matplotlib.pyplot as plt
    import numpy as np

    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def save_segmented_frames(
    video_segments, frames_dir, out_dir, frame_names, stride=5
):
    import io

    import matplotlib.pyplot as plt
    from PIL import Image

    frames_dir, out_dir = Path(frames_dir), Path(out_dir)

    frame_images = []
    inches_per_px = 1 / plt.rcParams["figure.dpi"]
    for out_frame_idx in range(0, len(frame_names), stride):
        frame = Image.open(frames_dir / frame_names[out_frame_idx])
        width, height = frame.size
        width, height = width - width % 2, height - height % 2
        fig, ax = plt.subplots(
            figsize=(width * inches_per_px, height * inches_per_px)
        )
        ax.axis("off")
        ax.imshow(frame)

        [
            show_mask(mask, ax, obj_id=obj_id)
            for (obj_id, mask) in video_segments[out_frame_idx].items()
        ]

        # Convert plot to PNG bytes
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        # fig.savefig(buf, format="png")
        buf.seek(0)
        frame_images.append(buf.getvalue())
        plt.close(fig)

    for ii, frame in enumerate(frame_images):
        (out_dir / f"frame_{str(ii).zfill(3)}.png").write_bytes(frame)

@app.local_entrypoint()
def main(
    input_video=Path(__file__).parent.parent / "input.mp4",
    x_point=1850,
    y_point=100,
):
    with video_vol.batch_upload(force=True) as batch:
        batch.put_file(input_video, "input.mp4")

    model = Model()

    if x_point is not None and y_point is not None:
        point_coords = [[x_point, y_point]]
    else:
        point_coords = None

    print(f"Running SAM 2 on {input_video}")
    video_bytes = model.generate_video_masks.remote(point_coords=point_coords)

    dir = Path("/tmp/sam2_outputs")
    dir.mkdir(exist_ok=True, parents=True)
    output_path = dir / "segmented_video.mp4"
    output_path.write_bytes(video_bytes)
    print(f"Saved output video to {output_path}")

