import argparse
import glob
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image

warnings.filterwarnings("ignore")

NUM_CLASSES = 2
MODEL_ARCH = "resnet50"
SUPPORTED_EXTS = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp", "*.tiff")


def build_cls_transform():
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def collect_images(image_dir):
    files = []
    for p in SUPPORTED_EXTS:
        files.extend(glob.glob(os.path.join(image_dir, p)))
    return sorted(files)


def load_cls_model(model_path, device):
    from utils.utils import get_network

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"分类模型不存在: {model_path}")

    model = get_network(MODEL_ARCH, num_classes=NUM_CLASSES).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        model.load_state_dict(checkpoint["model"])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    return model


def adjust_model_output_layer(model, target_channels=3):
    original_layer = model.out[2]
    new_layer = nn.Conv2d(
        in_channels=original_layer.in_channels,
        out_channels=target_channels,
        kernel_size=original_layer.kernel_size,
        stride=original_layer.stride,
        padding=original_layer.padding,
        bias=original_layer.bias is not None,
    )
    model.out[2] = new_layer
    return model


def load_weight_safely(model, checkpoint):
    state_dict = checkpoint if "model" not in checkpoint else checkpoint["model"]
    model_dict = model.state_dict()
    load_dict = {}
    for k, v in state_dict.items():
        if k in model_dict and v.shape == model_dict[k].shape:
            load_dict[k] = v
    model_dict.update(load_dict)
    model.load_state_dict(model_dict, strict=False)
    return model


def load_dire_model(dire_model_path, device, guided_dir):
    if not os.path.exists(dire_model_path):
        raise FileNotFoundError(f"DIRE扩散模型不存在: {dire_model_path}")
    if not os.path.isdir(guided_dir):
        raise FileNotFoundError(f"guided-diffusion目录不存在: {guided_dir}")

    if guided_dir not in sys.path:
        sys.path.insert(0, guided_dir)
    from guided_diffusion.script_util import create_model_and_diffusion  # type: ignore

    model, diffusion = create_model_and_diffusion(
        image_size=256,
        num_channels=256,
        num_res_blocks=2,
        attention_resolutions="32,16,8",
        class_cond=False,
        dropout=0.1,
        learn_sigma=False,
        channel_mult="",
        num_heads=8,
        num_head_channels=-1,
        num_heads_upsample=-1,
        diffusion_steps=1000,
        noise_schedule="linear",
        timestep_respacing="20",
        use_kl=False,
        predict_xstart=False,
        rescale_timesteps=True,
        rescale_learned_sigmas=True,
        use_checkpoint=True,
        use_scale_shift_norm=True,
        resblock_updown=False,
        use_fp16=(device == "cuda"),
        use_new_attention_order=False,
    )
    model = adjust_model_output_layer(model, target_channels=3)
    checkpoint = torch.load(dire_model_path, map_location=device)
    model = load_weight_safely(model, checkpoint).to(device).eval()
    if device == "cuda":
        model = model.half()
    return model, diffusion


def generate_dire_pil(image_path, dire_model, diffusion, device):
    img = Image.open(image_path).convert("RGB")
    trans = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    imgs = trans(img).unsqueeze(0).to(device)
    if device == "cuda":
        imgs = imgs.half()
    imgs = F.interpolate(imgs, size=(256, 256), mode="bicubic", antialias=True)

    with torch.no_grad():
        use_amp = device == "cuda"
        with torch.cuda.amp.autocast(enabled=use_amp):
            latent = diffusion.ddim_reverse_sample_loop(
                dire_model, imgs.shape, noise=imgs, clip_denoised=True, model_kwargs={}, real_step=0
            )
            recons = diffusion.ddim_sample_loop(
                dire_model, imgs.shape, noise=latent, clip_denoised=True, model_kwargs={}, real_step=0
            )

    dire = torch.abs(imgs.float() - recons.float())
    dire_np = (dire * 255.0 / 2.0).clamp(0, 255).to(torch.uint8).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(dire_np.astype(np.uint8))


def predict_from_pil(cls_model, device, dire_pil):
    tensor = build_cls_transform()(dire_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        output = cls_model(tensor)
        probs = torch.softmax(output, dim=1).cpu().numpy()[0]
    pred_cls = int(np.argmax(probs))
    return pred_cls, float(probs[0]), float(probs[1])


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser(description="一键版：原图 -> DIRE -> AI概率")
    parser.add_argument("--image_dir", type=str, default=os.path.join(script_dir, "test_image"), help="原图目录")
    parser.add_argument("--model", type=str, default=os.path.join(script_dir, "exp", "final_dire_model.pth"), help="分类模型(.pth)")
    parser.add_argument(
        "--dire_model",
        type=str,
        default=os.path.join(script_dir, "guided-diffusion", "256x256_diffusion_uncond.pt"),
        help="DIRE扩散模型(.pt)",
    )
    parser.add_argument(
        "--guided_dir",
        type=str,
        default=os.path.join(script_dir, "guided-diffusion"),
        help="guided-diffusion目录",
    )
    parser.add_argument(
        "--save_dire_dir",
        type=str,
        default=os.path.join(script_dir, "test_image_dire"),
        help="DIRE图保存目录",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.isdir(args.image_dir):
        raise FileNotFoundError(f"原图目录不存在: {args.image_dir}")
    image_files = collect_images(args.image_dir)
    if not image_files:
        raise ValueError(f"目录中没有可处理图片: {args.image_dir}")

    os.makedirs(args.save_dire_dir, exist_ok=True)
    cls_model = load_cls_model(args.model, device)
    dire_model, diffusion = load_dire_model(args.dire_model, device, args.guided_dir)

    print("=" * 72)
    print(f"原图目录: {args.image_dir}")
    print(f"DIRE目录: {args.save_dire_dir}")
    print(f"分类模型: {args.model}")
    print(f"DIRE模型: {args.dire_model}")
    print(f"运行设备: {device}")
    print(f"图片数量: {len(image_files)}")
    print("-" * 72)

    ai_probs = []
    for idx, image_path in enumerate(image_files, start=1):
        dire_img = generate_dire_pil(image_path, dire_model, diffusion, device)
        out_name = os.path.splitext(os.path.basename(image_path))[0] + ".png"
        dire_path = os.path.join(args.save_dire_dir, out_name)
        dire_img.save(dire_path)

        pred_cls, real_prob, ai_prob = predict_from_pil(cls_model, device, dire_img)
        ai_probs.append(ai_prob)
        pred_name = "AI生成" if pred_cls == 1 else "真实图片"
        print(
            f"[{idx:03d}/{len(image_files)}] {os.path.basename(image_path)} | "
            f"AI概率={ai_prob:.4f} | 真实概率={real_prob:.4f} | 预测={pred_name}"
        )

    print("-" * 72)
    print(f"平均AI生成概率: {sum(ai_probs) / len(ai_probs):.4f}")
    print("=" * 72)


if __name__ == "__main__":
    main()
