import argparse
import json
import os
from datetime import datetime

import numpy as np
import torch
from medpy import metric
from torch.utils.data import DataLoader

from dataloaders.dataset import BaseDataSets
from networks.net_factory import net_factory


def _normalize_to_uint8(image: np.ndarray) -> np.ndarray:
    img = image.astype(np.float32)
    vmin = float(np.min(img))
    vmax = float(np.max(img))
    if vmax - vmin < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    img = (img - vmin) / (vmax - vmin)
    img = np.clip(img * 255.0, 0, 255)
    return img.astype(np.uint8)


def _overlay_masks(gray_uint8: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> np.ndarray:
    h, w = gray_uint8.shape
    rgb = np.stack([gray_uint8, gray_uint8, gray_uint8], axis=-1).copy()
    gt = gt.astype(bool)
    pred = pred.astype(bool)
    overlap = gt & pred
    rgb[gt] = np.array([0, 255, 0], dtype=np.uint8)
    rgb[pred] = np.array([255, 0, 0], dtype=np.uint8)
    rgb[overlap] = np.array([255, 255, 0], dtype=np.uint8)
    return rgb


def calculate_metrics_binary(pred: np.ndarray, gt: np.ndarray) -> dict:
    pred = (pred > 0).astype(np.uint8)
    gt = (gt > 0).astype(np.uint8)

    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())

    if pred_sum == 0 and gt_sum == 0:
        return {"DSC": 1.0, "JC": 1.0, "95HD": 0.0, "ASD": 0.0}
    if pred_sum == 0 and gt_sum > 0:
        return {"DSC": 0.0, "JC": 0.0, "95HD": 100.0, "ASD": 100.0}
    if pred_sum > 0 and gt_sum == 0:
        return {"DSC": 0.0, "JC": 0.0, "95HD": 100.0, "ASD": 100.0}

    dsc = float(metric.binary.dc(pred, gt))
    jc = float(metric.binary.jc(pred, gt))
    hd95 = float(metric.binary.hd95(pred, gt))
    asd = float(metric.binary.asd(pred, gt))
    return {"DSC": dsc, "JC": jc, "95HD": hd95, "ASD": asd}


def _unwrap_net_output(out):
    if isinstance(out, dict):
        if "out" in out:
            return out["out"]
        if "logits" in out:
            return out["logits"]
        return next(iter(out.values()))
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained URRL U-Net checkpoint."
    )
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--ckpt_path", type=str, required=True)
    parser.add_argument("--model", type=str, default="unet")
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--patch_size", type=int, nargs=2, default=[256, 256])
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--save_dir", type=str, default="./test_results")
    parser.add_argument("--max_vis", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)
    vis_dir = os.path.join(args.save_dir, "vis")
    os.makedirs(vis_dir, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    net = net_factory(net_type=args.model, in_chns=1, class_num=args.num_classes)
    
    # Load either a plain state dict or a wrapped training checkpoint.
    checkpoint = torch.load(args.ckpt_path, map_location=device)
    
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
        
    # Strip the prefix produced by torch.nn.DataParallel.
    clean_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            clean_state_dict[k[7:]] = v
        else:
            clean_state_dict[k] = v

    net.load_state_dict(clean_state_dict)
    
    net.to(device)
    net.eval()

    ds = BaseDataSets(base_dir=args.root_path, split="test")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    all_rows = []
    saved_vis = 0

    with torch.no_grad():
        for batch in loader:
            image = batch["image"]
            label = batch["label"]
            idx = batch.get("idx", None)

            if isinstance(image, torch.Tensor):
                image_t = image.unsqueeze(1).float().to(device)
            else:
                image_t = torch.from_numpy(np.asarray(image)).unsqueeze(1).float().to(device)

            if isinstance(label, torch.Tensor):
                label_t = label.long()
            else:
                label_t = torch.from_numpy(np.asarray(label)).long()

            out = _unwrap_net_output(net(image_t))
            pred = torch.argmax(torch.softmax(out, dim=1), dim=1).cpu().numpy()

            label_np = label_t.cpu().numpy()
            image_np = image.cpu().numpy() if isinstance(image, torch.Tensor) else np.asarray(image)

            b = int(pred.shape[0])
            for i in range(b):
                gt_i = label_np[i]
                pred_i = pred[i]
                img_i = image_np[i]

                m = calculate_metrics_binary(pred_i, gt_i)

                if idx is None:
                    sample_idx = len(all_rows)
                else:
                    sample_idx = int(idx[i])

                image_path = None
                if hasattr(ds, "image_files") and 0 <= sample_idx < len(ds.image_files):
                    image_path = ds.image_files[sample_idx]

                row = {
                    "idx": sample_idx,
                    "image": os.path.basename(image_path) if image_path else None,
                    "metrics": m,
                }
                all_rows.append(row)

                if saved_vis < args.max_vis:
                    img_u8 = _normalize_to_uint8(img_i)
                    overlay = _overlay_masks(img_u8, gt_i, pred_i)
                    name = row["image"] if row["image"] else f"idx_{sample_idx}.png"
                    if not name.lower().endswith(".png"):
                        name = f"{os.path.splitext(name)[0]}.png"
                    out_path = os.path.join(vis_dir, name)
                    from PIL import Image
                    Image.fromarray(overlay).save(out_path)
                    saved_vis += 1

    arr = np.array([[r["metrics"]["DSC"], r["metrics"]["JC"], r["metrics"]["95HD"], r["metrics"]["ASD"]] for r in all_rows], dtype=np.float64)

    summary = {
        "n_test": int(arr.shape[0]),
        "DSC_mean": float(arr[:, 0].mean()) if arr.size else None,
        "DSC_std": float(arr[:, 0].std()) if arr.size else None,
        "JC_mean": float(arr[:, 1].mean()) if arr.size else None,
        "JC_std": float(arr[:, 1].std()) if arr.size else None,
        "95HD_mean": float(arr[:, 2].mean()) if arr.size else None,
        "95HD_std": float(arr[:, 2].std()) if arr.size else None,
        "ASD_mean": float(arr[:, 3].mean()) if arr.size else None,
        "ASD_std": float(arr[:, 3].std()) if arr.size else None,
        "checkpoint": os.path.abspath(args.ckpt_path),
        "root_path": os.path.abspath(args.root_path),
        "model": args.model,
        "num_classes": int(args.num_classes),
        "patch_size": [int(args.patch_size[0]), int(args.patch_size[1])],
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "vis_dir": os.path.abspath(vis_dir),
    }

    out_json = {
        "summary": summary,
        "per_case": all_rows,
    }

    out_path = os.path.join(args.save_dir, "test_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print("wrote:", out_path)


if __name__ == "__main__":
    main()
