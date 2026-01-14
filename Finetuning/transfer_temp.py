import argparse
import os
import torch
from utils import save_thresholds_json


def convert_checkpoint(src_path, dst_path=None, thresholds_out=None):
    ckpt = torch.load(src_path, weights_only=False)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    else:
        state_dict = ckpt

    if dst_path is None:
        dst_path = src_path.replace(".pth.tar", "_weights_only.pth.tar")

    os.makedirs(os.path.dirname(dst_path), exist_ok=True)
    torch.save({"state_dict": state_dict}, dst_path)

    ordinal = ckpt.get("ordinal_thresholds") if isinstance(ckpt, dict) else None
    thresholds_path = None
    if ordinal is not None:
        if thresholds_out is None:
            thresholds_out = dst_path.replace("_weights_only.pth.tar", "_thresholds.json")
        os.makedirs(os.path.dirname(thresholds_out), exist_ok=True)
        save_thresholds_json(thresholds_out, ordinal)
        thresholds_path = thresholds_out

    return dst_path, thresholds_path


def parse_args():
    parser = argparse.ArgumentParser(description="Convert advCheX_hyp_multi_level checkpoint to weights-only format.")
    parser.add_argument("src", help="source checkpoint (.pth.tar)")
    parser.add_argument("--dst", help="output weights-only checkpoint path", default=None)
    parser.add_argument("--thresholds_out", help="optional thresholds json output", default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    weights_path, th_path = convert_checkpoint(args.src, args.dst, args.thresholds_out)
    print(f"[convert] weights saved to: {weights_path}")
    if th_path:
        print(f"[convert] thresholds saved to: {th_path}")


if __name__ == "__main__":
    main()
