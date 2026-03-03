"""
Run MotionScript captioning on a .pt file of joint coords captured via MediaPipe.

Usage:
    python run_captioning.py --input my_motion.pt --output_dir ./captions_out
"""

import sys
import os
import argparse
import torch
import types
import importlib.util

# ── 1. Make src importable as text2pose ──────────────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SRC_DIR)

# Create fake package hierarchy so 'from text2pose.posescript.X import ...' works
text2pose  = types.ModuleType("text2pose")
posescript = types.ModuleType("text2pose.posescript")
text2pose.posescript = posescript
sys.modules["text2pose"]            = text2pose
sys.modules["text2pose.posescript"] = posescript

def load_module(name, filepath):
    spec = importlib.util.spec_from_file_location(name, filepath)
    mod  = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# Load in dependency order
load_module("text2pose.config",                              os.path.join(SRC_DIR, "config.py"))
load_module("text2pose.utils",                               os.path.join(SRC_DIR, "utils.py"))
load_module("text2pose.posescript.MS_Algorithms",            os.path.join(SRC_DIR, "MS_Algorithms.py"))
load_module("text2pose.posescript.captioning_data",          os.path.join(SRC_DIR, "captioning_data.py"))
load_module("text2pose.posescript.captioning_data_ablation", os.path.join(SRC_DIR, "captioning_data_ablation.py"))
load_module("text2pose.posescript.posecodes",                os.path.join(SRC_DIR, "posecodes.py"))

import captioning as captioning_py
from captioning import JOINT_NAMES2ID


# ── 2. Manually build the final joint tensor ─────────────────────────────────
def add_virtual_joints(coords):
    """
    Our MediaPipe tensor is (N, 26, 3):
      0-21  : 22 body joints matching JOINT_NAMES order
      22-23 : orientation / translation placeholders (zeros)
      24    : left_middle2
      25    : right_middle2

    JOINT_NAMES expects (N, 29, 3) after prepare_input adds 3 virtual joints:
      26: left_hand  = avg(left_wrist, left_middle2)
      27: right_hand = avg(right_wrist, right_middle2)
      28: torso      = avg(pelvis, neck, spine3)
    """
    left_hand  = 0.5 * (coords[:, JOINT_NAMES2ID["left_wrist"]]  + coords[:, 24])
    right_hand = 0.5 * (coords[:, JOINT_NAMES2ID["right_wrist"]] + coords[:, 25])
    torso      = (1/3) * (
        coords[:, JOINT_NAMES2ID["pelvis"]] +
        coords[:, JOINT_NAMES2ID["neck"]]   +
        coords[:, JOINT_NAMES2ID["spine3"]]
    )
    added = [j.view(-1, 1, 3) for j in [left_hand, right_hand, torso]]
    return torch.cat([coords] + added, dim=1)  # (N, 29, 3)


# ── 3. Main run function ──────────────────────────────────────────────────────
def run(input_path, output_dir):
    coords = torch.load(input_path)
    print(f"Loaded coords: {coords.shape}")  # expect (N, 26, 3)

    # Move to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    coords = coords.to(device)

    # Build full 29-joint tensor
    coords = add_virtual_joints(coords)
    print(f"After adding virtual joints: {coords.shape}")  # (N, 29, 3)

    os.makedirs(output_dir, exist_ok=True)

    # Bypass prepare_input since we already built the correct tensor
    original_prepare = captioning_py.prepare_input
    captioning_py.prepare_input = lambda x: x
    try:
        result = captioning_py.main(coords, save_dir=output_dir, verbose=True)
    finally:
        captioning_py.prepare_input = original_prepare

    # Save captions to file
    if result is not None:
        binning_details, motioncodes4vis, motion_descriptions_non_agg, motion_descriptions = result
        print(f"motion_descriptions type: {type(motion_descriptions)}")
        print(f"motion_descriptions length: {len(motion_descriptions)}")
        print(f"first few: {motion_descriptions[:3]}")
        print(f"motioncodes4vis type: {type(motioncodes4vis)}")
        print(f"motioncodes4vis: {motioncodes4vis[:3] if motioncodes4vis else 'empty'}")
        print(f"motion_descriptions_non_agg: {motion_descriptions_non_agg[:3]}")
        coords = torch.load("my_motion.pt")
        print("min:", coords.min().item())
        print("max:", coords.max().item())
        print("mean joint distance:", (coords[:, 1] - coords[:, 2]).norm(dim=1).mean().item())  # hip width
        out_path = os.path.join(output_dir, "captions.txt")
        with open(out_path, "w") as f:
            f.write(" ".join(motion_descriptions))
        print(f"\nDone. Captions saved to: {out_path}")
        print("\n--- CAPTIONS ---")
        print(" ".join(motion_descriptions))


# ── 4. Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="my_motion.pt",   help="Path to .pt joint tensor")
    parser.add_argument("--output_dir", default="./captions_out", help="Where to save captions")
    args = parser.parse_args()
    run(args.input, args.output_dir)