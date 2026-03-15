#!/usr/bin/env python3
"""Inspect ACE-Step model architecture by printing PyTorch layer structure."""

import argparse
import os


DEFAULTS = {
    "dit": "acestep-v15-turbo",
    "lm": "acestep-5Hz-lm-1.7B",
    "vae": "vae",
}


def main():
    parser = argparse.ArgumentParser(description="Print ACE-Step model layer structure")
    parser.add_argument(
        "--component",
        required=True,
        choices=["dit", "lm", "vae"],
        help="Model component to inspect",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model directory name (default depends on component)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        default="/workspace/checkpoints",
        help="Root checkpoint directory (default: /workspace/checkpoints)",
    )
    args = parser.parse_args()

    model_name = args.model if args.model else DEFAULTS[args.component]
    model_path = os.path.join(args.checkpoint_dir, model_name)

    print(f"Loading {args.component} from {model_path} ...")

    if args.component == "dit":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            model_path, trust_remote_code=True, attn_implementation="eager"
        )
    elif args.component == "lm":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True
        )
    elif args.component == "vae":
        from diffusers import AutoencoderOobleck

        model = AutoencoderOobleck.from_pretrained(model_path)

    print(model)


if __name__ == "__main__":
    main()
