#!/usr/bin/env python3
"""
setup_flux.py

Assembles a complete Diffusers FLUX pipeline at /tmp/flux_pipeline (or /app/flux_pipeline)
by copying local configs from /app/flux_template (or ./flux_template)
and linking/copying the 8 large .safetensors blobs from GCS or local staging.

This completely bypasses Hugging Face Hub downloads and FUSE write limits!
"""
import os
import shutil
import sys
import argparse

WEIGHT_MAP = {
    "ae.safetensors": "f7/f73eecf7c469ff442523dc712cc161d631df071bf4d9d793494fbf00cdd80a82",
    "text_encoder/model.safetensors": "41/413459000fdce4c7d7c50e2a7841333f6728021970bf2d905e8b384b8decab9b",
    "text_encoder_2/model-00001-of-00002.safetensors": "bd/bd8dd671f96c84eb482d580b02aef9e27624b49d28c0e6879064cea3d563e0df",
    "text_encoder_2/model-00002-of-00002.safetensors": "65/65b0cc40f50cac5edf11f8b6b89fb03c51e097cac28cc3ccc992d3756864e072",
    "transformer/diffusion_pytorch_model-00001-of-00003.safetensors": "b2/b27d0a753010fddf4ac3f5e15849d1b59da33abf134af2ce4ee36731c70b6835",
    "transformer/diffusion_pytorch_model-00002-of-00003.safetensors": "34/34934cfc1fd53f3a257bb3fe7fbe30024cccc0c36f029cb9592e52d190d17ea4",
    "transformer/diffusion_pytorch_model-00003-of-00003.safetensors": "4a/4ae4895d87e0c251654f20c588fd2b5bc6750dc494dbbaf97062ef1d7c9a764c",
    "vae/diffusion_pytorch_model.safetensors": "44/4479aac938c224dbaef8d126dc178a0650d09140dcc46885be6d7c72bb6f176f"
}

def is_pipeline_complete(path):
    if not os.path.isdir(path) or not os.path.isfile(os.path.join(path, "model_index.json")):
        return False
    for rel_dest in WEIGHT_MAP.keys():
        dest = os.path.join(path, rel_dest)
        if not (os.path.isfile(dest) or os.path.islink(dest)):
            return False
    return True

def find_blob_file(blob_dir, blob_rel):
    # Try nested path: e.g. blob_dir/f7/f73eec...
    p1 = os.path.join(blob_dir, blob_rel)
    if os.path.isfile(p1):
        return p1
    # Try flat filename: e.g. blob_dir/f73eec...
    fname = os.path.basename(blob_rel)
    p2 = os.path.join(blob_dir, fname)
    if os.path.isfile(p2):
        return p2
    # Recursive search
    for root, _, files in os.walk(blob_dir):
        if fname in files:
            return os.path.join(root, fname)
    return None

def main():
    parser = argparse.ArgumentParser(description="Assemble FLUX diffusers pipeline from local cache / blobs")
    parser.add_argument("--target", default=None, help="Target directory for FLUX pipeline (default: /tmp/flux_pipeline)")
    parser.add_argument("--blobs", default=None, help="Path to blobs directory containing safetensors")
    parser.add_argument("--copy", action="store_true", help="Copy instead of symlink")
    args = parser.parse_args()

    # If baked into container image already at /app/flux_pipeline, we are done
    if is_pipeline_complete("/app/flux_pipeline"):
        print("[setup_flux] Complete pre-baked FLUX pipeline detected at /app/flux_pipeline! (0 network I/O)")
        try:
            import patch_submodules
            patch_submodules.patch_flowedit()
        except Exception:
            pass
        return

    target_dir = args.target or "/tmp/flux_pipeline"

    if is_pipeline_complete(target_dir):
        print(f"[setup_flux] Complete FLUX pipeline already assembled at {target_dir}! Nothing to do.")
        try:
            import patch_submodules
            patch_submodules.patch_flowedit()
        except Exception:
            pass
        return

    # Find template
    template_candidates = [
        "/app/flux_template",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "flux_template"),
        "./flux_template"
    ]
    template_dir = None
    for cand in template_candidates:
        if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "model_index.json")):
            template_dir = cand
            break
            
    if not template_dir:
        print("[setup_flux] Template directory flux_template not found, skipping assembling FLUX pipeline")
        return

    # Find blobs directory
    if args.blobs:
        blobs_candidates = [args.blobs]
    else:
        blobs_candidates = [
            "/tmp/flux_blobs",
            "/workspace/flux_blobs",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "flux_blobs"),
            "./flux_blobs",
            "/mnt/gcs/hf_cache/blobs",
            "/mnt/gcs/blobs",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache", "blobs"),
            "./hf_cache/blobs"
        ]
    
    blob_dir = None
    for cand in blobs_candidates:
        if os.path.isdir(cand):
            # Check if directory has any blob files
            files = [f for f in os.listdir(cand) if not f.startswith('.')]
            if files:
                blob_dir = cand
                break

    if not blob_dir:
        # If running during docker build with empty staging directory, exit cleanly
        if target_dir == "/app/flux_pipeline":
            print("[setup_flux] No blob files found for docker baking. Skipping build-time assembly (will assemble at runtime).")
            return
        print(f"[setup_flux] Blob directory not found in candidates: {blobs_candidates}")
        return

    print(f"[setup_flux] Assembling FLUX pipeline from {template_dir} and weights from {blob_dir} -> {target_dir}")
    os.makedirs(target_dir, exist_ok=True)

    # 1. Copy config templates
    for item in os.listdir(template_dir):
        s = os.path.join(template_dir, item)
        d = os.path.join(target_dir, item)
        if os.path.isdir(s):
            if not os.path.exists(d):
                shutil.copytree(s, d)
        elif not os.path.exists(d):
            shutil.copy2(s, d)

    # 2. Link or copy large safetensors files
    missing = []
    should_copy = args.copy or (args.target == "/app/flux_pipeline")
    for rel_dest, blob_rel in WEIGHT_MAP.items():
        blob_path = find_blob_file(blob_dir, blob_rel)
        dest_path = os.path.join(target_dir, rel_dest)
        
        if not blob_path:
            missing.append(blob_rel)
            continue
            
        if os.path.exists(dest_path) or os.path.islink(dest_path):
            os.remove(dest_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)

        if should_copy:
            print(f"[setup_flux] Copying {rel_dest} <- {blob_path}...")
            shutil.copy2(blob_path, dest_path)
        else:
            os.symlink(blob_path, dest_path)
            print(f"[setup_flux] Linked {rel_dest} -> {blob_path}")

    if missing:
        print(f"[setup_flux] WARNING: Missing {len(missing)} blob files in {blob_dir}: {missing}")
    else:
        print(f"[setup_flux] SUCCESS: Complete FLUX pipeline assembled at {target_dir} with 0 downloads required!")

    # 3. Ensure idu_refine.py is patched at runtime to load from /app or /tmp and cache pipeline
    try:
        import patch_submodules
        patch_submodules.patch_flowedit()
    except Exception as e:
        print(f"[setup_flux] Notice running patch_submodules: {e}")

if __name__ == "__main__":
    main()
