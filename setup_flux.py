#!/usr/bin/env python3
"""
setup_flux.py

Assembles a complete Diffusers FLUX pipeline at /tmp/flux_pipeline
by copying local configs from /app/flux_template (or ./flux_template)
and symlinking the 8 large .safetensors blobs from /mnt/gcs/hf_cache/blobs.

This completely bypasses Hugging Face Hub downloads and FUSE write limits!
"""
import os
import shutil
import sys

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

def main():
    target_dir = "/tmp/flux_pipeline"
    template_candidates = [
        "/app/flux_template",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "flux_template"),
        "./flux_template"
    ]
    
    template_dir = None
    for cand in template_candidates:
        if os.path.isdir(cand):
            template_dir = cand
            break
            
    if not template_dir:
        print("[setup_flux] Template directory flux_template not found, skipping assembling /tmp/flux_pipeline")
        return

    # Check GCS blobs directory
    blobs_candidates = [
        "/mnt/gcs/hf_cache/blobs",
        "/mnt/gcs/blobs",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf_cache", "blobs"),
        "./hf_cache/blobs"
    ]
    blob_dir = None
    for cand in blobs_candidates:
        if os.path.isdir(cand):
            blob_dir = cand
            break

    if not blob_dir:
        print(f"[setup_flux] Blob directory not found in {blobs_candidates}")
        return

    print(f"[setup_flux] Assembling FLUX pipeline from {template_dir} and weights from {blob_dir} -> {target_dir}")
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

    # 1. Copy config template (4.8 MB)
    shutil.copytree(template_dir, target_dir)

    # 2. Symlink the large safetensors files
    missing = []
    for rel_dest, blob_rel in WEIGHT_MAP.items():
        blob_path = os.path.join(blob_dir, blob_rel)
        dest_path = os.path.join(target_dir, rel_dest)
        
        # Check if blob exists
        if not os.path.isfile(blob_path):
            missing.append(blob_path)
            continue
            
        if os.path.exists(dest_path) or os.path.islink(dest_path):
            os.remove(dest_path)
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        os.symlink(blob_path, dest_path)
        print(f"[setup_flux] Linked {rel_dest} -> {blob_path}")

    if missing:
        print(f"[setup_flux] WARNING: Missing {len(missing)} blob files in GCS: {missing}")
    else:
        print(f"[setup_flux] SUCCESS: Complete FLUX pipeline assembled at {target_dir} with 0 downloads required!")

    # 3. Ensure idu_refine.py is patched at runtime to load from /tmp/flux_pipeline
    for idu_path in ["/app/submodules/FlowEdit/idu_refine.py", "submodules/FlowEdit/idu_refine.py"]:
        if os.path.isfile(idu_path):
            try:
                content = open(idu_path).read()
                old_flux = 'pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)'
                new_flux = ('flux_name = "black-forest-labs/FLUX.1-dev"\n'
                            '            local_only = False\n'
                            '            if os.path.isdir("/tmp/flux_pipeline"):\n'
                            '                flux_name = "/tmp/flux_pipeline"\n'
                            '                local_only = True\n'
                            '                print(f"[FlowEdit] Using local pre-cached FLUX pipeline at: {flux_name}", flush=True)\n'
                            '            pipe = FluxPipeline.from_pretrained(flux_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=local_only)')
                if old_flux in content:
                    content = content.replace(old_flux, new_flux)
                    with open(idu_path, "w") as f:
                        f.write(content)
                    print(f"[setup_flux] Patched {idu_path} to use /tmp/flux_pipeline")
            except Exception as e:
                print(f"[setup_flux] Notice on {idu_path}: {e}")

if __name__ == "__main__":
    main()
