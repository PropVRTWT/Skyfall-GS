#!/usr/bin/env python3
"""
patch_submodules.py

Applies runtime and build-time patches to submodule files:
1. MoGe idu_depth.py: GCS local weight fallback & safe destructor
2. FlowEdit idu_refine.py: In-memory pipeline caching & /tmp/flux_pipeline usage
"""
import os
import re

def patch_moge():
    candidates = [
        "/app/submodules/MoGe/idu_depth.py",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "submodules/MoGe/idu_depth.py"),
        "submodules/MoGe/idu_depth.py"
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                content = open(path).read()
                old_init = 'self.model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()'
                new_init = (
                    'model_path = os.environ.get("MOGE_MODEL_PATH")\n'
                    '        if not model_path:\n'
                    '            for cand in ["/mnt/gcs/weights/moge-vitl/model.pt", "/mnt/gcs/models/moge-vitl/model.pt", "/mnt/gcs/weights/model.pt", "/mnt/gcs/moge-vitl/model.pt"]:\n'
                    '                if os.path.isfile(cand):\n'
                    '                    model_path = cand\n'
                    '                    print(f"[MoGeIDU] Found local weights in GCS: {cand}")\n'
                    '                    break\n'
                    '        if not model_path:\n'
                    '            model_path = "Ruicheng/moge-vitl"\n'
                    '        self.model = MoGeModel.from_pretrained(model_path).to(device).eval()'
                )
                if old_init in content:
                    content = content.replace(old_init, new_init)
                content = re.sub(
                    r"def __del__\(self\):[\s\S]*?(?=\s+@torch|\s+def run)",
                    "def __del__(self):\n        try:\n            if hasattr(self, \"model\"):\n                del self.model\n        except BaseException:\n            pass\n\n    ",
                    content
                )
                with open(path, "w") as f:
                    f.write(content)
                print(f"[patch_submodules] Successfully patched {path}")
            except Exception as e:
                print(f"[patch_submodules] Notice on {path}: {e}")

def patch_flowedit():
    candidates = [
        "/app/submodules/FlowEdit/idu_refine.py",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "submodules/FlowEdit/idu_refine.py"),
        "submodules/FlowEdit/idu_refine.py"
    ]
    for path in candidates:
        if os.path.isfile(path):
            try:
                content = open(path).read()
                if "_CACHED_PIPELINES" not in content:
                    content = content.replace("class FlowEditRefineIDU:", "_CACHED_PIPELINES = {}\n\nclass FlowEditRefineIDU:")

                if "if model_type in _CACHED_PIPELINES:" not in content:
                    old_flux = 'pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.float16)'
                    new_flux = (
                        'if model_type in _CACHED_PIPELINES:\n'
                        '            print(f"[FlowEdit] Reusing existing in-memory {model_type} pipeline (0s reload time)!", flush=True)\n'
                        '            pipe = _CACHED_PIPELINES[model_type].to(self.device)\n'
                        '        else:\n'
                        '            flux_name = "black-forest-labs/FLUX.1-dev"\n'
                        '            local_only = False\n'
                        '            if os.path.isdir("/tmp/flux_pipeline"):\n'
                        '                flux_name = "/tmp/flux_pipeline"\n'
                        '                local_only = True\n'
                        '                print(f"[FlowEdit] Using local pre-cached FLUX pipeline at: {flux_name}", flush=True)\n'
                        '            pipe = FluxPipeline.from_pretrained(flux_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=local_only).to(self.device)\n'
                        '            _CACHED_PIPELINES[model_type] = pipe'
                    )
                    if old_flux in content:
                        content = content.replace(old_flux, new_flux)

                content = re.sub(
                    r"def __del__\(self\):[\s\S]*?(?=\s+@contextmanager)",
                    "def __del__(self):\n        try:\n            if torch is not None and hasattr(torch, \"cuda\") and torch.cuda is not None and torch.cuda.is_available():\n                total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)\n                if total_vram_gb < 40 and hasattr(self, \"pipe\") and self.pipe is not None:\n                    self.pipe.to(\"cpu\")\n                    torch.cuda.empty_cache()\n        except BaseException:\n            pass\n\n    ",
                    content
                )
                with open(path, "w") as f:
                    f.write(content)
                print(f"[patch_submodules] Successfully patched {path}")
            except Exception as e:
                print(f"[patch_submodules] Notice on {path}: {e}")

if __name__ == "__main__":
    patch_moge()
    patch_flowedit()
