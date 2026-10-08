#!/usr/bin/env python3
"""
patch_submodules.py

Applies runtime and build-time patches to submodule files:
1. MoGe idu_depth.py: GCS local weight fallback & safe destructor
2. FlowEdit idu_refine.py: In-memory pipeline caching & /tmp/flux_pipeline usage
"""
import os
import re
import py_compile

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
                py_compile.compile(path, doraise=True)
                print(f"[patch_submodules] Successfully patched and validated {path}")
            except Exception as e:
                print(f"[patch_submodules] Notice on {path}: {e}")

FLOWEDIT_CLASS_CODE = '''_CACHED_PIPELINES = {}

class FlowEditRefineIDU:
    def __init__(self, save_path, device="cuda:0", model_type="FLUX"):
        self.device = device
        self.save_path = save_path
        self.model_type = model_type
        if model_type in _CACHED_PIPELINES:
            print(f"[FlowEdit] Reusing existing in-memory {model_type} pipeline (0s reload time)!", flush=True)
            pipe = _CACHED_PIPELINES[model_type]
            try:
                pipe = pipe.to(self.device)
            except Exception as e:
                print(f"[FlowEdit] pipe.to({self.device}): {e}", flush=True)
        else:
            if model_type == 'FLUX':
                flux_name = "black-forest-labs/FLUX.1-schnell" if os.environ.get("FLUX_MODEL") == "schnell" else "black-forest-labs/FLUX.1-dev"
                local_only = False
                for cand in ["/app/flux_pipeline", "/tmp/flux_pipeline"]:
                    if os.path.isdir(cand) and os.path.isfile(os.path.join(cand, "model_index.json")):
                        flux_name = cand
                        local_only = True
                        print(f"[FlowEdit] Using local pre-cached FLUX pipeline at: {flux_name}", flush=True)
                        break
                pipe = FluxPipeline.from_pretrained(flux_name, torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=local_only)
                pipe = pipe.to(self.device)
            elif model_type == 'SD3':
                pipe = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3-medium-diffusers", torch_dtype=torch.float16, low_cpu_mem_usage=True).to(self.device)
            else:
                raise NotImplementedError(f"Model type {model_type} not implemented")
            _CACHED_PIPELINES[model_type] = pipe

        self.scheduler = pipe.scheduler
        self.pipe = pipe
        os.makedirs(save_path, exist_ok=True)
        print(f"Initialized FlowEdit with {model_type} model.", flush=True)

    def __del__(self):
        try:
            if torch is not None and hasattr(torch, "cuda") and torch.cuda is not None and torch.cuda.is_available():
                total_vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                if total_vram_gb < 40 and hasattr(self, "pipe") and self.pipe is not None:
                    self.pipe.to("cpu")
                    torch.cuda.empty_cache()
        except BaseException:
            pass'''

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
                new_content = re.sub(
                    r"(_CACHED_PIPELINES\s*=\s*\{\}\s*)?class FlowEditRefineIDU:[\s\S]*?(?=\s+@contextmanager)",
                    FLOWEDIT_CLASS_CODE + "\n    ",
                    content
                )
                with open(path, "w") as f:
                    f.write(new_content)
                py_compile.compile(path, doraise=True)
                print(f"[patch_submodules] Successfully patched and validated {path}")
            except Exception as e:
                print(f"[patch_submodules] Notice on {path}: {e}")

if __name__ == "__main__":
    patch_moge()
    patch_flowedit()
