"""Flash/SRAM measurement of the C99 package with the ESP32-S3 toolchain (ESP-IDF v5.5.6).

1. Each translation unit (model, features, detector) is compiled with xtensa-esp32s3-elf-gcc -Os
   -fstack-usage; section sizes come from the ELF objects (verified, per object).
2. A minimal ESP-IDF application (firmware/esp32s3_app) that runs features -> model -> detector
   is built for esp32s3; idf.py size gives the whole-application flash/DRAM usage and the linker
   map gives the LYRA symbols' share (verified, linked).
Input window buffers belong to the acquisition code, which does not exist yet: their size is
computed from the window length and reported separately as a requirement, not as a measurement.
"""

from __future__ import annotations

import glob
import json
import os
import re
import shutil
import subprocess
from pathlib import Path

from . import config as C

IDF = Path.home() / "esp" / "esp-idf"
APP = C.REPO / "firmware" / "esp32s3_app"


def toolchain() -> dict:
    gcc = sorted(glob.glob(str(Path.home() / ".espressif/tools/xtensa-esp-elf/*/xtensa-esp-elf/bin/xtensa-esp32s3-elf-gcc")))
    if not gcc:
        return {}
    bin_dir = Path(gcc[-1]).parent
    ver = subprocess.run([gcc[-1], "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    return {"gcc": gcc[-1], "size": str(bin_dir / "xtensa-esp32s3-elf-size"), "version": ver}


def _sections(size_tool: str, obj: Path) -> dict:
    out = subprocess.run([size_tool, "-A", str(obj)], capture_output=True, text=True, check=True).stdout
    sec = {}
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith(".") and parts[1].isdigit():
            sec[parts[0]] = sec.get(parts[0], 0) + int(parts[1])
    # .data occupies both: its initial image is in flash and it lives in RAM
    flash = sum(v for k, v in sec.items() if k.startswith((".text", ".rodata", ".literal", ".data")))
    ram = sum(v for k, v in sec.items() if k.startswith((".data", ".bss")))
    return {"sections": sec, "flash_bytes": flash, "static_ram_bytes": ram}


def object_sizes(model_dir: Path, work: Path) -> dict:
    tc = toolchain()
    if not tc:
        return {"status": "ESP32-S3 toolchain not found: no measurement"}
    model_dir, work = model_dir.resolve(), work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    tu = {"model": "#include \"lyra_model.h\"\nfloat lyra_model_entry(const float *f) { return lyra_model_predict(f); }\n",
          "features": None, "detector": None}
    out = {"compiler": tc["version"], "flags": "-std=c99 -Os -mlongcalls -ffunction-sections -fdata-sections -fstack-usage",
           "objects": {}}
    for name, src in tu.items():
        path = work / f"{name}_tu.c" if src else model_dir / f"lyra_{name}.c"
        if src:
            path.write_text(src)
        obj = work / f"{name}.o"
        subprocess.run([tc["gcc"], "-std=c99", "-Os", "-mlongcalls", "-ffunction-sections", "-fdata-sections",
                        "-fstack-usage", "-Wall", "-Wextra", "-I", str(model_dir), "-c", str(path), "-o", str(obj)],
                       check=True, capture_output=True, cwd=work)
        s = _sections(tc["size"], obj)
        su = obj.with_suffix(".su")
        stack = {}
        if su.exists():
            for line in su.read_text().splitlines():
                m = re.match(r".*:(\w+)\s+(\d+)\s+(\w+)", line)
                if m:
                    stack[m.group(1)] = int(m.group(2))
        s["stack_bytes_per_function"] = stack
        out["objects"][name] = s
    return out


def idf_app_size(model_dir: Path) -> dict:
    """Build firmware/esp32s3_app for esp32s3 and return idf.py size (JSON) + LYRA symbols from the map."""
    if not (IDF / "export.sh").exists():
        return {"status": "ESP-IDF not installed: no application build"}
    py = Path.home() / "esp"  # python3 -> uv CPython 3.13 shim used for the IDF install
    pre = f"export PATH={py}:$PATH; . {IDF}/export.sh >/dev/null 2>&1; cd {APP} && "
    env = {k: v for k, v in os.environ.items() if not k.startswith(("VIRTUAL_ENV", "PYTHON"))}

    def idf(args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["bash", "-c", pre + args], capture_output=True, text=True, env=env)

    r = idf("idf.py -DIDF_TARGET=esp32s3 build >build.log 2>&1")
    if r.returncode:
        log = (APP / "build.log").read_text()[-3000:] if (APP / "build.log").exists() else r.stderr[-3000:]
        return {"status": "build failed", "log_tail": log}
    js = lambda s: json.loads(s[s.find("{"):])  # noqa: E731
    total = js(idf("idf.py size --format json2 2>/dev/null").stdout)
    files = js(idf("idf.py size-files --format json2 2>/dev/null").stdout)
    per_obj = {}
    for obj in ("lyra_features.c.obj", "lyra_detector.c.obj", "main.c.obj"):
        mt = files.get(obj, {}).get("memory_types", {})
        flash = sum(v["size"] for k, v in mt.items() if k.startswith("Flash"))
        ram = sum(s["size"] for k, v in mt.items() if k in ("DIRAM", "IRAM") for s in v["sections"].values())
        data = mt.get("DIRAM", {}).get("sections", {}).get(".data", {}).get("size", 0)
        per_obj[obj] = {"flash_bytes": flash + data, "ram_bytes": ram,
                        "sections": {k: {sk: sv["size"] for sk, sv in v["sections"].items()} for k, v in mt.items()}}
    ver = subprocess.run(["git", "-C", str(IDF), "describe", "--tags"], capture_output=True, text=True).stdout.strip()
    return {"status": "built", "esp_idf": ver, "target": "esp32s3", "application": total, "objects": per_obj,
            "note": "main.c.obj = generated model (inlined lyra_model.h) + the probe's input window buffers (.bss)"}


def clean_app_build() -> None:
    shutil.rmtree(APP / "build", ignore_errors=True)
    for f in ("build.log", "sdkconfig", "sdkconfig.old", "dependencies.lock"):
        (APP / f).unlink(missing_ok=True)
