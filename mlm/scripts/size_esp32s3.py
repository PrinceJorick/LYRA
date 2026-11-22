"""Measure the ESP32-S3 Flash/RAM cost of the pod C code with Espressif's xtensa toolchain.

Compiles mlm/c99/*.c and the generated forests (mlm/export/c99/*.c) with ESP-IDF style flags
(-mlongcalls -ffunction-sections -fdata-sections, -std=c99 -O2 -Wall -Wextra -Werror), then reports
per-section sizes (size -A), the worst per-function stack frame (-fstack-usage), and a test link
from lyra_step that shows how much newlib libm / libgcc soft-float code the models pull in.
Object and link sizes, not a flashed firmware image: FreeRTOS, drivers and BLE are not included.
Writes mlm/export/esp32s3_size.json (gitignored).
Usage: python scripts/size_esp32s3.py [--toolchain <dir with xtensa-esp32s3-elf-gcc>]
Default toolchain: Espressif crosstool-NG esp-14.4.0_20261030 in ~/.espressif/tools (ESP-IDF layout).
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TOOLCHAIN = Path.home() / ".espressif/tools/xtensa-esp-elf/esp-14.4.0_20261030/xtensa-esp-elf/bin"
FLAGS = ["-mlongcalls", "-std=c99", "-O2", "-Wall", "-Wextra", "-Werror", "-ffunction-sections", "-fdata-sections",
         "-fstack-usage"]
# The forests are exported one function per tree: a single ~370 KB function fails on Xtensa
# (L32R literal reach 256 KB, J reach 128 KB) even with -mtext-section-literals / -mauto-litpools.
FOREST_FLAGS: list[str] = []
FLASH = (".text", ".literal", ".rodata")  # with -ffunction-sections: .text.<fn>, .literal.<fn>, .rodata.<x>
RAM = (".data", ".bss")


def sections(size_tool: Path, obj: Path) -> dict[str, int]:
    out = subprocess.run([str(size_tool), "-A", str(obj)], capture_output=True, text=True, check=True).stdout
    totals = {k: 0 for k in FLASH + RAM}
    for line in out.splitlines()[2:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1].isdigit():
            for k in totals:
                if parts[0] == k or parts[0].startswith(k + "."):
                    totals[k] += int(parts[1])
    return totals


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--toolchain", type=Path, default=DEFAULT_TOOLCHAIN)
    args = ap.parse_args()
    gcc, size = args.toolchain / "xtensa-esp32s3-elf-gcc", args.toolchain / "xtensa-esp32s3-elf-size"
    sources = sorted((ROOT / "c99").glob("*.c")) + sorted((ROOT / "export" / "c99").glob("*.c"))
    report, objs = {"compiler": subprocess.run([str(gcc), "--version"], capture_output=True, text=True).stdout.split("\n")[0],
                    "flags": FLAGS, "forest_flags": FOREST_FLAGS, "modules": {}}, []
    with tempfile.TemporaryDirectory() as d:
        d = Path(d)
        for src in sources:
            obj = d / (src.stem + ".o")
            extra = FOREST_FLAGS if src.parent.name == "c99" and src.parent.parent.name == "export" else []
            subprocess.run([str(gcc), *FLAGS, *extra, f"-I{ROOT / 'c99'}", f"-I{ROOT / 'export' / 'c99'}", "-c", str(src),
                            "-o", str(obj)], check=True, cwd=d)
            sec = sections(size, obj)
            su = [line.rsplit(None, 2) for line in (d / (src.stem + ".su")).read_text().splitlines()]
            report["modules"][src.name] = {"sections": sec, "flash_bytes": sum(sec[k] for k in FLASH + (".data",)),
                                           "ram_bytes": sum(sec[k] for k in RAM),
                                           "max_stack_frame_bytes": max((int(s[1]) for s in su), default=0)}
            objs.append(obj)
        elf = d / "pod.elf"  # test link: everything reachable from lyra_step and the two forests
        link = subprocess.run([str(gcc), "-mlongcalls", "-nostartfiles", "-Wl,--gc-sections", "-Wl,-e,lyra_step",
                               "-Wl,--undefined=lyra_multimodal_rf_predict", "-Wl,--undefined=lyra_simenv_rf_predict",
                               "-o", str(elf), *map(str, objs), "-lm"], capture_output=True, text=True)
        if link.returncode == 0:
            sec = sections(size, elf)
            report["linked_from_lyra_step"] = {"sections": sec, "flash_bytes": sum(sec[k] for k in FLASH + (".data",)),
                                               "ram_bytes": sum(sec[k] for k in RAM)}
        else:
            report["linked_from_lyra_step"] = {"error": link.stderr[-2000:]}
    out = ROOT / "export" / "esp32s3_size.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    print(f"{'module':26s} {'flash B':>9s} {'ram B':>8s} {'max frame B':>12s}")
    for name, m in report["modules"].items():
        print(f"{name:26s} {m['flash_bytes']:9d} {m['ram_bytes']:8d} {m['max_stack_frame_bytes']:12d}")
    ln = report["linked_from_lyra_step"]
    print("linked (with libm / soft-float):", ln if "error" in ln else f"flash {ln['flash_bytes']} B, ram {ln['ram_bytes']} B")


if __name__ == "__main__":
    main()
