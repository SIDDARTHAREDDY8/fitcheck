"""GPU database: common inference cards with their VRAM.

Values are usable VRAM in GiB. Architecture matters for fitcheck only
insofar as it sets realistic defaults (e.g. fp8 KV cache on Hopper/Ada);
VRAM capacity is the driver of the verdict.
"""

# name -> (vram_gib, arch_note)
GPUS = {
    "4090": (24, "Ada Lovelace"),
    "rtx-4090": (24, "Ada Lovelace"),
    "4080": (16, "Ada Lovelace"),
    "rtx-4080": (16, "Ada Lovelace"),
    "3090": (24, "Ampere"),
    "rtx-3090": (24, "Ampere"),
    "3080": (12, "Ampere"),
    "rtx-3080": (12, "Ampere"),
    "a100-40": (40, "Ampere"),
    "a100-80": (80, "Ampere"),
    "h100": (80, "Hopper"),
    "h100-80": (80, "Hopper"),
    "v100-16": (16, "Volta"),
    "t4": (16, "Turing"),
    "6000-ada": (48, "Ada Lovelace"),
    "rtx-6000-ada": (48, "Ada Lovelace"),
    "5090": (32, "Blackwell"),
    "rtx-5090": (32, "Blackwell"),
}


def lookup(name: str):
    """Return (vram_gib, arch) for a GPU name, or None if unknown."""
    key = name.strip().lower().replace(" ", "-")
    return GPUS.get(key)


def known_names():
    return sorted(GPUS)
