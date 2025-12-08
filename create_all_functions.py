# ============================================================================
# RQ2: MULTIPLE TASKS TEMPLATE
# ============================================================================

import torch
import numpy as np
import os
import json
from collections import defaultdict

# Configuration
OUTPUT_DIR = "./data/rq2_boolean_discovery_expanded"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CelebA attributes (40 total)
ATTRIBUTE_NAMES = [
    "5_o_Clock_Shadow",
    "Arched_Eyebrows",
    "Attractive",
    "Bags_Under_Eyes",
    "Bald",
    "Bangs",
    "Big_Lips",
    "Big_Nose",
    "Black_Hair",
    "Blond_Hair",
    "Blurry",
    "Brown_Hair",
    "Bushy_Eyebrows",
    "Chubby",
    "Double_Chin",
    "Eyeglasses",
    "Goatee",
    "Gray_Hair",
    "Heavy_Makeup",
    "High_Cheekbones",
    "Male",
    "Mouth_Slightly_Open",
    "Mustache",
    "Narrow_Eyes",
    "No_Beard",
    "Oval_Face",
    "Pale_Skin",
    "Pointy_Nose",
    "Receding_Hairline",
    "Rosy_Cheeks",
    "Sideburns",
    "Smiling",
    "Straight_Hair",
    "Wavy_Hair",
    "Wearing_Earrings",
    "Wearing_Hat",
    "Wearing_Lipstick",
    "Wearing_Necklace",
    "Wearing_Necktie",
    "Young",
]


def get_attr_idx(attr_name):
    return ATTRIBUTE_NAMES.index(attr_name)


# ============================================================================
# EXECUTE ALL TASKS
# ============================================================================
if __name__ == "__main__":

