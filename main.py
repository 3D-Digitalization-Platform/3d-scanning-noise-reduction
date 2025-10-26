import os
import sys

from noisereduction.AlphaShapeDenoise import AlphaShapeDenoise
from noisereduction.HdbscanDenoise import HdbscanDenoise
from noisereduction.RansackDenoise import RansackDenoise


def main(obj_path: str):
    hddbscan_1 = HdbscanDenoise(obj_path).run_hdbscan()

    hole_noise_removed = RansackDenoise(hddbscan_1).detect_noise_by_hole()
    # os.startfile(hole_noise_removed)

    final_res = HdbscanDenoise(hole_noise_removed).run_hdbscan()

def test_alphaShapeDenoise(obj_path: str):
    hddbscan_1 = HdbscanDenoise(obj_path).run_hdbscan()
    final_obj = AlphaShapeDenoise(hddbscan_1).alpha_shape_filter_auto()
    os.startfile(final_obj)

def check_path(obj_path: str):
    if not obj_path.endswith(".obj"):
        print("Must be a .obj file")
        return False

    if not os.path.exists(obj_path):
        print(f"Error: File '{obj_path}' does not exist.")
        return False

    return True


obj_list = [
    './noiseReductionSample/Sample1/3DModel.obj',
    # './noiseReductionSample/Sample2/3DModel.obj',
    # './noiseReductionSample/Sample3/supermug.obj',
    # './noiseReductionSample/Sample4/ultramug.obj'
]

if __name__ == '__main__':
    for obj_path in obj_list:
        if obj_path:
            if check_path(obj_path):
                main(obj_path)
            else:
                sys.exit(1)
