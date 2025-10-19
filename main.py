import os
import shutil
import sys

from noisereduction.HdbscanDenoise import HdbscanDenoise
from noisereduction.RansackDenoise import RansackDenoise


def main(obj_path: str):
    hd1Res = HdbscanDenoise(obj_path).run_hdbscan()
    ranSackModel = RansackDenoise(hd1Res)
    margin = ranSackModel.estimate_margin_from_histogram(normalize_result=False, show_cutoff_fig=True)
    ransackRes = ranSackModel.run_cuboid_detection(margin=margin)
    os.startfile(ransackRes)
    hd2Res = HdbscanDenoise(ransackRes).run_hdbscan()
    print(f'your final denoised file in {hd2Res}')
    output_dir = os.path.dirname(obj_path)
    dest_path = os.path.join(output_dir, os.path.basename(hd2Res))
    shutil.copy(hd2Res, dest_path)
    print(f"Your final denoised file has been copied to: {dest_path}")
    os.startfile(dest_path)


def check_path(obj_path: str):
    if not obj_path.endswith(".obj"):
        print("Must be a .obj file")
        return False

    if not os.path.exists(obj_path):
        print(f"Error: File '{obj_path}' does not exist.")
        return False

    return True


if __name__ == '__main__':
    obj_path = './noiseReductionSample/Sample3/supermug.obj'
    if obj_path:
        if check_path(obj_path):
            main(obj_path)
        else:
            sys.exit(1)
