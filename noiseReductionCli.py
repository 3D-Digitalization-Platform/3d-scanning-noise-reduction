import os
import shutil
import sys

from noisereduction.HdbscanDenoise import HdbscanDenoise
from noisereduction.RansackDenoise import RansackDenoise


def main(obj_path: str):
    hd1Res = HdbscanDenoise(obj_path).run_hdbscan()
    ransack1 = RansackDenoise(hd1Res).run_ransack()
    hd2Res = HdbscanDenoise(ransack1).run_hdbscan()
    print(f'your final denoised file in {hd2Res}')
    output_dir = os.path.dirname(obj_path)
    dest_path = os.path.join(output_dir, os.path.basename(hd2Res))
    shutil.copy(hd2Res, dest_path)
    print(f"Your final denoised file has been copied to: {dest_path}")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python main.py <path_to_obj_file>")
        sys.exit(1)

    obj_path = sys.argv[1]

    if not obj_path.endswith(".obj"):
        print("Must be a .obj file")
        sys.exit(1)

    if not os.path.exists(obj_path):
        print(f"Error: File '{obj_path}' does not exist.")
        sys.exit(1)

    main(obj_path)