import os
import sys

from noisereduction.RansackDenoise import RansackDenoise


def main(obj_path: str):
    # os.startfile(obj_path)
    res = RansackDenoise(obj_path).detect_noise_by_ransac_plane()
    os.startfile(res)
    # res=RansackDenoise('noiseReductionSample/Sample2/3DModel_noise_reduced_ransack_noise_reduced_noise_by_y_range_noise_reduced.obj').detect_noise_by_y_range()
    # res=RansackDenoise(res).detect_noise_by_y_range()
    # print(res)
    # os.startfile(ranSackModel)
    # hd1Res = HdbscanDenoise(obj_path).run_hdbscan()
    # ranSackModel = RansackDenoise(hd1Res).detect_noise_enclosed_shape(bin_size=0.01, high_quantile=0.92)
    # os.startfile(ranSackModel)
    # ranSackModel2 = RansackDenoise(ranSackModel).run_ransack()
    # os.startfile(ranSackModel2)
    # hd2Res = HdbscanDenoise(ranSackModel2).run_hdbscan()
    #
    # output_dir = os.path.dirname(obj_path)
    # dest_path = os.path.join(output_dir, os.path.basename(hd2Res))
    # shutil.copy(hd2Res, dest_path)
    # print(f"Your final denoised file has been copied to: {dest_path}")
    # os.startfile(dest_path)

    # ranSackModel = RansackDenoise(hd1Res)
    # margin = ranSackModel.estimate_margin_from_histogram(normalize_result=False, show_cutoff_fig=True)
    # ransackRes = ranSackModel.run_cuboid_detection(margin=margin)
    # os.startfile(ransackRes)
    # hd2Res = HdbscanDenoise(ransackRes).run_hdbscan()
    # print(f'your final denoised file in {hd2Res}')
    # output_dir = os.path.dirname(obj_path)
    # dest_path = os.path.join(output_dir, os.path.basename(hd2Res))
    # shutil.copy(hd2Res, dest_path)
    # print(f"Your final denoised file has been copied to: {dest_path}")
    # os.startfile(dest_path)


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
    './noiseReductionSample/Sample2/3DModel.obj',
    './noiseReductionSample/Sample3/supermug.obj',
    './noiseReductionSample/Sample4/ultramug.obj'
]

if __name__ == '__main__':
    for obj_path in obj_list:
        if obj_path:
            if check_path(obj_path):
                main(obj_path)
            else:
                sys.exit(1)
