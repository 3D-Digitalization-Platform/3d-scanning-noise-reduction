# scan_quality_check.py
import numpy as np
import open3d as o3d


def bbox_surface_area(pcd):
    a = pcd.get_axis_aligned_bounding_box()
    minb = a.min_bound;
    maxb = a.max_bound
    dx, dy, dz = maxb - minb
    return 2 * (dx * dy + dy * dz + dz * dx), np.linalg.norm([dx, dy, dz])


def compute_knn_dists(points, k=8):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    dists = np.zeros(len(points))
    for i, p in enumerate(points):
        _, idx, dist2 = pcd_tree.search_knn_vector_3d(p, k)
        # dist2 includes squared distances
        if len(dist2) >= 2:
            dists[i] = np.sqrt(np.mean(dist2[1:]))  # ignore self distance
        else:
            dists[i] = 1e9
    return dists


def compute_curvature(points, k=20):
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    pcd_tree = o3d.geometry.KDTreeFlann(pcd)
    curvatures = np.zeros(len(points))
    for i, p in enumerate(points):
        _, idx, _ = pcd_tree.search_knn_vector_3d(p, k)
        if len(idx) < 4:
            curvatures[i] = 0
            continue
        nbr = points[idx]
        cov = np.cov(nbr.T)
        eig = np.linalg.eigvalsh(cov)
        s = np.sum(eig)
        if s <= 0:
            curvatures[i] = 0
        else:
            curvatures[i] = eig[0] / s
    return curvatures


def largest_cluster_ratio(points, eps=0.01, min_points=30):
    # DBSCAN from open3d
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    labels = np.array(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.size == 0:
        return 1.0
    max_label = labels.max()
    if max_label < 0:
        return 0.0
    counts = [(labels == i).sum() for i in range(max_label + 1)]
    return max(counts) / float(len(points))


def quality_score_from_pcd(pcd,
                           density_target=1.0,  # tuned per dataset (points per bbox area)
                           outlier_frac_thresh=0.05,
                           curv_target=0.01,
                           cluster_eps_ratio=0.01):
    points = np.asarray(pcd.points)
    if points.shape[0] < 10:
        return {'score': 0.0, 'reason': 'Too few points', 'metrics': {}}

    # bbox props
    surface_area, diag = bbox_surface_area(pcd)
    density = len(points) / (surface_area + 1e-12)

    # knn dists
    dists = compute_knn_dists(points, k=8)
    dmean, dstd = dists.mean(), dists.std()
    outliers = np.where(dists > dmean + 2.5 * dstd)[0]
    outlier_frac = len(outliers) / len(points)

    # curvature
    curv = compute_curvature(points, k=20)
    curv_mean = np.mean(curv)
    curv_90 = np.percentile(curv, 90)

    # largest cluster ratio - convert eps relative to diag
    eps = cluster_eps_ratio * diag
    lcr = largest_cluster_ratio(points, eps=eps, min_points=20)

    # normalize metrics into 0..1 (higher is better)
    D_norm = min(1.0, density / density_target)
    O_norm = 1.0 - min(1.0, outlier_frac / outlier_frac_thresh)
    R_norm = 1.0 - min(1.0, curv_90 / max(curv_target, 1e-9))
    C_norm = min(1.0, lcr / 0.6)

    # weights
    wD, wO, wR, wC = 0.15, 0.30, 0.25, 0.30
    score = wD * D_norm + wO * O_norm + wR * R_norm + wC * C_norm

    metrics = {
        'N_points': len(points),
        'bbox_area': surface_area,
        'bbox_diag': diag,
        'density': density,
        'D_norm': D_norm,
        'outlier_frac': outlier_frac,
        'O_norm': O_norm,
        'curv_90': curv_90,
        'R_norm': R_norm,
        'largest_cluster_ratio': lcr,
        'C_norm': C_norm
    }
    return {'score': float(score), 'metrics': metrics}


obj_list = [
    './noiseReductionSample/Sample1/3DModel.obj',
    './noiseReductionSample/Sample2/3DModel.obj',
    './noiseReductionSample/Sample3/supermug.obj',
    './noiseReductionSample/Sample4/ultramug.obj'
]


def score_obj(src):
    mesh = o3d.io.read_triangle_mesh(src)
    pcd = mesh.sample_points_uniformly(number_of_points=200000)
    result = quality_score_from_pcd(pcd,
                                    density_target=0.8,  # example tune
                                    outlier_frac_thresh=0.05,
                                    curv_target=0.005,
                                    cluster_eps_ratio=0.02)
    print("Score:", result['score'])
    print("Metrics:")
    for k, v in result['metrics'].items():
        print(f"  {k}: {v}")

    score = result['score']
    if score >= 0.85:
        print("Excellent")
    elif score >= 0.7:
        print("Accepted")
    elif score >= 0.5:
        print("Needs improvement")
    else:
        print("REJECT")


# Example usage:
if __name__ == "__main__":
    for obj in obj_list:
        print(f'===={obj}====')
        score_obj(obj)
