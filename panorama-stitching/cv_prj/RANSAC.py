# [핵심 기능] RANSAC 알고리즘을 이용한 Robust Homography Estimation
# - Outlier 제거: 무작위 샘플링과 에러 측정을 반복하여 이상치 제거
# - Model Refinement: 최적의 Inlier 집합으로 최종 Homography 재계산

from typing import Tuple
import numpy as np
from homography import compute_homography, compute_reprojection_errors


# [기능] RANSAC을 수행하여 최적의 Homography와 Inlier 집합 반환
def ransac_homography(
    pts1: np.ndarray,
    pts2: np.ndarray,
    n_iter: int = 2000,
    thresh: float = 3.0,
    min_inliers: int = 20,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    
    if pts1.ndim != 2 or pts2.ndim != 2:
        raise ValueError("ransac_homography: pts1, pts2는 (N, 2) 모양의 2D 배열이어야 합니다.")
    if pts1.shape != pts2.shape:
        raise ValueError("ransac_homography: pts1, pts2는 같은 shape이어야 합니다.")
    N = pts1.shape[0]
    if N < 4:
        raise ValueError(f"ransac_homography: 최소 4개의 대응점이 필요합니다. 현재 N={N}")

    rng = np.random.default_rng(random_state)

    best_H: np.ndarray | None = None
    best_inliers: np.ndarray | None = None
    best_count: int = 0

    # [반복] n_iter 횟수만큼 랜덤 샘플링 및 모델 검증 수행
    for _ in range(n_iter):
        # [샘플링] 4개의 점을 무작위로 선택
        sample_idx = rng.choice(N, size=4, replace=False)
        sample_pts1 = pts1[sample_idx]
        sample_pts2 = pts2[sample_idx]

        try:
            H_candidate = compute_homography(sample_pts1, sample_pts2)
        except Exception:
            continue

        # [검증] 전체 점들에 대한 Reprojection Error 계산 및 Inlier 개수 확인
        errors = compute_reprojection_errors(pts1, pts2, H_candidate)
        inliers = errors < thresh
        count = int(np.sum(inliers))

        if count > best_count:
            best_count = count
            best_inliers = inliers
            best_H = H_candidate

    if best_count < min_inliers:
        print(f"⚠ RANSAC found only {best_count} inliers (< min_inliers={min_inliers}). REJECTED.")
        return None, None 

    # [재추정] 선택된 Inlier들만을 사용하여 Homography 정밀 보정
    H_refined = compute_homography(pts1[best_inliers], pts2[best_inliers])
    
    return H_refined, best_inliers