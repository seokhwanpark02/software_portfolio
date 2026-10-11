# [핵심 기능] Homography Matrix 계산 모듈 (Normalized DLT)
# - Normalized DLT: 좌표 정규화 후 DLT 수행 (수치 안정성)
# - Reprojection Error: 변환된 좌표와 실제 좌표 간의 오차 계산 (RANSAC용)

from typing import Tuple
import numpy as np


# [기능] 좌표 정규화 행렬 계산 (Centroid를 원점으로, 평균 거리를 sqrt(2)로)
def _compute_normalization_matrix(pts: np.ndarray) -> np.ndarray:
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("_compute_normalization_matrix: pts는 (N, 2) 모양이어야 합니다.")

    mean = np.mean(pts, axis=0)
    diff = pts - mean
    dists = np.linalg.norm(diff, axis=1)
    mean_dist = np.mean(dists)

    if mean_dist < 1e-8:
        tx, ty = -mean[0], -mean[1]
        T = np.array([
            [1.0, 0.0, tx],
            [0.0, 1.0, ty],
            [0.0, 0.0, 1.0],
        ], dtype=np.float64)
        return T

    scale = np.sqrt(2.0) / mean_dist
    tx = -scale * mean[0]
    ty = -scale * mean[1]

    T = np.array([
        [scale,   0.0, tx],
        [0.0,   scale, ty],
        [0.0,    0.0, 1.0],
    ], dtype=np.float64)

    return T


# [기능] 좌표 정규화 수행 (x' = T @ x)
def _normalize_points(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    T = _compute_normalization_matrix(pts)

    ones = np.ones((pts.shape[0], 1), dtype=np.float64)
    pts_h = np.hstack([pts.astype(np.float64), ones]) 

    pts_h_norm = (T @ pts_h.T).T 
    pts_norm = pts_h_norm[:, 0:2] / pts_h_norm[:, 2:3]

    return pts_norm, T


# [기능] DLT 알고리즘을 위한 행렬 A 구성
def _build_dlt_matrix(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    N = pts1.shape[0]
    A = []

    for i in range(N):
        x1, y1 = pts1[i]
        x2, y2 = pts2[i]

        row1 = [-x1, -y1, -1.0,  0.0,  0.0,  0.0, x1 * x2, y1 * x2, x2]
        row2 = [ 0.0,  0.0,  0.0, -x1, -y1, -1.0, x1 * y2, y1 * y2, y2]

        A.append(row1)
        A.append(row2)

    A = np.array(A, dtype=np.float64)
    return A


# [기능] Normalized DLT를 이용한 Homography 추정
def compute_homography(pts1: np.ndarray, pts2: np.ndarray) -> np.ndarray:
    if pts1.ndim != 2 or pts2.ndim != 2:
        raise ValueError("compute_homography: pts1, pts2는 (N, 2) 모양의 2D 배열이어야 합니다.")
    if pts1.shape != pts2.shape:
        raise ValueError("compute_homography: pts1, pts2는 같은 shape이어야 합니다.")
    if pts1.shape[0] < 4:
        raise ValueError("compute_homography: 최소 4개의 대응점이 필요합니다. 현재 N={}".format(pts1.shape[0]))

    # [정규화] 입력 좌표 정규화
    pts1_norm, T1 = _normalize_points(pts1)
    pts2_norm, T2 = _normalize_points(pts2)

    # [DLT] SVD를 이용해 최소 자승 해(h) 구하기
    A = _build_dlt_matrix(pts1_norm, pts2_norm)
    U, S, Vt = np.linalg.svd(A)
    h = Vt[-1]

    H_norm = h.reshape(3, 3)

    # [복원] 정규화된 H를 원래 좌표계로 변환 (H = inv(T2) @ H_norm @ T1)
    T2_inv = np.linalg.inv(T2)
    H = T2_inv @ H_norm @ T1

    if abs(H[2, 2]) < 1e-8:
        max_abs = np.max(np.abs(H))
        if max_abs > 0:
            H /= max_abs
    else:
        H /= H[2, 2]

    return H.astype(np.float64)


# [기능] Homography 변환 적용 (점 투영)
def apply_homography(pts: np.ndarray, H: np.ndarray) -> np.ndarray:
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("apply_homography: pts는 (N, 2) 모양이어야 합니다.")
    if H.shape != (3, 3):
        raise ValueError("apply_homography: H는 (3, 3) 행렬이어야 합니다.")

    N = pts.shape[0]

    ones = np.ones((N, 1), dtype=np.float64)
    pts_h = np.hstack([pts.astype(np.float64), ones])

    pts_h_proj = (H @ pts_h.T).T
    w = pts_h_proj[:, 2]

    eps = 1e-8
    w_safe = np.where(np.abs(w) < eps, eps * np.sign(w + eps), w)

    xs = pts_h_proj[:, 0] / w_safe
    ys = pts_h_proj[:, 1] / w_safe

    pts_proj = np.stack([xs, ys], axis=1)
    return pts_proj


# [기능] Reprojection Error 계산 (RANSAC Inlier 판단용)
def compute_reprojection_errors(
    pts_src: np.ndarray,
    pts_dst: np.ndarray,
    H: np.ndarray
) -> np.ndarray:
    if pts_src.shape != pts_dst.shape:
        raise ValueError("compute_reprojection_errors: pts_src, pts_dst는 같은 shape이어야 합니다.")
    if H.shape != (3, 3):
        raise ValueError("compute_reprojection_errors: H는 (3, 3) 행렬이어야 합니다.")

    pts_proj = apply_homography(pts_src, H)
    diff = pts_proj - pts_dst
    errors = np.linalg.norm(diff, axis=1)

    return errors.astype(np.float64)