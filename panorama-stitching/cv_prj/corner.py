# [핵심 기능] Harris Corner Detection 구현
# - Gradient 계산 (Sobel) -> Structure Tensor 생성 -> Corner Response 산출 -> NMS로 특징점 추출
# - Adaptive Thresholding과 Grid-based NMS를 적용하여 코너 분포 개선

import numpy as np
from preprocess import convolve2d, create_gaussian_kernel


# [기능] Sobel Filter를 이용한 Image Gradient (Ix, Iy) 계산
def compute_sobel_gradients(gray: np.ndarray):
    if gray.ndim != 2:
        raise ValueError("compute_sobel_gradients: 입력은 2D Grayscale 이미지여야 합니다.")

    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [ 0,  0,  0],
                        [ 1,  2,  1]], dtype=np.float32)

    Ix = convolve2d(gray, sobel_x)
    Iy = convolve2d(gray, sobel_y)

    return Ix, Iy


# [기능] Harris Corner Response R 계산 (R = det(M) - k * trace(M)^2)
def compute_harris_response(
    gray: np.ndarray,
    k: float = 0.04,
    window_size: int = 3,
    sigma: float = 1.0
) -> np.ndarray:
    gray = gray.astype(np.float32)

    Ix, Iy = compute_sobel_gradients(gray)

    Ixx = Ix * Ix
    Iyy = Iy * Iy
    Ixy = Ix * Iy

    gauss = create_gaussian_kernel(ksize=window_size, sigma=sigma)

    Sxx = convolve2d(Ixx, gauss)
    Syy = convolve2d(Iyy, gauss)
    Sxy = convolve2d(Ixy, gauss)

    detM = Sxx * Syy - Sxy * Sxy          
    traceM = Sxx + Syy                    
    R = detM - k * (traceM ** 2)         

    return R


# [기능] Non-Maximum Suppression (NMS) 및 Grid 기반 코너 추출
def non_maximum_suppression(
    R: np.ndarray,
    nms_radius: int = 2,
    threshold_ratio: float = 0.0,
    max_points: int | None = 500,
    edge_margin: int = 5,
    use_grid: bool = True,
    grid_rows: int = 8,
    grid_cols: int = 12,
) -> np.ndarray:
    
    H, W = R.shape

    R = np.asarray(R, dtype=np.float32)
    if not np.isfinite(R).all():
        R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

    # [Thresholding] 노이즈성 약한 코너 제거
    R_max = float(np.max(R)) if R.size > 0 else 0.0
    thr = float(threshold_ratio) * R_max

    keypoints = []
    scores = []

    y_min = edge_margin
    y_max = H - edge_margin
    x_min = edge_margin
    x_max = W - edge_margin

    for y in range(y_min, y_max):
        y0 = max(y - nms_radius, 0)
        y1 = min(y + nms_radius + 1, H)

        for x in range(x_min, x_max):
            center_value = float(R[y, x])

            if center_value <= thr:
                continue

            x0 = max(x - nms_radius, 0)
            x1 = min(x + nms_radius + 1, W)

            local = R[y0:y1, x0:x1]
            local_max = float(np.max(local))

            if center_value >= local_max:
                keypoints.append((y, x))
                scores.append(center_value)

    if len(keypoints) == 0:
        return np.zeros((0, 2), dtype=np.int32)

    keypoints = np.array(keypoints, dtype=np.int32)
    scores = np.array(scores, dtype=np.float32)

    # [Grid-based Sampling] 이미지 전역에서 고르게 코너 추출 (Feature Distribution 개선)
    if (max_points is not None) and (len(keypoints) > max_points):
        if use_grid and grid_rows > 0 and grid_cols > 0:
            cell_h = max(1.0, (H - 2 * edge_margin) / float(grid_rows))
            cell_w = max(1.0, (W - 2 * edge_margin) / float(grid_cols))

            per_cell = int(np.ceil(max_points / float(grid_rows * grid_cols)))
            per_cell = max(1, per_cell)

            picked_idx = []
            for gr in range(grid_rows):
                y0 = edge_margin + int(gr * cell_h)
                y1 = edge_margin + int((gr + 1) * cell_h)
                for gc in range(grid_cols):
                    x0 = edge_margin + int(gc * cell_w)
                    x1 = edge_margin + int((gc + 1) * cell_w)

                    in_cell = np.where(
                        (keypoints[:, 0] >= y0) & (keypoints[:, 0] < y1) &
                        (keypoints[:, 1] >= x0) & (keypoints[:, 1] < x1)
                    )[0]
                    if in_cell.size == 0:
                        continue

                    order = np.argsort(-scores[in_cell])
                    take = in_cell[order[:per_cell]]
                    picked_idx.append(take)

            if len(picked_idx) > 0:
                picked_idx = np.concatenate(picked_idx, axis=0)
                order = np.argsort(-scores[picked_idx])
                picked_idx = picked_idx[order[:max_points]]
                keypoints = keypoints[picked_idx]
            else:
                order = np.argsort(-scores)
                keypoints = keypoints[order[:max_points]]
        else:
            order = np.argsort(-scores)
            keypoints = keypoints[order[:max_points]]

    return keypoints


# [기능] 전체 Corner Detection 파이프라인 (Response 계산 -> NMS)
def harris_corners(
    gray: np.ndarray,
    k: float = 0.04,
    window_size: int = 3,
    sigma: float = 1.0,
    nms_radius: int = 2,
    threshold_ratio: float = 0.01,
    max_points: int | None = 500,
    edge_margin: int = 5
) -> np.ndarray:
    
    R = compute_harris_response(
        gray,
        k=k,
        window_size=window_size,
        sigma=sigma
    )

    keypoints = non_maximum_suppression(
        R,
        nms_radius=nms_radius,
        threshold_ratio=threshold_ratio,
        max_points=max_points,
        edge_margin=edge_margin
    )

    return keypoints