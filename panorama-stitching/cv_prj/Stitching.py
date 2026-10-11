# [핵심 기능] 파노라마 스티칭 및 블렌딩 (Warping & Blending)
# - Inverse Warping: 캔버스 좌표를 원본 이미지 좌표로 역투영하여 픽셀 값 샘플링
# - Blending: 중심부 가중치(Feathering)와 선명한 픽셀 우선(Dominance Rule)을 결합하여 고스트 현상 최소화

from typing import Tuple
import numpy as np


# [기능] Bilinear Interpolation을 이용한 픽셀 샘플링
def bilinear_interpolate(img: np.ndarray, x: float, y: float) -> np.ndarray:
    if img.ndim == 2:
        H, W = img.shape
        C = None
    elif img.ndim == 3:
        H, W, C = img.shape
    else:
        raise ValueError("bilinear_interpolate: img는 2D 또는 3D 배열이어야 합니다.")

    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1

    if x0 < 0 or x1 >= W or y0 < 0 or y1 >= H:
        x0_clamped = min(max(x0, 0), W - 1)
        y0_clamped = min(max(y0, 0), H - 1)
        return img[y0_clamped, x0_clamped].astype(np.float32)

    Ia = img[y0, x0].astype(np.float32)
    Ib = img[y0, x1].astype(np.float32)
    Ic = img[y1, x0].astype(np.float32)
    Id = img[y1, x1].astype(np.float32)

    wa = (x1 - x) * (y1 - y)
    wb = (x - x0) * (y1 - y)
    wc = (x1 - x) * (y - y0)
    wd = (x - x0) * (y - y0)

    if C is None:
        value = wa * Ia + wb * Ib + wc * Ic + wd * Id
    else:
        value = wa * Ia + wb * Ib + wc * Ic + wd * Id

    return value


# [기능] 이미지의 4개 모서리 좌표 반환
def _get_image_corners(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        H, W = img.shape
    else:
        H, W = img.shape[:2]

    corners = np.array([
        [0,     0],
        [W - 1, 0],
        [W - 1, H - 1],
        [0,     H - 1],
    ], dtype=np.float64)
    return corners


# [기능] 두 이미지 스티칭을 위한 캔버스 크기 및 변환 행렬 계산
def compute_panorama_canvas(
    img1: np.ndarray,
    img2: np.ndarray,
    H_1_to_2: np.ndarray,
    margin: int = 0
) -> Tuple[Tuple[int, int], np.ndarray, np.ndarray]:
    
    corners1 = _get_image_corners(img1)
    corners2 = _get_image_corners(img2)

    ones = np.ones((corners1.shape[0], 1), dtype=np.float64)
    corners1_h = np.hstack([corners1, ones])
    projected1_h = (H_1_to_2 @ corners1_h.T).T

    projected1 = projected1_h[:, :2] / projected1_h[:, 2:3]

    all_points = np.vstack([projected1, corners2])

    x_coords = all_points[:, 0]
    y_coords = all_points[:, 1]

    x_min = x_coords.min()
    x_max = x_coords.max()
    y_min = y_coords.min()
    y_max = y_coords.max()

    tx = -x_min + margin
    ty = -y_min + margin

    T = np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    W_canvas = int(np.ceil(x_max - x_min)) + 2 * margin
    H_canvas = int(np.ceil(y_max - y_min)) + 2 * margin

    canvas_shape = (H_canvas, W_canvas)

    H_canvas_1 = T @ H_1_to_2
    H_canvas_2 = T

    return canvas_shape, H_canvas_1, H_canvas_2


# [기능] 다중 이미지 스티칭을 위한 전체 캔버스 크기 계산 (왜곡 보정 포함)
def compute_panorama_canvas_multi(
    imgs: list[np.ndarray],
    H_to_ref_list: list[np.ndarray],
    ref_index: int = 0,
    margin: int = 0,
    global_weights: list[float] | None = None,
    area_ratio_max: float = 100.0, 
    scale_max: float = 20.0,
) -> tuple[tuple[int, int], list[np.ndarray]]:
    
    num_imgs = len(imgs)
    if global_weights is None:
        global_weights = [1.0] * num_imgs

    projected_list: list[np.ndarray | None] = [None] * num_imgs
    valid_flags = [False] * num_imgs

    for i in range(num_imgs):
        if float(global_weights[i]) <= 0.0:
            continue

        img = imgs[i]
        corners = _get_image_corners(img)
        H_i_to_ref = np.eye(3, dtype=np.float64) if i == ref_index else H_to_ref_list[i].astype(np.float64)

        if i == ref_index:
            projected = corners.astype(np.float64)
        else:
            ones = np.ones((4, 1), dtype=np.float64)
            corners_h = np.hstack([corners.astype(np.float64), ones])
            proj_h = (H_i_to_ref @ corners_h.T).T

            w = proj_h[:, 2]
            ok = np.isfinite(w) & (w > 1e-4)
            if not np.all(ok):
                continue

            projected = proj_h[:, :2] / proj_h[:, 2:3]

        if not np.all(np.isfinite(projected)):
            continue

        x_min, x_max = float(projected[:, 0].min()), float(projected[:, 0].max())
        y_min, y_max = float(projected[:, 1].min()), float(projected[:, 1].max())

        bw = max(1.0, x_max - x_min)
        bh = max(1.0, y_max - y_min)
        area = bw * bh
        
        img_h, img_w = img.shape[:2]
        orig_area = float(img_h * img_w)
        
        # [예외처리] 과도한 왜곡(면적 폭발) 발생 시 캔버스 계산에서 제외
        if area > orig_area * 500.0:
             continue

        projected_list[i] = projected
        valid_flags[i] = True

    if not any(valid_flags):
        valid_flags[ref_index] = True
        projected_list[ref_index] = _get_image_corners(imgs[ref_index]).astype(np.float64)
    
    all_points = []
    for i in range(num_imgs):
        if not valid_flags[i]:
            continue
        if projected_list[i] is None:
            continue
        all_points.append(projected_list[i])

    all_points = np.vstack(all_points)

    x_coords = all_points[:, 0]
    y_coords = all_points[:, 1]

    x_min = float(x_coords.min())
    x_max = float(x_coords.max())
    y_min = float(y_coords.min())
    y_max = float(y_coords.max())

    tx = -x_min + margin
    ty = -y_min + margin

    T = np.array([
        [1.0, 0.0, tx],
        [0.0, 1.0, ty],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    W_canvas = int(np.ceil(x_max - x_min)) + 2 * margin
    H_canvas = int(np.ceil(y_max - y_min)) + 2 * margin
    canvas_shape = (H_canvas, W_canvas)

    H_canvas_list: list[np.ndarray] = []
    for i in range(num_imgs):
        H_i_to_ref = np.eye(3, dtype=np.float64) if i == ref_index else H_to_ref_list[i].astype(np.float64)
        H_canvas_list.append(T @ H_i_to_ref)

    return canvas_shape, H_canvas_list


# [기능] Warping된 이미지를 파노라마 캔버스에 누적 (Weighted Blending)
def _accumulate_warped_image_into_panorama(
    pano_accum: np.ndarray,
    weight: np.ndarray,
    img: np.ndarray,
    H_dst_src: np.ndarray,
    dst_bounds: tuple[int, int, int, int],
    global_gain: float,
    max_block_pixels: int = 5_000_000,
) -> None:
    
    y_min, y_max, x_min, x_max = dst_bounds
    H_canvas, W_canvas = weight.shape

    y0 = max(0, int(y_min)); y1 = min(H_canvas, int(y_max))
    x0 = max(0, int(x_min)); x1 = min(W_canvas, int(x_max))
    if y1 <= y0 or x1 <= x0:
        return

    if img.ndim == 2:
        H_src, W_src = img.shape
        C = 1
    else:
        H_src, W_src, C = img.shape

    Hm = H_dst_src.astype(np.float32)
    a, b, c = Hm[0, 0], Hm[0, 1], Hm[0, 2]
    d, e, f = Hm[1, 0], Hm[1, 1], Hm[1, 2]
    g, h, i_ = Hm[2, 0], Hm[2, 1], Hm[2, 2]

    W = x1 - x0
    H = y1 - y0
    rows_per_block = max(1, int(max_block_pixels // max(1, W)))
    x_coords = np.arange(x0, x1, dtype=np.float32)

    feather_floor = 0.0
    feather_gamma = 1.5        
    dominance_ratio = 0.5

    denom_x = max(1.0, 0.5 * float(W_src - 1))
    denom_y = max(1.0, 0.5 * float(H_src - 1))

    for yb in range(y0, y1, rows_per_block):
        ye = min(y1, yb + rows_per_block)
        rows = ye - yb
        if rows <= 0:
            continue

        xs = np.tile(x_coords, rows)
        ys = np.repeat(np.arange(yb, ye, dtype=np.float32), W)

        w = g * xs + h * ys + i_
        valid = np.isfinite(w) & (np.abs(w) > 1e-8)
        if not np.any(valid):
            continue

        invw = np.empty_like(w, dtype=np.float32)
        invw[valid] = 1.0 / w[valid]

        x_s = (a * xs + b * ys + c) * invw
        y_s = (d * xs + e * ys + f) * invw

        valid &= (x_s >= 0.0) & (x_s <= (W_src - 2)) & (y_s >= 0.0) & (y_s <= (H_src - 2))
        if not np.any(valid):
            continue

        xs_v = x_s[valid]
        ys_v = y_s[valid]

        x0_s = np.floor(xs_v).astype(np.int32)
        y0_s = np.floor(ys_v).astype(np.int32)
        x1_s = x0_s + 1
        y1_s = y0_s + 1

        dx = xs_v - x0_s.astype(np.float32)
        dy = ys_v - y0_s.astype(np.float32)

        x_d = xs[valid].astype(np.int32)
        y_d = ys[valid].astype(np.int32)

        # [가중치] Feathering Weight 계산 (중심부일수록 가중치 높음)
        dist_l = xs_v
        dist_r = (W_src - 1) - xs_v
        dist_t = ys_v
        dist_b = (H_src - 1) - ys_v

        wx = np.minimum(dist_l, dist_r) / denom_x
        wy = np.minimum(dist_t, dist_b) / denom_y
        w_feather = np.minimum(wx, wy)
        w_feather = np.clip(w_feather, 0.0, 1.0)
        
        w_total = (w_feather ** feather_gamma) * float(global_gain)
        w_total = w_total.astype(np.float32)

        # [블렌딩] Dominance Rule 적용 (Ghosting 방지)
        w_old = weight[y_d, x_d]
        keep = (w_old < 1e-4) | (w_total > (w_old * dominance_ratio))
        
        if not np.any(keep):
            continue

        x_d = x_d[keep]
        y_d = y_d[keep]
        x0_s = x0_s[keep]; y0_s = y0_s[keep]
        x1_s = x1_s[keep]; y1_s = y1_s[keep]
        dx = dx[keep]; dy = dy[keep]
        w_total = w_total[keep]

        if C == 1:
            I00 = img[y0_s, x0_s].astype(np.float32)
            I10 = img[y0_s, x1_s].astype(np.float32)
            I01 = img[y1_s, x0_s].astype(np.float32)
            I11 = img[y1_s, x1_s].astype(np.float32)

            vals = (I00 * (1.0 - dx) * (1.0 - dy) +
                    I10 * dx * (1.0 - dy) +
                    I01 * (1.0 - dx) * dy +
                    I11 * dx * dy)

            pano_accum[y_d, x_d] += vals * w_total
        else:
            I00 = img[y0_s, x0_s, :].astype(np.float32)
            I10 = img[y0_s, x1_s, :].astype(np.float32)
            I01 = img[y1_s, x0_s, :].astype(np.float32)
            I11 = img[y1_s, x1_s, :].astype(np.float32)

            w00 = ((1.0 - dx) * (1.0 - dy))[:, None]
            w10 = (dx * (1.0 - dy))[:, None]
            w01 = ((1.0 - dx) * dy)[:, None]
            w11 = (dx * dy)[:, None]
            vals = I00 * w00 + I10 * w10 + I01 * w01 + I11 * w11

            pano_accum[y_d, x_d, :] += vals * w_total[:, None]

        weight[y_d, x_d] += w_total


# [기능] 단일 이미지 워핑 (테스트용)
def warp_image_with_homography(
    img: np.ndarray,
    H_dst_src: np.ndarray,
    out_shape: Tuple[int, int],
    dst_bounds: Tuple[int, int, int, int] | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if img.ndim == 2:
        H_src, W_src = img.shape
        C = 1
    else:
        H_src, W_src, C = img.shape

    H_dst, W_dst = out_shape

    if C == 1:
        warped = np.zeros((H_dst, W_dst), dtype=np.float32)
    else:
        warped = np.zeros((H_dst, W_dst, C), dtype=np.float32)
    mask = np.zeros((H_dst, W_dst), dtype=bool)

    if dst_bounds is None:
        y0, y1, x0, x1 = 0, H_dst, 0, W_dst
    else:
        y_min, y_max, x_min, x_max = dst_bounds
        y0 = max(0, int(y_min))
        y1 = min(H_dst, int(y_max))
        x0 = max(0, int(x_min))
        x1 = min(W_dst, int(x_max))

    if y1 <= y0 or x1 <= x0:
        return warped, mask

    h_box = y1 - y0
    w_box = x1 - x0
    ys, xs = np.indices((h_box, w_box), dtype=np.float64)
    ys += y0
    xs += x0

    ones = np.ones_like(xs)
    dst_pts = np.stack([xs.ravel(), ys.ravel(), ones.ravel()], axis=0)
    src_pts = H_dst_src @ dst_pts

    w = src_pts[2, :]
    valid_w = (w > 1e-6)

    x_s = np.empty_like(w, dtype=np.float64)
    y_s = np.empty_like(w, dtype=np.float64)
    
    w_safe = w.copy()
    w_safe[~valid_w] = 1.0 
    
    x_s = src_pts[0, :] / w_safe
    y_s = src_pts[1, :] / w_safe

    valid = valid_w.copy()
    valid &= (x_s >= 0) & (x_s <= W_src - 2) & (y_s >= 0) & (y_s <= H_src - 2)

    if not np.any(valid):
        return warped, mask

    x_s_valid = x_s[valid]
    y_s_valid = y_s[valid]

    x0_s = np.floor(x_s_valid).astype(int)
    y0_s = np.floor(y_s_valid).astype(int)
    x1_s = x0_s + 1
    y1_s = y0_s + 1

    dx = x_s_valid - x0_s
    dy = y_s_valid - y0_s

    if C == 1:
        I00 = img[y0_s, x0_s].astype(np.float32)
        I10 = img[y0_s, x1_s].astype(np.float32)
        I01 = img[y1_s, x0_s].astype(np.float32)
        I11 = img[y1_s, x1_s].astype(np.float32)

        w00 = (1.0 - dx) * (1.0 - dy)
        w10 = dx * (1.0 - dy)
        w01 = (1.0 - dx) * dy
        w11 = dx * dy

        vals = I00 * w00 + I10 * w10 + I01 * w01 + I11 * w11
    else:
        I00 = img[y0_s, x0_s, :].astype(np.float32)
        I10 = img[y0_s, x1_s, :].astype(np.float32)
        I01 = img[y1_s, x0_s, :].astype(np.float32)
        I11 = img[y1_s, x1_s, :].astype(np.float32)

        w00 = ((1.0 - dx) * (1.0 - dy))[:, None]
        w10 = (dx * (1.0 - dy))[:, None]
        w01 = ((1.0 - dx) * dy)[:, None]
        w11 = (dx * dy)[:, None]

        vals = I00 * w00 + I10 * w10 + I01 * w01 + I11 * w11

    xs_flat = xs.ravel()
    ys_flat = ys.ravel()
    x_d_valid = xs_flat[valid].astype(int)
    y_d_valid = ys_flat[valid].astype(int)

    if C == 1:
        warped[y_d_valid, x_d_valid] = vals
    else:
        warped[y_d_valid, x_d_valid, :] = vals

    mask[y_d_valid, x_d_valid] = True

    return warped, mask


# [기능] 두 이미지 스티칭 (단순 합성)
def stitch_two_images(
    img1: np.ndarray,
    img2: np.ndarray,
    H_1_to_2: np.ndarray,
    margin: int = 0
) -> np.ndarray:
    canvas_shape, H_canvas_1, H_canvas_2 = compute_panorama_canvas(
        img1, img2, H_1_to_2, margin=margin
    )

    H_canvas, W_canvas = canvas_shape
    H_dst_to_img1 = np.linalg.inv(H_canvas_1)
    H_dst_to_img2 = np.linalg.inv(H_canvas_2)

    warped1, mask1 = warp_image_with_homography(
        img1, H_dst_to_img1, out_shape=canvas_shape
    )
    warped2, mask2 = warp_image_with_homography(
        img2, H_dst_to_img2, out_shape=canvas_shape
    )

    if img1.ndim == 2:
        pano = np.zeros((H_canvas, W_canvas), dtype=np.float32)
        C = 1
    else:
        C = img1.shape[2]
        pano = np.zeros((H_canvas, W_canvas, C), dtype=np.float32)

    only1 = mask1 & ~mask2
    only2 = ~mask1 & mask2
    both  = mask1 & mask2

    if not np.any(both):
        if C == 1:
            pano[only1] = warped1[only1]
            pano[only2] = warped2[only2]
        else:
            pano[only1, :] = warped1[only1, :]
            pano[only2, :] = warped2[only2, :]
        return pano

    if C == 1:
        diff = (warped1 - warped2) ** 2
        diff[~both] = 0.0
        col_error = diff.sum(axis=0)
    else:
        diff = (warped1 - warped2) ** 2
        diff[~both, :] = 0.0
        col_error = diff.sum(axis=(0, 2))

    overlap_cols = np.where(col_error > 0)[0]
    if overlap_cols.size == 0:
        if C == 1:
            pano[only1] = warped1[only1]
            pano[only2] = warped2[only2]
            pano[both]  = 0.5 * warped1[both] + 0.5 * warped2[both]
        else:
            pano[only1, :] = warped1[only1, :]
            pano[only2, :] = warped2[only2, :]
            pano[both, :]  = 0.5 * warped1[both, :] + 0.5 * warped2[both, :]
        return pano

    seam_col = overlap_cols[np.argmin(col_error[overlap_cols])]

    xs = np.arange(W_canvas, dtype=np.int32)[None, :]
    left_region  = xs <= seam_col
    right_region = xs > seam_col

    if C == 1:
        left_mask  = mask1 & left_region
        right_mask = mask2 & right_region
        pano[left_mask]  = warped1[left_mask]
        pano[right_mask] = warped2[right_mask]
    else:
        left_mask  = mask1 & left_region
        right_mask = mask2 & right_region
        pano[left_mask, :]  = warped1[left_mask, :]
        pano[right_mask, :] = warped2[right_mask, :]

    band_width = 5
    x_left  = max(seam_col - band_width, 0)
    x_right = min(seam_col + band_width, W_canvas - 1)
    band_region = (xs >= x_left) & (xs <= x_right)
    band_mask   = both & band_region

    if np.any(band_mask):
        alpha = (xs.astype(np.float32) - float(x_left)) / max(float(x_right - x_left), 1.0)
        alpha = np.clip(alpha, 0.0, 1.0)
        alpha = np.broadcast_to(alpha, (H_canvas, W_canvas))  # (1, W) -> (H, W): band_mask로 인덱싱하기 위해 높이 방향 확장
        if C == 1:
            a = alpha
            pano[band_mask] = (
                (1.0 - a)[band_mask] * warped1[band_mask]
                + a[band_mask] * warped2[band_mask]
            )
        else:
            a = alpha[..., None]
            pano[band_mask, :] = (
                (1.0 - a)[band_mask, :] * warped1[band_mask, :]
                + a[band_mask, :] * warped2[band_mask, :]
            )

    return pano


# [기능] 3장 이미지 스티칭 (Left-Center-Right)
def stitch_three_images(
    img_left: np.ndarray,
    img_center: np.ndarray,
    img_right: np.ndarray,
    H_left_to_center: np.ndarray,
    H_right_to_center: np.ndarray,
    margin: int = 0
) -> np.ndarray:
    imgs = [img_left, img_center, img_right]
    H_to_ref_list = [
        H_left_to_center,
        np.eye(3, dtype=np.float64),
        H_right_to_center
    ]
    ref_index = 1
    return stitch_multi_images(imgs, H_to_ref_list, ref_index=ref_index, margin=margin)


# [기능] 다중 이미지 스티칭 및 블렌딩
def stitch_multi_images(
    imgs: list[np.ndarray],
    H_to_ref_list: list[np.ndarray],
    ref_index: int = 0,
    margin: int = 0,
    global_weights: list[float] | None = None,
) -> np.ndarray:
    
    assert len(imgs) == len(H_to_ref_list), "imgs와 H_to_ref_list의 길이가 다릅니다."
    if global_weights is None:
        global_weights = [1.0] * len(imgs)

    canvas_shape, H_canvas_list = compute_panorama_canvas_multi(
        imgs,
        H_to_ref_list,
        ref_index=ref_index,
        margin=margin,
        global_weights=global_weights,
    )

    H_canvas, W_canvas = canvas_shape

    if imgs[0].ndim == 2:
        C = 1
        pano_accum = np.zeros((H_canvas, W_canvas), dtype=np.float32)
    else:
        C = imgs[0].shape[2]
        pano_accum = np.zeros((H_canvas, W_canvas, C), dtype=np.float32)

    weight = np.zeros((H_canvas, W_canvas), dtype=np.float32)

    for idx, (img, H_canvas_i) in enumerate(zip(imgs, H_canvas_list)):
        g = float(global_weights[idx])
        if g <= 0.0:
            continue

        corners = _get_image_corners(img)
        ones = np.ones((4, 1), dtype=np.float64)
        corners_h = np.hstack([corners.astype(np.float64), ones])
        projected_h = (H_canvas_i @ corners_h.T).T

        w = projected_h[:, 2]
        if not np.all(np.isfinite(w)) or np.any(w < 1e-4):
            continue
        
        projected = projected_h[:, :2] / projected_h[:, 2:3]
        if not np.all(np.isfinite(projected)):
            continue

        x_coords = projected[:, 0]
        y_coords = projected[:, 1]

        x_min = int(np.floor(x_coords.min()))
        x_max = int(np.ceil(x_coords.max())) + 1
        y_min = int(np.floor(y_coords.min()))
        y_max = int(np.ceil(y_coords.max())) + 1

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(W_canvas, x_max)
        y_max = min(H_canvas, y_max)

        dst_bounds = (y_min, y_max, x_min, x_max)

        print(f"[stitch] warping img{idx+1}/{len(imgs)} into canvas y[{y_min},{y_max}), x[{x_min},{x_max})")

        try:
            H_dst_to_img = np.linalg.inv(H_canvas_i)
        except np.linalg.LinAlgError:
            continue

        _accumulate_warped_image_into_panorama(
            pano_accum=pano_accum,
            weight=weight,
            img=img,
            H_dst_src=H_dst_to_img,
            dst_bounds=dst_bounds,
            global_gain=g,
        )

    valid = weight > 1e-6
    if C == 1:
        pano = np.zeros_like(pano_accum, dtype=np.float32)
        pano[valid] = pano_accum[valid] / weight[valid]
    else:
        pano = np.zeros_like(pano_accum, dtype=np.float32)
        pano[valid] = pano_accum[valid] / weight[valid, None]

    return pano