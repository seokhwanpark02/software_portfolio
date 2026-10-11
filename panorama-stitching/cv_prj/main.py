# [핵심 기능] N장 이미지 파노라마 생성 메인 스크립트
# 흐름: 전처리 -> 특징점 추출 -> 매칭 -> 호모그래피 추정(RANSAC) -> 보정 -> 스티칭
import os
from pathlib import Path
import math
import numpy as np

from io_utils import load_image, save_image, save_debug_image
from preprocess import preprocess_image
from corner import harris_corners, compute_harris_response
from descriptor import extract_patch_descriptors, l2_normalize_descriptors
from Point_Matching import match_descriptors, build_correspondence_points
from homography import compute_homography, compute_reprojection_errors
from Stitching import (
    stitch_two_images,
    stitch_three_images,
    stitch_multi_images,   
)
from RANSAC import ransac_homography
from Group_Adjustment import adjust_vertical_centers
from Tone_Mapping import simple_exposure_compensation
from Tone_Mapping import apply_gamma_correction
from Stitching import _get_image_corners
from preprocess import gaussian_blur_gray

# [검증] 호모그래피 행렬의 기하학적 유효성 검사 (Scale, Perspective, Area)
def validate_homography_matrix(H, src_shape, dst_shape, area_ratio_max=10.0, perspective_thresh=0.002):
    if H is None: return False
    H = np.asarray(H, dtype=np.float64)
    if H.shape != (3, 3): return False
    if not np.isfinite(H).all(): return False

    if perspective_thresh is not None:
        p_mag = np.sqrt(H[2, 0]**2 + H[2, 1]**2)
        if p_mag > perspective_thresh:
            return False

    det = np.linalg.det(H[:2, :2])
    if det < 0.05 or det > 20.0:
        return False

    from Stitching import _get_image_corners
    
    dummy_src = np.zeros(src_shape, dtype=np.uint8)
    corners_src = _get_image_corners(dummy_src)

    ones = np.ones((4, 1), dtype=np.float64)
    corners_src_h = np.hstack([corners_src, ones])
    proj_h = (H @ corners_src_h.T).T
    
    denom = proj_h[:, 2:3]
    if np.any(np.abs(denom) < 1e-8): return False
    
    if not (np.all(denom > 0) or np.all(denom < 0)):
        return False

    proj = proj_h[:, :2] / denom
    if not np.isfinite(proj).all(): return False

    def quad_area(pts):
        x = pts[:, 0]; y = pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    area_src = quad_area(corners_src)
    area_proj = quad_area(proj)

    if area_src <= 1e-6 or area_proj <= 1e-6: return False

    ratio = area_proj / area_src
    
    if ratio > area_ratio_max or ratio < (1.0 / area_ratio_max):
        return False

    return True

# [검증] 매칭 방향성(좌/우) 일치 여부 확인
def check_homography_direction(H, sign_type: str, min_shift: float = 20.0) -> bool:
    if sign_type is None:
        return True
    
    tx = H[0, 2]
    
    if sign_type == 'positive':
        if tx < -min_shift:
            return False
    elif sign_type == 'negative':
        if tx > min_shift:
            return False
            
    return True

# [필터] RANSAC 전, 모션 벡터(dx, dy)의 일관성을 기반으로 Outlier 1차 제거
def prefilter_correspondences_by_motion(
    pts_i: np.ndarray,
    pts_j: np.ndarray,
    img_i_shape,
    img_j_shape,
    expected_tx_sign: str | None,
    k_mad: float = 3.5,
    min_keep: int = 12,
) -> tuple[np.ndarray, np.ndarray]:
    if pts_i is None or pts_j is None:
        return pts_i, pts_j
    if len(pts_i) < 4:
        return pts_i, pts_j

    pts_i = np.asarray(pts_i, dtype=np.float64)
    pts_j = np.asarray(pts_j, dtype=np.float64)
    if pts_i.shape != pts_j.shape or pts_i.shape[1] != 2:
        return pts_i, pts_j

    dx = pts_j[:, 0] - pts_i[:, 0]
    dy = pts_j[:, 1] - pts_i[:, 1]

    med_dx = float(np.median(dx))
    med_dy = float(np.median(dy))

    if expected_tx_sign == "positive" and med_dx < 0:
        pass
    elif expected_tx_sign == "negative" and med_dx > 0:
        pass

    mad_dx = float(np.median(np.abs(dx - med_dx))) + 1e-9
    mad_dy = float(np.median(np.abs(dy - med_dy))) + 1e-9

    H_i, W_i = img_i_shape[:2]
    H_j, W_j = img_j_shape[:2]
    min_dx = 0.01 * max(W_i, W_j)
    min_dy = 0.01 * max(H_i, H_j)

    thr_dx = max(k_mad * mad_dx, min_dx)
    thr_dy = max(k_mad * mad_dy, min_dy)

    keep = (np.abs(dx - med_dx) <= thr_dx) & (np.abs(dy - med_dy) <= thr_dy)

    if int(keep.sum()) < min_keep:
        return pts_i, pts_j

    return pts_i[keep], pts_j[keep]

# [보정] 호모그래피 정규화
def regularize_homography(H: np.ndarray, weight: float = 0.0) -> np.ndarray:
    H_reg = H.copy()
    H_reg[2, 0] *= (1.0 - weight)
    H_reg[2, 1] *= (1.0 - weight)
    if abs(H_reg[2, 2]) > 1e-8:
        H_reg /= H_reg[2, 2]
    return H_reg

# [전처리] 로그 변환을 통한 대비 강조
def enhance_contrast_log(img_gray: np.ndarray) -> np.ndarray:
    img_log = np.log1p(img_gray)
    
    min_val = np.min(img_log)
    max_val = np.max(img_log)
    if max_val - min_val > 1e-6:
        img_log = (img_log - min_val) / (max_val - min_val)
    
    return img_log

# [검증] 이미지 중심점 순서 역전 여부 확인
def check_center_ordering(
    H_curr_to_ref: np.ndarray,
    H_prev_to_ref: np.ndarray,
    img_curr_shape,
    img_prev_shape,
    direction: str
) -> bool:
    if H_curr_to_ref is None or H_prev_to_ref is None:
        return False

    h_c, w_c = img_curr_shape[:2]
    center_curr = np.array([w_c / 2.0, h_c / 2.0, 1.0])
    
    h_p, w_p = img_prev_shape[:2]
    center_prev = np.array([w_p / 2.0, h_p / 2.0, 1.0])

    proj_curr = H_curr_to_ref @ center_curr
    proj_prev = H_prev_to_ref @ center_prev

    if abs(proj_curr[2]) < 1e-8 or abs(proj_prev[2]) < 1e-8:
        return False

    cx_curr = proj_curr[0] / proj_curr[2]
    cx_prev = proj_prev[0] / proj_prev[2]

    margin = 10.0
    
    if direction == 'left':
        return (cx_curr < cx_prev - margin)
    elif direction == 'right':
        return (cx_curr > cx_prev + margin)
    
    return False

# [기능] 2장 이미지 파노라마 생성 (테스트용)
def build_panorama_two_images(
    testimg1_path: str,
    testimg2_path: str,
    output_path: str,
):
    print(f"[1] Loading images: {testimg1_path}, {testimg2_path}")
    testimg1 = load_image(testimg1_path)
    testimg2 = load_image(testimg2_path)

    print("[2] Preprocessing (grayscale + gaussian blur)")
    gray1 = preprocess_image(testimg1, use_blur=True, blur_ksize=5, blur_sigma=1.0)
    gray2 = preprocess_image(testimg2, use_blur=True, blur_ksize=5, blur_sigma=1.0)

    print("[3] Detecting Harris corners")
    k = 0.04
    window_size = 5
    sigma = 1.0
    nms_radius = 2
    threshold_ratio = 0.01
    max_points = 1000
    edge_margin = 10

    kpts1 = harris_corners(
        gray1,
        k=k,
        window_size=window_size,
        sigma=sigma,
        nms_radius=nms_radius,
        threshold_ratio=threshold_ratio,
        max_points=max_points,
        edge_margin=edge_margin,
    )
    kpts2 = harris_corners(
        gray2,
        k=k,
        window_size=window_size,
        sigma=sigma,
        nms_radius=nms_radius,
        threshold_ratio=threshold_ratio,
        max_points=max_points,
        edge_margin=edge_margin,
    )

    print(f"  - corners testimg1: {len(kpts1)}")
    print(f"  - corners testimg2: {len(kpts2)}")

    print("[4] Extracting patch descriptors")
    patch_size = 9

    kpts1_valid, desc1 = extract_patch_descriptors(
        gray1, kpts1, patch_size=patch_size, normalize=True
    )
    kpts2_valid, desc2 = extract_patch_descriptors(
        gray2, kpts2, patch_size=patch_size, normalize=True
    )

    print(f"  - valid keypoints testimg1 (after border check): {len(kpts1_valid)}")
    print(f"  - valid keypoints testimg2 (after border check): {len(kpts2_valid)}")

    desc1 = l2_normalize_descriptors(desc1)
    desc2 = l2_normalize_descriptors(desc2)

    print("[5] Matching descriptors (NN + ratio test + cross-check)")
    matches = match_descriptors(
        desc1,
        desc2,
        ratio_thresh=0.9,
        max_distance=None,
        cross_check=True
    )

    print(f"  - number of raw matches: {len(matches)}")

    if len(matches) < 4:
        print("❌ 매칭된 점이 4개 미만이라 Homography를 계산할 수 없습니다.")
        return

    print("[6] Building correspondence point sets")
    pts1, pts2 = build_correspondence_points(
        kpts1_valid, kpts2_valid, matches, swap_xy=True
    )

    print("[7] Estimating Homography with RANSAC (testimg1 -> testimg2)")
    H_1_to_2, inliers = ransac_homography(
        pts1, pts2,
        n_iter=2000,
        thresh=3.0,
        min_inliers=8,
        random_state=42
    )
    print("  - Homography matrix H_1_to_2 (RANSAC refined):")
    print(H_1_to_2)
    print(f"  - inliers: {inliers.sum()} / {len(inliers)}")

    errors = compute_reprojection_errors(pts1[inliers], pts2[inliers], H_1_to_2)
    print(f"  - mean reprojection error (inliers): {errors.mean():.4f} pixels")
    print(f"  - median reprojection error (inliers): {np.median(errors):.4f} pixels")

    print("[8] Stitching two images into panorama")
    pano = stitch_two_images(
        testimg1,
        testimg2,
        H_1_to_2,
        margin=50,
    )

    print(f"[9] Saving panorama to: {output_path}")
    save_image(output_path, pano)
    print("✅ Done!")

# [검증] 누적 호모그래피의 왜곡 정도 확인
def is_chain_homography_reasonable_to_ref(
    H_i_to_ref: np.ndarray,
    img_i_shape,
    ref_shape,
    area_ratio_max: float = 5.0,
    center_shift_factor: float = 5.0,
) -> bool:
    if H_i_to_ref is None:
        return False

    H_i_to_ref = np.asarray(H_i_to_ref, dtype=np.float64)
    if H_i_to_ref.shape != (3, 3):
        return False
    if not np.isfinite(H_i_to_ref).all():
        return False

    dummy_i = np.zeros(img_i_shape, dtype=np.uint8)
    corners_i = _get_image_corners(dummy_i)

    ones = np.ones((4, 1), dtype=np.float64)
    corners_i_h = np.hstack([corners_i, ones])

    proj_h = (H_i_to_ref @ corners_i_h.T).T
    denom = proj_h[:, 2:3]
    if np.any(np.abs(denom) < 1e-9):
        return False

    proj = proj_h[:, :2] / denom
    if not np.isfinite(proj).all():
        return False

    def quad_area(pts):
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))

    area_src = quad_area(corners_i)
    area_proj = quad_area(proj)

    if area_src <= 1e-6 or area_proj <= 1e-6:
        return False

    area_ratio = area_proj / area_src
    if area_ratio <= 0:
        return False

    if area_ratio > area_ratio_max or area_ratio < 0.2:
        return False

    return True

# [평가] 매칭 신뢰도 점수 계산 (Inlier 개수 및 비율 기반)
def homography_confidence(res: dict, min_inliers_floor: int) -> float:
    if res is None:
        return 0.0

    inliers = int(res.get("inliers", 0))
    matches = int(res.get("matches", inliers))
    median_err = float(res.get("median_err", np.inf))

    if matches <= 0:
        return 0.0

    inlier_ratio = inliers / float(matches)

    denom = float(max(1, int(min_inliers_floor)) * 3)
    inlier_score = min(1.0, inliers / denom)

    ratio_score = min(1.0, inlier_ratio / 0.5)

    if not np.isfinite(median_err):
        err_score = 0.0
    else:
        err_score = math.exp(-median_err / 3.0)

    conf = 0.5 * inlier_score + 0.3 * ratio_score + 0.2 * err_score

    conf = float(np.clip(conf, 0.05, 1.0))
    return conf

# [핵심] 두 이미지 간 최적 호모그래피 추정 (RANSAC + 검증 + 적응형 Threshold)
def estimate_pair_homography(
    kpts_i: np.ndarray,
    desc_i: np.ndarray,
    kpts_j: np.ndarray,
    desc_j: np.ndarray,
    img_i_shape,
    img_j_shape,
    ransac_iter: int,
    ransac_thresh: float,
    min_inliers_floor: int,
    ratio_trials=(0.7, 0.8, 0.9),
    cross_check_trials=(True, True, False),
    area_ratio_max: float = 4.0,
    tag: str = "",
    ransac_thresh_trials: tuple[float, ...] | None = (3.0, 6.0, 10.0),
    expected_tx_sign: str = None,
):
    if desc_i is None or desc_j is None:
        return None
    if len(desc_i) < 4 or len(desc_j) < 4:
        return None

    best = None

    if ransac_thresh_trials is None:
        ransac_thresh_trials = (ransac_thresh,)

    ABSOLUTE_MIN_INLIERS = 8
    MIN_INLIER_RATIO = 0.12

    for ratio, cc in zip(ratio_trials, cross_check_trials):
        matches = match_descriptors(desc_i, desc_j, ratio_thresh=ratio, cross_check=cc)
        if len(matches) < ABSOLUTE_MIN_INLIERS:
            continue

        pts_i, pts_j = build_correspondence_points(
            kpts_i, kpts_j, matches, swap_xy=True
        )

        pts_i_f, pts_j_f = prefilter_correspondences_by_motion(
            pts_i,
            pts_j,
            img_i_shape=img_i_shape,
            img_j_shape=img_j_shape,
            expected_tx_sign=expected_tx_sign,
            k_mad=3.5,
            min_keep=max(ABSOLUTE_MIN_INLIERS, 12),
        )

        if pts_i_f is None or pts_j_f is None or len(pts_i_f) < ABSOLUTE_MIN_INLIERS:
            continue

        for thr in ransac_thresh_trials:
            H, inliers_mask = ransac_homography(
                pts_i_f,
                pts_j_f,
                n_iter=ransac_iter,
                thresh=float(thr),
                min_inliers=ABSOLUTE_MIN_INLIERS,
            )

            if H is None:
                continue

            inlier_count = int(inliers_mask.sum())
            if inlier_count < ABSOLUTE_MIN_INLIERS:
                continue

            inlier_ratio = inlier_count / float(len(pts_i_f))
            if inlier_ratio < MIN_INLIER_RATIO:
                continue

            if not check_homography_direction(H, expected_tx_sign):
                continue

            if not validate_homography_matrix(
                H,
                img_i_shape,
                img_j_shape,
                area_ratio_max=area_ratio_max,
                perspective_thresh=0.002,
            ):
                continue

            errors = compute_reprojection_errors(
                pts_i_f[inliers_mask],
                pts_j_f[inliers_mask],
                H,
            )
            median_err = float(np.median(errors)) if len(errors) > 0 else np.inf

            if median_err > float(thr) * 2.0:
                continue

            score = (inlier_count * inlier_count) / (median_err + 1e-6)

            if best is None or score > best["score"]:
                best = {
                    "H": H,
                    "inliers": inlier_count,
                    "inlier_ratio": float(inlier_ratio),
                    "median_err": median_err,
                    "matches": len(matches),
                    "ratio": ratio,
                    "cross_check": cc,
                    "thresh": float(thr),
                    "score": score,
                    "tag": tag,
                }

    return best

# [기능] 이미지 전처리 및 특징점(Corner, Descriptor) 추출
def extract_features_for_image(img: np.ndarray, label: str = "image"):
    print(f"[2-{label}] Preprocessing (grayscale + gaussian blur)")
    gray = preprocess_image(img, use_blur=True, blur_ksize=5, blur_sigma=1.0)

    print(f"[3-{label}] Detecting Harris corners")

    k = 0.04
    window_size = 5
    sigma = 1.0
    edge_margin = 15

    target_kpts = 600
    min_kpts = 250

    threshold_trials = [0.01, 0.007, 0.005, 0.003, 0.002, 0.001]
    nms_trials =       [3,    3,     3,     2,     2,     2]
    max_points = 1500

    best_kpts = None
    best_params = None

    for thr, nms_r in zip(threshold_trials, nms_trials):
        kpts_try = harris_corners(
            gray,
            k=k,
            window_size=window_size,
            sigma=sigma,
            nms_radius=nms_r,
            threshold_ratio=thr,
            max_points=max_points,
            edge_margin=edge_margin,
        )

        if best_kpts is None or len(kpts_try) > len(best_kpts):
            best_kpts = kpts_try
            best_params = (thr, nms_r)

        if len(kpts_try) >= min_kpts:
            best_kpts = kpts_try
            best_params = (thr, nms_r)
            break

    thr_used, nms_used = best_params
    kpts = best_kpts

    print(f"  - corners {label}: {len(kpts)} (thr={thr_used}, nms_radius={nms_used})")

    print(f"[4-{label}] Extracting patch descriptors")
    patch_size = 11
    kpts_valid, desc = extract_patch_descriptors(
        gray, kpts, patch_size=patch_size, normalize=True
    )
    desc = l2_normalize_descriptors(desc)
    return gray, kpts_valid, desc


# [기능] 3장 이미지 파노라마 생성 (테스트용)
def build_panorama_three_images(
    img1_path: str,
    img2_path: str,
    img3_path: str,
    output_path: str,
):
    print(f"[1] Loading images: {img1_path}, {img2_path}, {img3_path}")
    img1 = load_image(img1_path)
    img2 = load_image(img2_path)
    img3 = load_image(img3_path)

    gray1, kpts1, desc1 = extract_features_for_image(img1, label="img1")
    gray2, kpts2, desc2 = extract_features_for_image(img2, label="img2")
    gray3, kpts3, desc3 = extract_features_for_image(img3, label="img3")

    print("[5-1] Matching descriptors between img1 and img2")
    matches12 = match_descriptors(
        desc1, desc2,
        ratio_thresh=0.9,
        max_distance=None,
        cross_check=True
    )
    print(f"  - matches 1-2: {len(matches12)}")

    if len(matches12) < 4:
        print("❌ img1-img2 매칭이 너무 적어서 Homography를 계산할 수 없습니다.")
        return

    pts1_12, pts2_12 = build_correspondence_points(
        kpts1, kpts2, matches12, swap_xy=True
    )

    print("[6-1] Estimating H_1_to_2 with RANSAC")
    H_1_to_2, inliers12 = ransac_homography(
        pts1_12, pts2_12,
        n_iter=2000,
        thresh=3.0,
        min_inliers=30,
        random_state=42,
    )
    print("  - H_1_to_2:")
    print(H_1_to_2)
    print(f"  - inliers 1-2: {inliers12.sum()} / {len(inliers12)}")

    err12 = compute_reprojection_errors(
        pts1_12[inliers12], pts2_12[inliers12], H_1_to_2
    )
    print(f"  - mean reprojection error 1-2 (inliers): {err12.mean():.4f}")

    print("[5-2] Matching descriptors between img3 and img2")
    matches32 = match_descriptors(
        desc3, desc2,
        ratio_thresh=0.75,
        max_distance=None,
        cross_check=True
    )
    print(f"  - matches 3-2: {len(matches32)}")

    if len(matches32) < 4:
        print("❌ img3-img2 매칭이 너무 적어서 Homography를 계산할 수 없습니다.")
        return

    pts3_32, pts2_32 = build_correspondence_points(
        kpts3, kpts2, matches32, swap_xy=True
    )

    print("[6-2] Estimating H_3_to_2 with RANSAC")
    H_3_to_2, inliers32 = ransac_homography(
        pts3_32, pts2_32,
        n_iter=2000,
        thresh=3.0,
        min_inliers=8,
        random_state=24,
    )
    print("  - H_3_to_2:")
    print(H_3_to_2)
    print(f"  - inliers 3-2: {inliers32.sum()} / {len(inliers32)}")

    err32 = compute_reprojection_errors(
        pts3_32[inliers32], pts2_32[inliers32], H_3_to_2
    )
    print(f"  - mean reprojection error 3-2 (inliers): {err32.mean():.4f}")

    print("[7] Stitching three images into panorama (center = img2)")
    pano = stitch_three_images(
        img1, img2, img3,
        H_left_to_center=H_1_to_2,
        H_right_to_center=H_3_to_2,
        margin=50,
    )

    print(f"[8] Saving panorama to: {output_path}")
    save_image(output_path, pano)
    print("✅ Done (3 images)!")


# [메인] N장 이미지 파노라마 생성 파이프라인
def build_panorama_n_images(
    image_paths,
    ref_index=None,
    ratio_thresh=0.7,
    ransac_iter=2000,
    ransac_thresh=4.0,
    ransac_min_inliers=8,
    margin=100,
    output_path="results/result_n.jpg",
):
    # 1. 이미지 로드 및 노출 보정
    imgs = [load_image(p) for p in image_paths]

    print("[Tone Mapping] Adjusting exposure across images...")
    imgs = simple_exposure_compensation(imgs)

    if ref_index is None:
        ref_index = len(imgs) // 2

    num_imgs = len(imgs)
    img_conf_weights = [1.0 for _ in range(num_imgs)]

    # 2. 특징점 및 디스크립터 추출
    keypoints_list = []
    descriptors_list = []

    for i, img in enumerate(imgs):
        gray, kpts, desc = extract_features_for_image(img, label=f"img{i+1}")
        keypoints_list.append(kpts)
        descriptors_list.append(desc)

    H_to_ref_list = [None] * num_imgs
    H_to_ref_list[ref_index] = np.eye(3, dtype=np.float64)

    ref_kpts = keypoints_list[ref_index]
    ref_desc = descriptors_list[ref_index]

    # 3-a. 기준 이미지(Ref)와 직접 매칭 시도
    for i in range(num_imgs):
        if i == ref_index:
            continue
        if H_to_ref_list[i] is not None:
            continue

        print(f"[Direct] Matching img{i+1} -> img{ref_index+1}")
        
        expected_sign = 'negative' if i < ref_index else 'positive'

        res = estimate_pair_homography(
            keypoints_list[i], descriptors_list[i],
            ref_kpts, ref_desc,
            imgs[i].shape[:2], imgs[ref_index].shape[:2],
            ransac_iter=ransac_iter,
            ransac_thresh=ransac_thresh,
            min_inliers_floor=ransac_min_inliers,
            ratio_trials=(0.7, 0.8, 0.9),
            cross_check_trials=(True, True, False),
            area_ratio_max=50.0,
            tag=f"direct {i+1}->{ref_index+1}",
            ransac_thresh_trials=(ransac_thresh, 10.0, 30.0),
            expected_tx_sign=expected_sign
        )

        if res is None:
            print(f"  ⚠ direct img{i+1}-img{ref_index+1} 실패. 체인 방식으로 시도합니다.")
            continue
        
        H_direct = res["H"]

        if not validate_homography_matrix(
            H_direct, imgs[i].shape[:2], imgs[ref_index].shape[:2],
            area_ratio_max=500.0,
            perspective_thresh=None
        ):
            print(f"  ⚠ direct Homography 비정상(왜곡 과다). 체인으로 넘김.")
            continue

        direction = 'left' if i < ref_index else 'right'
        if not check_center_ordering(H_direct, np.eye(3), imgs[i].shape, imgs[ref_index].shape, direction):
            print(f"  ⚠ direct img{i+1} 위치 역전 감지. 체인으로 넘김.")
            continue
        
        print(f"  ✅ direct img{i+1} -> img{ref_index+1} Homography 사용")
        H_to_ref_list[i] = H_direct
        img_conf_weights[i] = homography_confidence(res, ransac_min_inliers)

    # 3-b. 좌측 체인 매칭 (Robust Chaining)
    for i in range(ref_index - 1, -1, -1):
        if H_to_ref_list[i] is not None:
            continue

        candidate_js = []
        for step in (1, 2, 3, 4, 5): 
            j = i + step
            if 0 <= j < num_imgs and j <= ref_index:
                candidate_js.append(j)

        best_picked = None
        best_picked_res = None
        best_score = -1e18

        for j in candidate_js:
            if H_to_ref_list[j] is None:
                continue

            print(f"[N-left] Matching img{i+1} -> img{j+1}")

            res = estimate_pair_homography(
                keypoints_list[i], descriptors_list[i],
                keypoints_list[j], descriptors_list[j],
                imgs[i].shape[:2], imgs[j].shape[:2],
                ransac_iter=ransac_iter,
                ransac_thresh=ransac_thresh,
                min_inliers_floor=ransac_min_inliers,
                area_ratio_max=20.0,
                tag=f"chain L {i+1}->{j+1}",
                expected_tx_sign='negative',
                ransac_thresh_trials=(ransac_thresh, 10.0, 30.0),
            )

            if res is None:
                res_rev = estimate_pair_homography(
                    keypoints_list[j], descriptors_list[j],
                    keypoints_list[i], descriptors_list[i],
                    imgs[j].shape[:2], imgs[i].shape[:2],
                    ransac_iter=ransac_iter,
                    ransac_thresh=ransac_thresh,
                    min_inliers_floor=ransac_min_inliers,
                    area_ratio_max=20.0,
                    tag=f"chain L (rev) {j+1}->{i+1}",
                    expected_tx_sign='positive',
                    ransac_thresh_trials=(ransac_thresh, 10.0, 30.0),
                )
                if res_rev is None:
                    continue
                try:
                    H_i_to_j = np.linalg.inv(res_rev["H"])
                except np.linalg.LinAlgError:
                    continue
                res = res_rev
            else:
                H_i_to_j = res["H"]

            H_candidate = H_to_ref_list[j] @ H_i_to_j

            if not validate_homography_matrix(
                H_candidate, imgs[i].shape[:2], imgs[ref_index].shape[:2],
                area_ratio_max=2000.0,
                perspective_thresh=None
            ):
                print(f"  ⚠ chain L img{i+1} 누적 Homography 왜곡 과다.")
                continue

            if not check_center_ordering(H_candidate, H_to_ref_list[j], imgs[i].shape, imgs[j].shape, 'left'):
                print(f"  ⚠ chain L img{i+1} 위치 역전. Skip.")
                continue

            score = float(res.get("score", 0.0))
            if score > best_score:
                best_score = score
                best_picked = (j, H_candidate)
                best_picked_res = res

        if best_picked is None:
            j = i + 1
            print(f"⚠ img{i+1} 정렬 실패. img{j+1} 위치로 fallback (weight=0.10).")
            H_fallback = H_to_ref_list[j].copy()
            width_shift = -imgs[i].shape[1] * 0.9
            T_shift = np.array([[1.0, 0.0, width_shift], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            H_to_ref_list[i] = T_shift @ H_fallback 
            img_conf_weights[i] = 0.15  
            continue

        j, H_candidate = best_picked
        H_to_ref_list[i] = H_candidate
        img_conf_weights[i] = homography_confidence(best_picked_res, ransac_min_inliers) 

    # 3-c. 우측 체인 매칭 (Robust Chaining)
    for i in range(ref_index + 1, num_imgs):
        if H_to_ref_list[i] is not None:
            continue

        candidate_js = []
        for step in (1, 2, 3, 4, 5):
            j = i - step
            if 0 <= j < num_imgs and j >= ref_index:
                candidate_js.append(j)

        best_picked = None
        best_picked_res = None
        best_score = -1e18

        for j in candidate_js:
            if H_to_ref_list[j] is None:
                continue

            print(f"[N-right] Matching img{i+1} -> img{j+1}")

            res = estimate_pair_homography(
                keypoints_list[i], descriptors_list[i],
                keypoints_list[j], descriptors_list[j],
                imgs[i].shape[:2], imgs[j].shape[:2],
                ransac_iter=ransac_iter,
                ransac_thresh=ransac_thresh,
                min_inliers_floor=ransac_min_inliers,
                ratio_trials=(0.7, 0.85, 0.95),
                cross_check_trials=(True, True, False),
                area_ratio_max=20.0,
                tag=f"chain R {i+1}->{j+1}",
                expected_tx_sign='positive',
                ransac_thresh_trials=(ransac_thresh, 10.0, 30.0),
            )

            if res is None:
                res_rev = estimate_pair_homography(
                    keypoints_list[j], descriptors_list[j],
                    keypoints_list[i], descriptors_list[i],
                    imgs[j].shape[:2], imgs[i].shape[:2],
                    ransac_iter=ransac_iter,
                    ransac_thresh=ransac_thresh,
                    min_inliers_floor=ransac_min_inliers,
                    ratio_trials=(0.7, 0.85, 0.95),
                    cross_check_trials=(True, True, False),
                    area_ratio_max=20.0,
                    tag=f"chain R (rev) {j+1}->{i+1}",
                    expected_tx_sign='negative',
                    ransac_thresh_trials=(ransac_thresh, 10.0, 30.0),
                )
                if res_rev is None:
                    continue
                try:
                    H_i_to_j = np.linalg.inv(res_rev["H"])
                except np.linalg.LinAlgError:
                    continue
                res = res_rev
            else:
                H_i_to_j = res["H"]

            H_candidate = H_to_ref_list[j] @ H_i_to_j

            if not validate_homography_matrix(
                H_candidate, imgs[i].shape[:2], imgs[ref_index].shape[:2],
                area_ratio_max=2000.0,
                perspective_thresh=None
            ):
                print(f"  ⚠ chain R img{i+1} 누적 Homography 왜곡 과다.")
                continue

            if not check_center_ordering(H_candidate, H_to_ref_list[j], imgs[i].shape, imgs[j].shape, 'right'):
                print(f"  ⚠ chain R img{i+1} 위치 역전. Skip.")
                continue

            score = float(res.get("score", 0.0))
            if score > best_score:
                best_score = score
                best_picked = (j, H_candidate)
                best_picked_res = res

        if best_picked is None:
            j = i - 1
            print(f"⚠ img{i+1} 정렬 실패. img{j+1} 위치로 fallback (weight=0.10).")
            H_fallback = H_to_ref_list[j].copy()
            width_shift = imgs[i].shape[1] * 0.9
            T_shift = np.array([[1.0, 0.0, width_shift], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
            H_to_ref_list[i] = T_shift @ H_fallback
            img_conf_weights[i] = 0.15  
            continue

        j, H_candidate = best_picked
        H_to_ref_list[i] = H_candidate
        img_conf_weights[i] = homography_confidence(best_picked_res, ransac_min_inliers)

    # 4. 수직 중심 보정 및 최종 스티칭
    for i, H in enumerate(H_to_ref_list):
        if H is None:
            print(f"⚠ H_to_ref_list[{i}]가 None입니다. 단위행렬로 대체.")
            H_to_ref_list[i] = np.eye(3, dtype=np.float64)
            img_conf_weights[i] = 0.0

    H_to_ref_list = adjust_vertical_centers(
        H_to_ref_list, imgs, ref_index=ref_index, img_weights=img_conf_weights,
    )

    pano_float = stitch_multi_images(
        imgs, H_to_ref_list, ref_index=ref_index, margin=margin, global_weights=img_conf_weights,
    )

    if pano_float.dtype != np.uint8:
        pano = np.clip(pano_float, 0.0, 255.0).astype(np.uint8)
    else:
        pano = pano_float

    pano = apply_gamma_correction(pano, gamma=1.1) 
    save_image(output_path, pano)
    
    return pano


if __name__ == "__main__":
    num_images = 10
    image_pattern = "cv_prj/img_data/testimg{}.jpg"

    image_paths = [image_pattern.format(i) for i in range(1, num_images + 1)]
    ref_index = num_images // 2

    build_panorama_n_images(
        image_paths,
        ref_index=ref_index,
        output_path="cv_prj/results/result.jpg",
    )