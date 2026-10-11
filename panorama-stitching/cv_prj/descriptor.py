# [핵심 기능] 특징점(Keypoint) 주변의 픽셀 정보를 벡터화하여 기술자(Descriptor) 생성
# - Patch Descriptor: 특징점 주변 픽셀 패치를 추출하여 벡터로 변환
# - Normalization: 조명 변화에 강인하도록 Z-score 정규화 및 L2 정규화 수행

from typing import Tuple
import numpy as np


# [기능] Keypoint 주변 패치를 추출하여 정규화된 기술자(Descriptor)로 변환
def extract_patch_descriptors(
    gray: np.ndarray,
    keypoints: np.ndarray,
    patch_size: int = 9,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    
    if gray.ndim != 2:
        raise ValueError("extract_patch_descriptors: gray는 2D Grayscale 이미지여야 합니다.")
    if keypoints.ndim != 2 or keypoints.shape[1] != 2:
        raise ValueError("extract_patch_descriptors: keypoints는 (N, 2) 모양의 배열이어야 합니다.")

    H, W = gray.shape

    if patch_size % 2 == 0:
        raise ValueError("extract_patch_descriptors: patch_size는 홀수여야 합니다. (예: 7, 9, 11 ...)")

    half = patch_size // 2 

    descriptors_list = []
    valid_kpts_list = []

    for (y, x) in keypoints:
        y = int(y)
        x = int(x)

        # [경계 처리] 패치가 이미지 범위를 벗어나면 제외
        if y - half < 0 or y + half >= H:
            continue
        if x - half < 0 or x + half >= W:
            continue

        # [패치 추출] 특징점 중심의 patch_size x patch_size 영역 추출
        patch = gray[y - half : y + half + 1,
                     x - half : x + half + 1]

        # [벡터화] 2D 패치를 1D 벡터로 펼침
        vec = patch.flatten().astype(np.float32)

        # [정규화] Z-score Normalization (평균 0, 표준편차 1) -> 조명 변화 강인성 확보
        if normalize:
            mean = float(vec.mean())
            std = float(vec.std())
            eps = 1e-6
            vec = (vec - mean) / (std + eps)

        descriptors_list.append(vec)
        valid_kpts_list.append((y, x))

    if len(descriptors_list) == 0:
        return (
            np.zeros((0, 2), dtype=np.int32),
            np.zeros((0, patch_size * patch_size), dtype=np.float32),
        )

    valid_keypoints = np.array(valid_kpts_list, dtype=np.int32)
    descriptors = np.vstack(descriptors_list).astype(np.float32)

    return valid_keypoints, descriptors


# [기능] Descriptor 벡터를 단위 벡터로 변환 (Cosine Similarity 계산용)
def l2_normalize_descriptors(descriptors: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    if descriptors.ndim != 2:
        raise ValueError("l2_normalize_descriptors: descriptors는 (N, D) 모양의 2D 배열이어야 합니다.")

    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    norms = np.maximum(norms, eps)
    norm_desc = descriptors / norms
    return norm_desc.astype(np.float32)