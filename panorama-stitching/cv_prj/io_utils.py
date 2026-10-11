# [핵심 기능] 이미지 입출력 관리 (OpenCV 배제, PIL 사용)
# - 로딩: 단일/시퀀스 이미지 로드 및 RGB 변환
# - 저장: Numpy 배열의 타입/범위를 고려한 자동 스케일링 및 파일 저장

import os
from typing import List, Tuple

import numpy as np
from PIL import Image


# [기능] 파일 경로에서 이미지를 읽어 RGB Numpy 배열(uint8)로 반환
def load_image(path: str) -> np.ndarray:
    if not os.path.exists(path):
        raise FileNotFoundError(f"load_image: 파일을 찾을 수 없습니다: {path}")

    # [로딩] PIL로 열고 RGB로 변환 후 Numpy 배열화
    pil_img = Image.open(path)
    pil_img = pil_img.convert("RGB")
    img = np.array(pil_img, dtype=np.uint8)

    return img


# [기능] 패턴과 인덱스 리스트를 받아 여러 이미지를 순차적으로 로드
def load_image_sequence(pattern: str, indices: List[int]) -> Tuple[List[np.ndarray], List[str]]:
    images: List[np.ndarray] = []
    paths: List[str] = []

    for i in indices:
        path = pattern.format(i)
        img = load_image(path)

        images.append(img)
        paths.append(path)

    return images, paths


# [기능] Numpy 배열을 이미지 파일로 저장 (Data Type 및 Range 자동 처리)
def save_image(path: str, img: np.ndarray) -> None:
    dir_name = os.path.dirname(path)
    if dir_name != "" and not os.path.exists(dir_name):
        os.makedirs(dir_name, exist_ok=True)

    arr = img

    # [변환] Float 타입(0~1)인 경우 0~255로 스케일링
    if np.issubdtype(arr.dtype, np.floating):
        vmin = float(arr.min())
        vmax = float(arr.max())

        if vmax <= 1.0 + 1e-3:          
            arr = np.clip(arr, 0.0, 1.0)
            arr = (arr * 255.0).astype(np.uint8)
        else:                           
            arr = np.clip(arr, 0.0, 255.0).astype(np.uint8)

    else:
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    if np.issubdtype(arr.dtype, np.integer):
        arr = np.clip(arr, 0, 255).astype(np.uint8)

    # [저장] 차원에 따라 Grayscale/RGB 모드 설정 후 저장
    if arr.ndim == 2:
        pil_img = Image.fromarray(arr, mode="L")
    elif arr.ndim == 3 and arr.shape[2] == 3:
        pil_img = Image.fromarray(arr, mode="RGB")
    else:
        raise ValueError(
            f"save_image: 지원하지 않는 이미지 형식입니다. shape={arr.shape}"
        )

    pil_img.save(path)


# [기능] 디버깅 데이터(Response map 등)를 정규화하여 이미지로 저장
def save_debug_image(path: str, img: np.ndarray, normalize: bool = False) -> None:
    arr = img.astype(np.float32)

    # [정규화] Min-Max Scaling 적용 (옵션)
    if normalize:
        vmin = float(arr.min())
        vmax = float(arr.max())
        if vmax - vmin < 1e-8:
            arr = np.zeros_like(arr, dtype=np.float32)
        else:
            arr = (arr - vmin) / (vmax - vmin)

    save_image(path, arr)