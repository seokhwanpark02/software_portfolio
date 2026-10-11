# [핵심 기능] 이미지 전처리 모듈 (Normalization, Grayscale 변환, Gaussian Blur)
# - to_float32: 이미지를 0~1 범위의 float32로 변환
# - rgb_to_gray: RGB 이미지를 Grayscale로 변환
# - gaussian_blur_gray: 가우시안 블러링을 통한 노이즈 제거

import numpy as np


# [기능] uint8 이미지를 0~1 범위의 float32로 정규화
def to_float32(img):
    if np.issubdtype(img.dtype, np.floating):
        return img.astype(np.float32)

    return (img.astype(np.float32) / 255.0)


# [기능] RGB 이미지를 Grayscale로 변환 (가중치: Y = 0.299R + 0.587G + 0.114B)
def rgb_to_gray(img):
    if img.ndim != 3 or img.shape[2] != 3:
        raise ValueError("rgb_to_gray: 입력 이미지는 (H, W, 3) 형태의 RGB 여야 합니다.")

    gray = np.dot(img[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    return gray


# [기능] 2D Gaussian Kernel 생성 (가우시안 필터용)
def create_gaussian_kernel(ksize=5, sigma=1.0):
    if ksize % 2 == 0:
        raise ValueError("create_gaussian_kernel: ksize는 홀수여야 합니다.")

    k = ksize // 2
    x = np.arange(-k, k + 1, 1, dtype=np.float32)
    y = np.arange(-k, k + 1, 1, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma**2))
    kernel /= np.sum(kernel)
    return kernel.astype(np.float32)


# [기능] 2D Convolution 연산 구현 (Zero-padding 포함)
def convolve2d(image, kernel):
    if image.ndim != 2:
        raise ValueError("convolve2d: image는 2D Grayscale이어야 합니다.")

    ih, iw = image.shape
    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    kernel_flipped = np.flipud(np.fliplr(kernel))
    padded = np.pad(image, ((pad_h, pad_h), (pad_w, pad_w)), mode='constant', constant_values=0)
    output = np.zeros_like(image, dtype=np.float32)

    # [연산] 모든 픽셀에 대해 Convolution 수행
    for y in range(ih):
        for x in range(iw):
            region = padded[y:y+kh, x:x+kw]
            output[y, x] = np.sum(region * kernel_flipped)

    return output


# [기능] Grayscale 이미지에 Gaussian Blur 적용 (노이즈 제거)
def gaussian_blur_gray(gray, ksize=5, sigma=1.0):
    if gray.ndim != 2:
        raise ValueError("gaussian_blur_gray: 입력은 Grayscale(2D) 이어야 합니다.")

    kernel = create_gaussian_kernel(ksize, sigma)
    blurred = convolve2d(gray, kernel)
    return blurred


# [기능] 이미지 전처리 파이프라인 (타입변환 -> Grayscale -> Blur)
def preprocess_image(img, use_blur=True, blur_ksize=5, blur_sigma=1.0):
    img_f = to_float32(img)
    gray = rgb_to_gray(img_f)

    if use_blur:
        gray = gaussian_blur_gray(gray, ksize=blur_ksize, sigma=blur_sigma)

    return gray