# [핵심 기능] 톤 매핑 및 밝기 보정 모듈
# - Exposure Compensation: 이미지 간 평균 밝기 차이를 완화하여 자연스러운 연결 유도
# - Gamma Correction: 최종 파노라마 결과물의 대비(Contrast) 조정

import numpy as np

# [기능] 이미지들의 평균 밝기를 계산하여 이득(Gain) 조절 (노출 보정)
def simple_exposure_compensation(imgs: list[np.ndarray]) -> list[np.ndarray]:
    means = []
    for img in imgs:
        means.append(np.mean(img))
    
    avg_mean = np.mean(means)
    adjusted_imgs = []
    
    for i, img in enumerate(imgs):
        # [보정] 전체 평균 밝기에 맞추기 위한 Gain 계산 (급격한 변화 방지 클리핑 적용)
        gain = avg_mean / (means[i] + 1e-6)
        gain = np.clip(gain, 0.7, 1.3)
        adjusted = img.astype(np.float32) * gain
        adjusted_imgs.append(np.clip(adjusted, 0, 255).astype(np.uint8))
        
    return adjusted_imgs

# [기능] 감마 보정을 통한 대비(Contrast) 향상 (Lookup Table 사용)
def apply_gamma_correction(img: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
    
    H, W, C = img.shape
    flat_img = img.flatten()
    corrected = table[flat_img]
    return corrected.reshape(H, W, C)