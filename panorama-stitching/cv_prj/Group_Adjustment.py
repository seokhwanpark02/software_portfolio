# [핵심 기능] 파노라마 전체의 휘어짐(Drift) 보정
# - Global Slope Correction: 이미지 중심점들의 추세선을 구해 전체 기울기를 수평으로 보정
# - Vertical Center Alignment: 기준 이미지(Reference)의 Y축 높이에 맞춰 개별 이미지의 수직 위치 정렬

from __future__ import annotations
from typing import List, Optional
import numpy as np


# [기능] 전체 호모그래피 리스트를 받아 수직 정렬 및 기울기 보정을 수행
def adjust_vertical_centers(
    H_to_ref_list: List[np.ndarray],
    imgs: List[np.ndarray],
    ref_index: int = 0,
    img_weights: Optional[List[float]] = None,
) -> List[np.ndarray]:
    
    num_imgs = len(H_to_ref_list)
    assert len(imgs) == num_imgs, "imgs 와 H_to_ref_list 길이가 다릅니다."

    if img_weights is None:
        img_weights = [1.0] * num_imgs

    # [초기화] 원본 리스트 복사
    new_H_list: List[np.ndarray] = []
    for H in H_to_ref_list:
        if H is None:
            new_H_list.append(np.eye(3, dtype=np.float64))
        else:
            new_H_list.append(np.asarray(H, dtype=np.float64).copy())

    # [1단계] Global Slope Correction (기울기 보정)
    centers_x = []
    centers_y = []
    
    for i in range(num_imgs):
        if float(img_weights[i]) <= 0.0:
            continue
            
        H = new_H_list[i]
        h, w = imgs[i].shape[:2]
        
        # [투영] 각 이미지의 중심점을 Ref 좌표계로 변환
        cx, cy = w * 0.5, h * 0.5
        pt_h = H @ np.array([cx, cy, 1.0])
        
        if abs(pt_h[2]) > 1e-8:
            px = pt_h[0] / pt_h[2]
            py = pt_h[1] / pt_h[2]
            centers_x.append(px)
            centers_y.append(py)

    if len(centers_x) >= 2:
        # [추세선] 중심점들의 1차 함수 근사 (y = ax + b)
        slope, intercept = np.polyfit(centers_x, centers_y, 1)
        
        angle = -np.arctan(slope)
        
        # [회전] 과도한 회전 방지 (최대 10도 제한)
        ref_h, ref_w = imgs[ref_index].shape[:2]
        cx_ref, cy_ref = ref_w * 0.5, ref_h * 0.5
        
        max_angle = np.deg2rad(10)
        angle = np.clip(angle, -max_angle, max_angle)

        if abs(angle) > 1e-4:
            c_val = np.cos(angle)
            s_val = np.sin(angle)
            
            # [행렬] 중심 기준 회전 변환 행렬 생성 (T2 @ Rot @ T1)
            T1 = np.array([[1, 0, -cx_ref], [0, 1, -cy_ref], [0, 0, 1]])
            Rot = np.array([[c_val, -s_val, 0], [s_val, c_val, 0], [0, 0, 1]])
            T2 = np.array([[1, 0, cx_ref], [0, 1, cy_ref], [0, 0, 1]])
            
            R_global = T2 @ Rot @ T1
            
            for i in range(num_imgs):
                new_H_list[i] = R_global @ new_H_list[i]

    # [2단계] Vertical Center Alignment (수직 높이 정렬)
    H_ref = new_H_list[ref_index]
    h_ref, w_ref = imgs[ref_index].shape[:2]
    pt_ref = H_ref @ np.array([w_ref/2, h_ref/2, 1.0])
    target_y = pt_ref[1] / pt_ref[2]

    for i in range(num_imgs):
        if i == ref_index: 
            continue
        if float(img_weights[i]) <= 0.0:
            continue
            
        H_i = new_H_list[i]
        h_i, w_i = imgs[i].shape[:2]
        
        # [차이 계산] Ref 중심 Y와 현재 이미지 중심 Y의 차이(dy) 계산
        pt_i = H_i @ np.array([w_i/2, h_i/2, 1.0])
        curr_y = pt_i[1] / pt_i[2]
        
        dy = target_y - curr_y
        
        # [이동] 수직 이동 행렬 적용
        T_y = np.array([
            [1, 0, 0],
            [0, 1, dy],
            [0, 0, 1]
        ], dtype=np.float64)
        
        new_H_list[i] = T_y @ new_H_list[i]

    return new_H_list