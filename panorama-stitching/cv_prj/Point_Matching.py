# [핵심 기능] 디스크립터 간 유사도 측정 및 매칭 (Feature Matching)
# - Cosine Similarity: 벡터 방향성 기반 유사도 측정 (조명 변화 강인)
# - Ratio Test & Cross Check: 매칭 품질 향상 및 오매칭 제거

from typing import List, Tuple
import numpy as np


# [기능] 두 디스크립터 집합 간 최적 매칭 쌍 찾기 (Cosine Distance + Ratio Test)
def match_descriptors(
    desc1: np.ndarray,
    desc2: np.ndarray,
    ratio_thresh: float = 0.8,
    max_distance: float | None = None,
    cross_check: bool = False
) -> List[Tuple[int, int]]:
    
    if desc1.ndim != 2 or desc2.ndim != 2:
        raise ValueError("match_descriptors: desc1, desc2는 (N, D) 모양이어야 합니다.")

    # [정규화] Cosine Similarity 계산을 위해 단위 벡터로 변환
    norm1 = np.linalg.norm(desc1, axis=1, keepdims=True) + 1e-6
    desc1 = desc1 / norm1
    
    norm2 = np.linalg.norm(desc2, axis=1, keepdims=True) + 1e-6
    desc2 = desc2 / norm2

    N1 = desc1.shape[0]
    N2 = desc2.shape[0]

    matches: List[Tuple[int, int]] = []
    
    # [유사도] 내적(Dot Product)을 통한 Cosine Similarity 행렬 계산
    sim_matrix = desc1 @ desc2.T  

    min_sim = 0.5  

    for i in range(N1):
        sims = sim_matrix[i]
        
        sorted_idx = np.argsort(sims)[::-1]
        
        best_idx = sorted_idx[0]
        second_idx = sorted_idx[1] if N2 > 1 else -1
        
        best_sim = sims[best_idx]
        second_sim = sims[second_idx] if second_idx != -1 else 0.0
        
        if best_sim < min_sim:
            continue
            
        # [검증] Lowe's Ratio Test (Cosine Distance 관점)
        dist1 = 1.0 - best_sim
        dist2 = 1.0 - second_sim
        
        if ratio_thresh is not None:
            if dist1 > ratio_thresh * dist2:
                continue
        
        matches.append((i, int(best_idx)))

    if not cross_check:
        return matches

    # [검증] Cross Check (양방향 매칭 일치 여부 확인)
    final_matches = []
    for (i, j) in matches:
        col_sims = sim_matrix[:, j]
        best_i_rev = np.argmax(col_sims)
        
        if best_i_rev == i:
            final_matches.append((i, j))
            
    return final_matches


# [기능] 매칭된 인덱스를 실제 좌표(x, y) 배열로 변환
def build_correspondence_points(
    keypoints1: np.ndarray,
    keypoints2: np.ndarray,
    matches: List[Tuple[int, int]],
    swap_xy: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    if keypoints1.ndim != 2 or keypoints2.ndim != 2:
        raise ValueError("build_correspondence_points: keypoints는 (N, 2) 모양의 배열이어야 합니다.")

    pts1_list = []
    pts2_list = []

    for (i, j) in matches:
        y1, x1 = keypoints1[i]
        y2, x2 = keypoints2[j]

        if swap_xy:
            pts1_list.append((float(x1), float(y1)))
            pts2_list.append((float(x2), float(y2)))
        else:
            pts1_list.append((float(y1), float(x1)))
            pts2_list.append((float(y2), float(x2)))

    if len(pts1_list) == 0:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0, 2), dtype=np.float32),
        )

    pts1 = np.array(pts1_list, dtype=np.float32)
    pts2 = np.array(pts2_list, dtype=np.float32)

    return pts1, pts2