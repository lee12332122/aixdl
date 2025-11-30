---
layout: post
title: "밀도 기반 군집화(DBSCAN) 완전 정리"
date: 2025-01-01 00:00:00 +0900
categories: [Machine Learning, Clustering]
tags: [DBSCAN, Density-Based, 군집화, 머신러닝]
toc: true
toc_sticky: true
---

**서울 기계공학부 김도윤**  
**서울 기계공학부 박준우**  
**서울 신소재공학부 이주연**  
**서울 기계공학부 이준범**

# 밀도 기반 군집화(DBSCAN) 완전 정리

PDF 문서를 바탕으로 **DBSCAN 알고리즘의 개념, 역사, 응용, 장단점, 실습 코드**를 모두 정리한 글입니다.

---

## 📌 Part 1. 도입 (Introduction)

### 왜 군집화가 필요한가?
데이터 속에서 ‘비슷한 것끼리 묶기’는 인공지능과 데이터 분석의 핵심 과제입니다.

예시:
- 고객을 구매 패턴별로 분류  
- 지도에서 위치 기반 이벤트 군집 찾기  
- 의료 데이터에서 질병 발생 지역 파악  

### K-means의 한계
K-means는 다음 문제점이 존재합니다:

1. 클러스터 개수 k를 미리 지정해야 함  
2. 구형(spherical) 클러스터만 감지  
3. 이상치(outlier)를 강제로 군집에 포함  
4. 복잡·비선형 구조 군집 탐지 불가  

이 한계를 해결하기 위해 등장한 것이 **밀도 기반 군집화(DBSCAN)**입니다.

---

## 📌 Part 2. 핵심 개념 (Core Concepts)

### DBSCAN이란?
DBSCAN(Density-Based Spatial Clustering of Applications with Noise)은  
데이터의 **밀도 차이**를 기반으로 클러스터를 찾는 알고리즘입니다.

### 핵심 용어
| 용어 | 의미 |
|------|------|
| **Epsilon (ε)** | 이웃으로 간주하는 거리 반경 |
| **MinPts** | Core Point가 되기 위한 최소 이웃 수 |
| **Core Point** | ε 내 이웃 ≥ MinPts |
| **Border Point** | Core에 연결되지만 자신은 Core가 아님 |
| **Noise Point** | 어느 군집에도 속하지 않는 점 |

### 알고리즘 작동 원리
1. 임의 포인트 P 선택  
2. ε 반경 내 이웃 개수 확인  
3. 이웃 ≥ MinPts → Core Point  
4. Core Point로부터 연결된 점 확장  
5. 조건 미달 시 Border 또는 Noise  
6. 전체 데이터 처리 시까지 반복  

---

## 📌 Part 3. 역사와 발전 (History & Evolution)
- 1996년 Ester, Kriegel, Sander, Xu가 DBSCAN 발표  
- 이후 OPTICS, HDBSCAN, DENCLUE로 발전  
- 고정밀 위치 데이터, 이상치 분리 필요성 증가로 활용 급증  

---

## 📌 Part 4. 실제 응용 사례 (Real-World Applications)

### 1. 도시 교통 데이터 분석
- GPS 데이터 → 혼잡 지역 자동 탐지  
- 신규 교통 노선 후보 발굴  

### 2. 금융 사기 탐지
- 이상치 거래 자동 분리  
- 정상 패턴과 다른 거래 탐지  

### 3. 반도체 결함 분석
- 웨이퍼 결함 위치 군집 → 공정 이상 조기 발견  

### 4. 감염병 확산 분석
- 환자 위치 기반 핫스팟 자동 탐색  

### 5. 부동산 가격 분석
- 유사 가격대 지역 분류  
- 이상거래 탐지  

---

## 📌 Part 5. 실습 가이드 (Hands-on Tutorial)

### Python DBSCAN 구현 예시

```python
from sklearn.cluster import DBSCAN
import numpy as np

X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
dbscan = DBSCAN(eps=3, min_samples=2)
labels = dbscan.fit_predict(X)

print(labels)
