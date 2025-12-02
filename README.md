

# 밀도 기반 군집화(DBSCAN)의 이해와 사용 방안의 고찰

**서울 기계공학부 김도윤**  
**서울 기계공학부 박준우**  
**서울 신소재공학부 이주연**  
**서울 기계공학부 이준범**

---
<div style="margin-top: 10000px; margin-bottom: 60px; font-size: 60px; line-height: 1.8;">

## 목차  
1. **Intro: 왜 '밀도'인가? (K-Means의 한계와 비교)**
   
2. **Core Theory: DBSCAN란 (Epsilon, MinPts, Core/Border/Noise Point 정의)**

3. **Mathematical Insight: 밀도 도달 가능성(Density-Reachability)과 연결성(Connectivity)**
   
4. **Deep Dive: 파라미터(ϵ, MinPts) 선정 (K-distance Graph 활용법)**
   
5. **Code & Visualization: Python (Scikit-learn) 구현 및 단계별 시각화)**
    
6. **Beyond DBSCAN: 한계점과 진화된 모델들 (OPTICS, HDBSCAN 소개)**
 
7. **Real-world Application: 실제 적용 사례**

8. **Conclusion & References: 요약 및 논문/자료 출처)**

</div>


---

##   <Intro> ‘군집화’는 왜 필요하며, 왜 하필 '밀도'인가? 
 "클러스터링(비슷한 데이터끼리 묶기)"는 데이터의 구조와 패턴을 파악하여 시너지 효과를 극대화 하는 데이터 과학의 핵심 기법이다. 데이터의 구조를 이해하고 데이터의 의미를 발견하여 숨어 있는 패턴까지 찾아내 업무자동화, 고객분석, 리스크 예측에 탁월한 효과를 보인다.
예시:
·	고객을 구매 패턴별로 그룹화
·	지도에서 비슷한 위치의 이벤트를 묶어 핫스팟 탐지
·	의료 데이터에서 질병 발생 지역 군집 파악
클러스터링을 입문할 때 보통 K-Means부터 시작하지만, K-Means는 동그랗고 비슷한 클러스터를 가정하기 때문에 초승달 모양이나 도넛 모양처럼 비선형 구조가 섞여 있는 복잡한 데이터의 경우 결과가 쉽게 왜곡되고 군집 개수 K를 미리 알아야 한다는 한계가 있다. 또한 이상치(outlier)에 민감해 군집의 중심이 끌려간다는 문제가 있다.
▼️ K-means의 한계:
1.	클러스터 개수(k)를 미리 지정해야 함
2.	구형(spherical) 클러스터만 잘 감지
3.	이상치(outlier)도 강제로 군집에 포함
4.	비선형·복잡한 형태의 군집은 감지 못함
이러한 한계를 극복하기 위해 밀도 기반 군집화(Density-Based Clustering)가 등장했다.

**_밀도 기반 군집화(density-based clustering)의 아이디어는 단순하다. “주변에 이웃이 빽빽하게 몰려 있는 점들은 같은 군집으로 보고, 외딴 점들은 노이즈로 버리자.” 이렇게 하면 군집 개수를 미리 정할 필요가 없고, 모양이 비선형이든 상관없이 “밀도가 높은 영역”만 잘 찾아내면 된다._**

 <img width="991" height="540" alt="image" src="https://github.com/user-attachments/assets/5a53e4ea-493e-4588-8b8a-3a3e1cecbcb9" />

그림 1이 그림은 초승달 데이터와 도넛 데이터에서 K‑Means와 DBSCAN을 비교한 결과를 보여준다. 위의 두 그래프는 K‑Means가 비선형 초승달 모양을 제대로 분리하지 못해 경계가 어색하게 갈리는 반면, DBSCAN은 두 개의 초승달 군집을 깔끔하게 분리하고 일부 점을 노이즈로 처리한다. 아래쪽 데이터에서도 K‑Means는 군집을 잘못 나누지만, DBSCAN은 각 고리를 하나의 군집으로 정확히 찾아내 밀도 기반 군집화의 강점을 직관적으로 드러낸다

⟪+ 더 알아보기 ⟫ 역사와 발전 (History & Evolution)
·	DBSCAN은 1996년 Ester, Kriegel, Sander, Xu가 논문에서 공개한 이래(‘A Density-Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise’) 군집 알고리즘의 표준이 되었다.
·	고정밀 공간 데이터와 이상치 분리 필요성이 늘어나면서 활용도가 폭증했고 이후 OPTICS, HDBSCAN, DENCLUE 같은 진화된 밀도기반 알고리즘이 개발되어, 다양한 밀도·복잡구조 데이터에 더 강력하게 대응하도록 변형되었다.

| 알고리즘 |특징 |사용 사례
|---------|------|------------|
|DBSCAN |	고정 반경 사용 |	균일 밀도 데이터 |
|HDBSCAN |	다변형 거리 설정 |	다양한 밀도 데이터 |
|OPTICS |	거리 정렬 구조 |	계층적 밀도 분석 |
|LOF(Local Outlier Factor) |	상대적 밀도 비교 |	로컬 이상치 탐지 |

<img width="965" height="384" alt="image" src="https://github.com/user-attachments/assets/79a652ba-424a-4313-8547-7e9711fba715" />

<Core Theory> DBSCAN 이해하기
DBSCAN(DBSCAN: Density-Based Spatial Clustering of Applications with Noise)은 밀도 기반 군집화 알고리즘의 대표 주자로서 두 가지 하이퍼파라미터만 있으면 된다. 반경 ϵ과 최소 이웃 수 MinPts이다. ϵ 은 이웃을 얼마나 넓게 볼지 정하는 거리 기준이고, MinPts는 그 반경 안에 최소 몇 개의 점이 있어야 “밀도가 충분하다”고 인정할지에 대한 기준이다. 이 두 값을 기준으로 각 점은 세 가지 타입으로 나뉜다.

<img width="738" height="389" alt="image" src="https://github.com/user-attachments/assets/0e49d52c-1545-4ea4-bc5b-d702a4f06de3" />

   **-핵심 점(Core point): 반경 ϵ 안에 MinPts 이상 이웃이 있는 점**
	**-경계 점(Border point): 핵심 점 근처에 있지만, 자기 자신은 핵심 조건을 만족하지 않는 점**
	**-노이즈(Noise): 어느 핵심 점과도 충분히 가깝지 않은 외딴 점**
▼▼알고리즘 작동 원리(DBSCAN Algorithm Flow)
1. 임의의 포인트 P 선택	 
2. P의 ε 반경 내 이웃 개수 확인	
3. 이웃 ≥ MinPts → P는 Core Point
   - P로부터 연결된 모든 Core/Border Point를 같은 군집으로 확장	
4. 이웃 < MinPts이지만 Core Point의 이웃 → Border Point	
5. 그 외 → Noise Point	
6. 모든 포인트가 분류될 때까지 반복	

<img width="608" height="801" alt="image" src="https://github.com/user-attachments/assets/12622cee-2d13-4456-8dc7-0cb3c1ea7bb4" />

DBSCAN은 임의의 점에서 시작해, 밀도가 충분한 이웃들을 계속 따라가며 확장해 나간다. 이렇게 연결된 점들의 집합이 하나의 군집이 되고, 어디에도 속하지 못한 점들은 이상치로 남는다. 이 때문에 DBSCAN은 군집 개수를 미리 정할 필요가 없고, 노이즈 검출까지 동시에 처리할 수 있는 것이다.

ex)
**Code 1. 가장 간단한 DBSCAN 예제 (numpy 배열 6개 점)**
•	from sklearn.datasets import make_moons, make_circles
import matplotlib.pyplot as plt

X_moon, y_moon = make_moons(n_samples=1000, noise=0.05, random_state=42)
X_circle, y_circle = make_circles(n_samples=1000, factor=0.5, noise=0.05, random_state=42)

-------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[^0].scatter(X_moon[:, 0], X_moon[:, 1], s=5)
axes[^0].set_title("Two Moons")
axes[^1].scatter(X_circle[:, 0], X_circle[:, 1], s=5)
axes[^1].set_title("Concentric Circles")
plt.show()
-------------------------------------------
기본 수식
	ϵ-이웃(ϵ-neighborhood):
N_ϵ (p)={q∈R^d∣‖p-q‖≤ϵ}
	핵심 점(Core point):
"core"(p) ⟺ |N_ϵ (p)|≥"MinPts" 
	경계 점(Border point): 핵심 점은 아니지만, 어떤 핵심 점 q의 ϵ-이웃 안에 있는 점.
	노이즈(Noise): 위 두 조건을 모두 만족하지 않는 점.
-------------------------------------------
**Code2. 기본 사용 코드**

from sklearn.cluster import DBSCAN

dbscan = DBSCAN(eps=0.3, min_samples=5)  # eps = ε, min_samples = MinPts
labels = dbscan.fit_predict(X_moon)

# -1 label은 노이즈
import numpy as np
unique_labels = np.unique(labels)
print("Clusters:", unique_labels)

# 시각화
plt.figure(figsize=(5, 4))
for lab in unique_labels:
    mask = labels == lab
    if lab == -1:
        plt.scatter(X_moon[mask, 0], X_moon[mask, 1], c="gray", s=5, label="Noise")
    else:
        plt.scatter(X_moon[mask, 0], X_moon[mask, 1], s=5, label=f"Cluster {lab}")
plt.legend()
plt.title("DBSCAN on Two Moons")
plt.show()
-------------------------------------
**<Mathematical Insight > — 밀도 도달 가능성과 연결성**

DBSCAN의 군집은 밀도 도달 가능성(density‑reachability) 과 밀도 연결성(density‑connectivity) 으로 정의된다. 점 p가 점 q로부터 직접 밀도 도달 가능하려면, p가 q의 ϵ-이웃 안에 있고, q가 핵심 점이어야 한다. 여러 개의 직접 도달 가능한 점들을 체인처럼 이어 나가서 q=p_1,p_2,…,p_n=p 를 만들 수 있다면, p는 q로부터 밀도 도달 가능하다고 정의한다.
두 점 p,q가 어떤 제3의 점 o로부터 모두 밀도 도달 가능하면, 이 둘은 밀도 연결되어 있다고 본다. DBSCAN은 “서로 밀도 연결된 점들의 최대 집합”을 하나의 군집으로 취급한다. 따라서 군집의 외곽이 꼬불꼬불하든 끊겨 보이든 상관없이, 실제로 고밀도 영역이 이어져 있으면 하나의 클러스터로 묶을 수 있고, 밀도가 급격히 떨어지는 영역은 자연스럽게 군집 경계가 된다.

<img width="991" height="540" alt="image" src="https://github.com/user-attachments/assets/faa4c093-dc62-4537-a28e-25b5c02d2006" />

수식 정의
	직접 밀도 도달 가능(Directly density-reachable):
〖"DirReach" 〗_ϵ (p←q) ⟺ (p∈N_ϵ (q))∧"core"(q)
	밀도 도달 가능(Density-reachable):
〖"Reach" 〗_ϵ (p←q) ⟺ ∃{p_1,…,p_n}:p_1=q,p_n=p,∀i,〖"DirReach" 〗_ϵ (p_(i+1)←p_i)
	밀도 연결(Density-connected):
〖"Conn" 〗_ϵ (p,q) ⟺ ∃o:〖"Reach" 〗_ϵ (p←o)∧〖"Reach" 〗_ϵ (q←o)
--------------------------------------
**Code3. reachability 개념 이해 코드**

from sklearn.neighbors import NearestNeighbors

k = 4  # MinPts와 비슷하게 잡음
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X_moon)
distances, indices = nn.kneighbors(X_moon)

# 각 점의 k번째 이웃 거리 (core distance 후보)
core_dist = distances[:, -1]
-------------------------------------
**Deep Dive — ϵ·MinPts 선정의 예술**

DBSCAN에서 가장 중요한 것 중 하나가 파라미터 튜닝이다. MinPts는 보통 데이터 차원 수보다 하나 이상 크게 잡고(예: 2D 데이터면 4 이상), 실무에서는 4~10 사이 값에서 시작해 보는 경우가 많다. ϵ 은 k‑distance plot으로 정하는 것이 표준적이다. 각 점에 대해 k="MinPts" 번째 최근접 이웃까지의 거리를 계산하고, 이 값을 오름차순으로 나열해 그리면 그래프가 처음에는 완만히 증가하다가 어느 지점부터 급격히 꺾인다. 
k-거리 그래프 함수 f(j)=d_k (p_((j))) 
각 점 p_i에 대해 k번째 최근접 이웃 거리 d_k (p_i) 를 계산하고, 이를 오름차순 정렬한 함수 f(j) 를 그린다.
f(j)=d_k (p_((j))),d_k (p_((1)))≤⋯≤d_k (p_((n)))
여기서 곡률이 급격히 변하는 지점(“엘보”)의 y값이 ϵ 후보가 된다.
<img width="991" height="540" alt="image" src="https://github.com/user-attachments/assets/061e967e-21c7-40b4-bfc7-d4b4bfc6b31a" />

그림 3이 그래프는 DBSCAN에서 적절한 ϵ 값을 선택하기 위해 사용하는 k-거리 플롯(k-distance plot) 을 보여준다. 각 점의 k번째 최근접 이웃까지의 거리를 오름차순으로 정렬해 그리면 처음에는 완만히 증가하다가, 어느 지점부터 급격히 꺾이는 ‘엘보(elbow)’ 형태가 나타나는데, 그래프에 표시된 빨간 점이 바로 그 엘보 지점으로 좋은 ϵ 후보 값을 의미한다.

ϵ 이 너무 작으면 대부분의 점이 서로 연결되지 못해 군집이 잘게 쪼개지거나 노이즈가 과도하게 많아진다. 반대로 ϵ 이 너무 크면 떨어져 있어야 할 군집들까지 하나의 큰 덩어리로 합쳐져 세부 구조를 잃어버린다. 같은 데이터를 대상으로 ϵ 값을 여러 개 바꿔가며 DBSCAN을 적용한 3×3 그리드 이미지를 함께 보면, “너무 작음–적당함–너무 큼”에 따라 군집 구조가 어떻게 무너지는지 직관적으로 이해할 수 있다. 

<img width="991" height="991" alt="image" src="https://github.com/user-attachments/assets/83d0fa7b-bb22-413b-993b-525d38b5f1d1" />
<그림 4 “DBSCAN에서 핵심 하이퍼파라미터인 ϵ”>

ϵ 값에 따라 동일한 데이터라도 군집 결과가 어떻게 달라지는지를 보여준다. 왼쪽(ε = 0.1, 너무 작음)에서는 대부분의 점이 노이즈로 남거나 잘게 쪼개진 작은 군집들로만 나타나고, 가운데(ε = 0.5, 최적 구간)에서는 밀도가 높은 영역이 적절한 개수의 클러스터로 깔끔하게 분리된다. 오른쪽(ε = 1.0, 너무 큼)에서는 서로 다른 집단까지 하나의 거대한 군집으로 뭉개져 버려, ε 선택이 과소군집·과대군집을 결정하는 핵심이라는 점을 직관적으로 보여준다.
----------------------------------
   Deep Dive — ϵ·MinPts 선택
	code4. k-distance plot : NearestNeighbors로 k번째 거리 계산, plt.plot으로 그래프 생성.
	k = 4
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X_moon)
distances, _ = nn.kneighbors(X_moon)

k_distances = np.sort(distances[:, -1])
plt.figure(figsize=(6, 4))
plt.plot(k_distances)
plt.ylabel("k-distance")
plt.xlabel("Points sorted by distance to k-th nearest neighbor")
plt.title("k-distance plot for ε selection")
plt.show()

   epsilon 민감도 그리드 코드 스케치
	eps_values = [0.1, 0.2, 0.3]
min_samples_values = [3, 5, 10]

fig, axes = plt.subplots(len(min_samples_values), len(eps_values), figsize=(9, 9))
for i, ms in enumerate(min_samples_values):
    for j, eps in enumerate(eps_values):
        model = DBSCAN(eps=eps, min_samples=ms)
        labels = model.fit_predict(X_moon)
        ax = axes[i, j]
        ax.scatter(X_moon[:, 0], X_moon[:, 1], c=labels, s=5, cmap="tab10")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"eps={eps}, MinPts={ms}")
plt.tight_layout()
plt.show()
-------------------------------------
**< Code & Visualization — Python 실습>**

이제 코드를 통해 직접 눈으로 확인해 보자. Python에서는 scikit‑learn의 DBSCAN 클래스를 사용해 몇 줄만으로 밀도 기반 군집화를 시도할 수 있다. make_moons, make_circles 같은 toy 데이터셋을 활용하면, K‑Means가 실패하는 비선형 구조를 DBSCAN이 어떻게 자연스럽게 복원하는지 한눈에 볼 수 있다.
 먼저 데이터를 생성하고, K‑Means와 DBSCAN을 각각 학습시킨다. 그런 다음 2×2 또는 3×1 레이아웃으로 결과를 시각화해 “K‑Means vs DBSCAN”, epsilon 변화에 따른 군집 구조 변화, 노이즈 포인트 시각화 등을 반복해서 보여 준다. 이 패턴만 익혀 두면, 뒤에서 다룰 도시 교통 데이터나 금융 거래, 반도체 웨이퍼 데이터에도 그대로 응용할 수 있다.

**Code5. Python으로 DBSCAN 살려 보기**

from sklearn.cluster import KMeans

# K-Means vs DBSCAN on two moons
kmeans = KMeans(n_clusters=2, random_state=42)
km_labels = kmeans.fit_predict(X_moon)

dbscan = DBSCAN(eps=0.2, min_samples=5)
db_labels = dbscan.fit_predict(X_moon)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[^0].scatter(X_moon[:, 0], X_moon[:, 1], c=km_labels, s=5, cmap="tab10")
axes[^0].set_title("K-Means")

axes[^1].scatter(X_moon[:, 0], X_moon[:, 1], c=db_labels, s=5, cmap="tab10")
axes[^1].set_title("DBSCAN")

for ax in axes:
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("K-Means vs DBSCAN on Two Moons")
plt.show()
-------------------------------------------
**< 장단점 & 언제 사용하는 것이 좋을까>**
장점
★ K‑Means로는 잡기 어려운 비선형 군집과 이상치를 한 번에 처리할 수 있다.
★ 군집 개수를 사전에 정할 필요가 없다.
★ 밀도가 높은 영역만 기준으로 삼기 때문에, 원형·초승달·도넛·곁가지가 많은 형태 등 다양한 모양의 클러스터를 자연스럽게 찾아낸다.
★  군집에 속하지 못한 외따로 떨어진 점을 노이즈로 자동 분리하므로, 이상치 탐지의 백본 알고리즘으로도 널리 쓰인다. 
단점
☆ 파라미터 ϵ, MinPts 튜닝에 민감하다.
☆ 데이터마다 적절한 밀도 스케일이 달라 하나의 ϵ 값으로는 서로 다른 밀도를 가진 영역을 동시에 잘 설명하기 어렵다.
☆ 차원이 높아질수록 거리 개념 자체가 희석되기 때문에, 성능이 급격히 떨어지고 오히려 K‑Means나 다른 방법이 더 안정적일 수 있다.
☆ 잡음이 많고 밀도가 조금씩 다른 군집이 섞여 있는 데이터에서는 한 쪽 밀도에 맞추면 다른 쪽이 무너지기 쉽다. 
그렇다면 언제 DBSCAN을 쓰는 것이 좋을까? 
	군집 개수를 미리 알 수 없고, 데이터의 분포가 복잡하거나 비선형일 때
	이상치 탐지가 중요하고, “노이즈 분리” 자체가 목적일 때
	위치 기반(GPS, 지도), 자연현상, 로그 데이터 등, 거리·밀도 해석이 직관적인 저차원 데이터일 때
반대로,
	고차원 특성 공간,
	매우 큰 데이터셋에서의 실시간 처리,
	군집 개수가 명확하고 거의 구형에 가까운 구조가 예상될 때는 K‑Means·가우시안 혼합·스펙트럴 클러스터링 등을 먼저 고려하는 편이 합리적이다. 

<img width="864" height="535" alt="image" src="https://github.com/user-attachments/assets/69ab766f-4854-49e9-94b7-71fe8e805f9e" />
-----------------------------------------------

**<Real-world Application: 실제 데이터 적용 사례 (지도 데이터, 이상치 탐지 등)>**

1.	도시 교통 빅데이터: 택시·버스 GPS 좌표를 DBSCAN으로 군집화 → 승하차 핫스팟 분석 → 도시 내 교통 혼잡구역, 신규 노선 후보 발굴 

<img width="991" height="540" alt="image" src="https://github.com/user-attachments/assets/d60114c9-f4e4-4947-909c-3c31a7856f05" />

그림 4도시 전체 지도 위에 택시·버스 GPS 좌표를 DBSCAN으로 군집화한 결과를 시각화한 그림이다. 중심부의 대형 클러스터는 ‘Downtown Transfer Hub’와 같은 주요 환승 거점의 승하차 집중 구역을 나타내고, 주변의 여러 붉은·주황색 원은 대학가, 공항 셔틀 정류장 등 추가적인 승객 밀집 지점을 보여준다. 이러한 핫스팟 간을 잇는 점선 화살표는 실제 교통 빅데이터를 활용해 교통 혼잡 구역 파악과 신규 노선 후보를 자동으로 발굴할 수 있음을 직관적으로 설명한다.

2.	이상거래/금융사기 탐지: 거래 금액·장소·패턴의 밀도기반 군집화 → 노이즈가 의미있는 이상거래로 분리

<img width="991" height="540" alt="image" src="https://github.com/user-attachments/assets/34d01a04-b376-4bff-b183-6709c56b707e" />

그림 5이 그림은 신용카드 거래 데이터를 DBSCAN으로 분석해 정상 소비 패턴과 이상 거래(사기)를 구분하는 과정을 시각화한 것이다. 파란색으로 조밀하게 모여 있는 두 개의 군집은 금액·빈도가 안정적인 ‘일상적 소비(정상 클러스터)’를 나타내고, 주변에 듬성듬성 흩어진 빨간 사각형 점들은 해외 결제나 야간의 대액 결제처럼 정상 군집에서 멀리 떨어진 노이즈 포인트 = 이상 거래(사기 의심) 로 표시된다. 오른쪽 상단의 작은 박스에는 금액, 시간대, 위치, 빈도와 같은 주요 입력 특성이 정리되어 있어, 이런 특징 공간에서 DBSCAN이 밀도가 높은 영역을 정상으로, 희소한 점들을 이상치로 자동 식별한다는 점을 한눈에 보여준다.

3.	반도체 공정결함 분석: 웨이퍼 표면 결함점 좌표로 DBSCAN → 결함핫스팟 자동탐지→장비 이상, 공정 편차 빠르게 발견하여 품질 개선 

<img width="991" height="540" alt="image" src="https://github.com/user-attachments/assets/bee122f0-8e37-4f0b-a84c-f0501639b3fc" />


4. 감염병 핫스팟 식별: 환자 위치 데이터로 질병확산 군집 분석 →고위험 지역 우선 방역전략 수립
5. 부동산 시장 분석: 실거래가, 위치 기반 DBSCAN → 유사 가격권/이상거래군 자동분류

-------------------------------
##  사용 방안 고찰

### 버스·지하철 승하차 데이터를 활용한 도시 핫플레이스 탐구

DBSCAN은 공간 기반 데이터를 다루는 데 특히 강력하기 때문에, 대중교통 승하차 데이터 분석에 매우 효과적으로 활용될 수 있다.  
버스 정류장·지하철역 승하차량은 시·공간적 밀도 변화가 뚜렷하게 나타나므로, DBSCAN의 ‘밀도 기반 군집화’ 속성과 직접적으로 연결된다.
이러한 특성은 교통 혼잡의 분석에 사용되지만 관점을 바꾸어 교통량이 집중되는 곳을 핫플레이스라 지정하고 요즘 트렌드와 유행을 지리적으로 알아차릴 수 있다.

---

###  1. 데이터 선정

- **버스 승·하차 데이터**  
  - 정류장 코드, 좌표(위·경도), 시간대별 승·하차 인원  
- **지하철 승·하차 데이터**  
  - 역 ID, 위치 정보, 출입구 단위 승·하차량  

이 데이터를 활용하면 다음 단계로 군집 구조를 만들 수 있다.

---

###  2. DBSCAN 적용 방식

핵심 아이디어는 특정 시간대·특정 지역에서 승하차 인원이 과밀하게 형성된 지점을 군집(Hotplace)으로 파악하는 것이다.

1. 모든 버스 정류장·지하철역을 하나의 데이터셋으로 통합  
2. 각 포인트에 다음과 같은 특징(feature)을 부여  
   - 공간 정보: (위도, 경도)  
   - 시간 가중 특징: 특정 시간대의 승차량 또는 하차량  
   - 평일/주말 구분, 출퇴근 시간대 인자 추가 가능  
3. DBSCAN으로 군집화  
   - 군집 = 특정 시간대에 사람들이 몰리는 ‘핫플레이스’
   - Noise = 비정상적으로 튀는 값 → 특별 이벤트 혹은 이상 패턴

---

###  3. 기대되는 분석 결과

#### **① 시간대별 도시 핫플레이스 탐색**
- 오전 7–9시 → 출근 중심지(대기업 밀집 구역, 지하철 환승역)  
- 오후 6–8시 → 퇴근·저녁 시간 상권  
- 주말 오후 → 쇼핑몰, 공원, 전시회 주변  
- 심야 시간대 → 유흥 밀집 거리  

DBSCAN은 군집 수를 미리 정하지 않기 때문에  
**자연스럽게 드러나는 진짜 도시의 흐름(people flow)**을 파악할 수 있다.

---

#### **② 지역 정책 설계에 활용**
- 승·하차 과밀 지역 → 버스 노선 재조정, 지하철 증편  
- 특정 군집이 갑자기 팽창 → 이벤트·행사 감지  
- 외곽 지역에 작은 군집 발생 → 신규 거점 개발 후보 탐지  

---

#### **③ 관광·소비 트렌드 파악**
- ‘SNS 인기 지역’, ‘맛집 거리’, ‘유흥가’ 등이  
  **실제 이동 데이터 기반으로 검증된 핫플레이스**로 확인  
- 계절·기온·날씨에 따라 군집 분포가 어떻게 이동하는지도 분석 가능  

---

#### **④ 안전·방범 활용**
- 심야 시간대 군집 감소 지역 → 방범 취약 지역  
- 군집이 특정 골목에 집중 → CCTV 필요 구역  

---

###  4. 실제 흐름 예시 (가상의 시나리오)

- 오전 8시: 강남역·여의도역·시청역 주변 군집 폭발 → 직장 중심  
- 오후 2시: 홍대·성수·잠실 롯데월드타워 군집 증가 → 관광/여가  
- 오후 8시: 건대입구·홍대입구역 주변 군집 최대 → 저녁 식사 중심  
- 심야 1시: 이태원·홍대 일부 골목만 군집 유지 → 야간 상권  

이처럼 시간에 따라 ‘핫플레이스의 이동’을 시각적으로 추적할 수 있다.

---

###  5. 결론: DBSCAN의 강점과 활용 가치

버스·지하철 승하차 데이터는 도시의 실제 움직임을 가장 잘 보여주는 데이터 중 하나다.  
DBSCAN을 적용하면,

- 미리 군집 수를 정할 필요 없고  
- 이상치(Noise)를 자연스럽게 분리하고  
- 복잡하고 비정형적인 공간 패턴을 그대로 반영할 수 있어  

**도시 핫플레이스 탐색, 교통 정책 설계, 상권 분석, 안전 시스템 구축** 등 다양한 분야에 직접 활용 가능하다.

---
###  Part 8. 결론 및 추가 자료

핵심 요약
- DBSCAN은 K-means로는 잡히지 않는 복잡·비구형 군집이나 노이즈를 잘 
구분하는 강력한 도구
- 단, eps·MinPts 등 파라미터 조정 필요성과 데이터의 밀도 특성 고려가 핵심



**< 결론 & 추천 자료 >**

정리하자면, DBSCAN은 “밀도가 높은 영역=군집”이라는 단순한 아이디어로부터 출발해, K‑Means가 놓치는 복잡한 클러스터 구조와 이상치를 동시에 다룰 수 있는 강력한 도구다. ϵ 과 MinPts만 잘 고르면, 군집 개수를 사전에 정하지 않아도 패턴을 안정적으로 찾아내고, 외따로 떨어진 점은 자연스럽게 노이즈로 분리된다. 
그러나 파라미터 선택에 민감하고, 밀도 스케일이 여러 개 섞인 데이터나 고차원 데이터에서는 성능이 불안정할 수 있다는 한계를 보완하기 위해, 여러 밀도 수준을 한 번에 표현하는 OPTICS, 계층 구조와 안정도 개념을 도입한 HDBSCAN 같은 후속 알고리즘들이 제안되었다. 실제 사용시, 데이터 특성(노이즈 정도, 밀도 편차, 차원) 을 먼저 진단한 뒤, DBSCAN·OPTICS·HDBSCAN·K‑Means 등을 상황에 맞게 조합하여야 한다.
마지막으로, 더 깊이 공부할 수 있는 자료를 정리하면 다음과 같다.
	논문
	Ester et al., 1996, “A Density‑Based Algorithm for Discovering Clusters in Large Spatial Databases with Noise (DBSCAN)”
	Ankerst et al., 1999, “OPTICS: Ordering Points To Identify the Clustering Structure”[6]
	Campello et al., 2013, “A Hierarchical Density‑Based Clustering Method and Its Applications (HDBSCAN)”
	이 글의 목표는 “DBSCAN을 처음 접하는 독자가, 이론–수식–코드–시각화–응용 사례까지 한 번에 훑고, 실제 프로젝트에 바로 써 볼 수 있도록 돕는 것”이다. 앞에서 살펴본 예제와 코드, 시각화 템플릿만 잘 변형해도, 도시 교통, 금융 사기, 반도체, 감염병, 부동산 등 다양한 도메인에 밀도 기반 군집화를 적용해 볼 수 있을 것이다.
	code6. Real-world Applications — 간단 예시 코드 스케치
	예: 위도·경도 데이터에서 밀집 영역 찾기 (실제 블로그에서는 사용자 데이터로 교체).
	import pandas as pd
from sklearn.preprocessing import StandardScaler

# df: columns = ["lat", "lon"]
coords = df[["lat", "lon"]].values
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

db = DBSCAN(eps=0.2, min_samples=10)
labels = db.fit_predict(coords_scaled)

plt.figure(figsize=(6, 6))
plt.scatter(df["lon"], df["lat"], c=labels, s=5, cmap="tab20")
plt.xlabel("Longitude"); plt.ylabel("Latitude")
plt.title("Density-based clustering of geographic coordinates")
plt.show()

 
Beyond DBSCAN — OPTICS와 HDBSCAN
OPTICS 핵심 수식 (reachability / core distance)
	코어 거리(core distance):
〖"core\_dist" 〗_"MinPts"  (p)={■("distance to MinPts-th nearest neighbor of " p,&|N_ϵ (p)|≥"MinPts" @∞,&"otherwise" )┤
	도달 가능 거리(reachability distance):
"reachability\_dist"(p∣o)=max(〖"core\_dist" 〗_"MinPts"  (o),d(o,p))
OPTICS는 각 점을 방문하면서 이 reachability distance를 기록하고, 이를 y축으로 하는 reachability plot을 만든다.
code7. OPTICS 간단 코드
from sklearn.cluster import OPTICS

optics = OPTICS(min_samples=10, xi=0.05, min_cluster_size=0.05)
optics_labels = optics.fit_predict(X_moon)

plt.figure(figsize=(5, 4))
plt.scatter(X_moon[:, 0], X_moon[:, 1], c=optics_labels, s=5, cmap="tab10")
plt.title("OPTICS clustering")
plt.show()

 
HDBSCAN 개념 수식: HDBSCAN은 다양한 MinPts(논문에서는 m_pts) 값에 대해 mutual reachability distance 로 그래프를 만들고, 그 MST에서 여러 밀도 레벨의 군집 계층을 도출한다.
	mutual reachability distance:
d_"mreach"  (p,q)=max("core\_dist"(p),"core\_dist"(q),d(p,q))
이 거리를 가중치로 하는 MST 위에서, 거리를 증가시키며 군집이 분리되는 과정을 추적하면 “밀도 기반 덴드로그램”을 얻을 수 있다.
code8. HDBSCAN 파이썬 예시 
!pip install hdbscan

import hdbscan

clusterer = hdbscan.HDBSCAN(min_cluster_size=20, min_samples=5)
hdb_labels = clusterer.fit_predict(X_moon)

plt.figure(figsize=(5, 4))
plt.scatter(X_moon[:, 0], X_moon[:, 1], c=hdb_labels, s=5, cmap="tab10")
plt.title("HDBSCAN clustering")
plt.show()


 
   https://hex.tech/blog/comparing-density-based-methods/      
	https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html   
	http://techblog.netflix.com/2015/07/tracking-down-villains-outlier.html 
	https://blog.dailydoseofds.com/p/the-limitations-of-dbscan-clustering  
	https://en.wikipedia.org/wiki/OPTICS_algorithm   
	https://domino.ai/blog/topology-and-density-based-clustering 
	https://pro.arcgis.com/en/pro-app/latest/tool-reference/spatial-statistics/how-density-based-clustering-works.htm 
	https://scikit-learn.org/stable/modules/clustering.html    
	HDBSCANnonmun.pdf  
	https://www.scirp.org/reference/referencespapers  
	https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html 
	https://machinecurve.com/index.php/2020/12/09/performing-dbscan-clustering-with-python-and-scikit-learn 
	https://www.datacamp.com/tutorial/dbscan-clustering-algorithm 
	https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/ 
	https://www.nature.com/articles/s41598-021-88822-3 
	https://thesai.org/Downloads/Volume14No11/Paper_85-Hotspot_Identification_Through_Pick_Up.pdf 
	https://pure.seoultech.ac.kr/en/publications/a-unified-defect-pattern-analysis-of-wafer-maps-using-density-bas/ 
	https://hex.tech/blog/comparing-density-based-methods/
	https://www.baeldung.com/cs/k-means-flaws-improvements 
	https://www.geeksforgeeks.org/machine-learning/k-means-clustering-introduction/ 
	https://domino.ai/blog/topology-and-density-based-clustering  
	https://www.getfocal.co/post/how-density-based-clustering-works  
	https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html   
	https://machinecurve.com/index.php/2020/12/09/performing-dbscan-clustering-with-python-and-scikit-learn    
	https://programminghistorian.org/en/lessons/clustering-with-scikit-learn-in-python   
	https://www.newhorizons.com/resources/blog/dbscan-vs-kmeans-a-guide-in-python 
	https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html 
	https://www.naftaliharris.com/blog/visualizing-dbscan-clustering/ 
	https://blog.dailydoseofds.com/p/the-limitations-of-dbscan-clustering  
	https://www.youtube.com/watch?v=FxBZ5D9o1HU  
	https://blog.quantinsti.com/dbscan-vs-kmeans/ 
	https://www.naftaliharris.com/blog/ 
	https://www.sciencedirect.com/science/article/abs/pii/S1568494624001935 
	https://gregorredinger.github.io/vis_clustering_algorithms/M1/index.html 
	http://techblog.netflix.com/2015/07/tracking-down-villains-outlier.html 
	https://scikit-learn.org/stable/modules/clustering.html 
	https://www.scirp.org/reference/referencespapers 
	https://github.com/chriswernst/dbscan-python
	 https://www.datacamp.com/tutorial/dbscan-clustering-algorithm 



---



```python
from sklearn.cluster import DBSCAN
import numpy as np

X = np.array([[1, 2], [2, 2], [2, 3], [8, 7], [8, 8], [25, 80]])
dbscan = DBSCAN(eps=3, min_samples=2)
labels = dbscan.fit_predict(X)

print(labels)
