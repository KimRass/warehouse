# Watershed Algorithm
- Source: https://m.blog.naver.com/PostView.naver?blogId=laonple&logNo=220902777415&targetKeyword=&targetRecommendationCode=1
- 여기서는 가장 많이 쓰이는 Flooding 알고리즘에 대하여 살펴볼 예정이다. 이 알고리즘은 기본적으로 각각의 Catchment Basin의 최소값(local minima)에 구멍이 있고, 그 구멍에서 물이 차오르기 시작한다고 가정에서 출발한다. 물이 일정 수위에 오르게 되면, 서로 다른 2개의 Catchment Basin이 합쳐지는 상황이 발생할 수 있는데, 이 때는 물이 합쳐지는 막기 위해 댐(dam)을 설치한다. 이 댐이 바로 영역을 구별하는 영역의 경계(boundary) 역할을 하게 되며, 이것이 Flooding 알고리즘의 기본 개념이다.
- 아래 그림 A는 3개의 basin에 각각 마커를 할당한 경우로, 각 basin에서 물이 점점 차오르면, ﻿어느 순간에 2개의 basin이 합쳐지는 상황이 발생하게 되는데, 그러면 그 자리에 댐을 건설한다. B는 애초에 2 개의 마커만을 할당하였기 때문에, 최종적으로는 영역이 2개로 나눠지게 된다.
- ![watershed](https://mblogthumb-phinf.pstatic.net/MjAxNzAxMDRfMTg0/MDAxNDgzNTI1Mzg4ODAy.WgEGwwNTFUhAc_qTSPLhcfZLo7dRBHLJK3b_iTNVW6og.BBkGw1mXXGCZonkcJLgAmuBYHPZLuvXJgJBjQ_F-dpcg.PNG.laonple/%EC%9D%B4%EB%AF%B8%EC%A7%80_3.png?type=w2)
```python
distance = ndimage.distance_transform_edt(image)
coords = peak_local_max(distance, footprint=np.ones((3, 3)), labels=image)
mask = np.zeros(distance.shape, dtype=bool)
mask[tuple(coords.T)] = True
markers, _ = ndimage.label(mask)
segmentation_map = watershed(-distance, markers, mask=image)
```