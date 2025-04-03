ℹ️ *이 레포지토리의 모든 내용물은 본인이 직접 작성한 소스 및 자료들만 포함하고 있습니다. 본인이 작성하지 않은 부분이 조금이라도 포함된 소스는 보관하지 않습니다.*

---
## Autograd의 원리를 직접 구현하고 최적화하기

### 프로젝트 개요

- 본문: [autograd-from-scratch.ipynb](https://github.com/hkyoon94/autograd-from-scratch/blob/main/autograd-from-scratch.ipynb)

이 프로젝트는, 제가 딥러닝 연구자로서 약 2년간 직접 `PyTorch`를 사용하며 모델을 개발하면서 자연스럽게 갖게 된 질문에서 출발했습니다:  

>"Autograd는 내부적으로 어떻게 작동할까? 그리고 어떻게 그렇게 유연하고 빠르게 작동할 수 있을까?"

이를 수학적인 기초부터 차근차근 탐구하기 위해:

1. 연쇄법칙(chain rule)과 텐서 수축의 원리를 정리하며 수동 autograd를 직접 구현해보았고,
2. 현재 PyTorch가 autograd 엔진에 사용 중인 계산 그래프 추적과 벡터-자코비안 곱(VJP)을 사용한 연산 방식으로 최적화를 해보고 PyTorch와 성능 비교를 하였으며,
3. 2에 더해 VJP를 수행하는 C++ 커널을 직접 작성하는 최적화 실험을 진행해보았습니다.

위 순서대로 프로젝트를 진행한 결과, 다음과 같은 결과를 얻을 수 있었습니다.

| 구현 방식 | Backward 총 소요 시간 (1000 iter) | Iteration당 소요 시간 |
|------------------|-----------------------|------------------------|
| Jacobi 텐서를 명시적으로 생성해 einsum으로 chain rule 계산 | 301.25 sec | 0.301 sec |
| einsum을 활용한 VJP 기반 연산 |  0.23 sec | 0.00023 sec |
| PyTorch Autograd |  0.33 sec | 0.00033 sec |
| **커스텀 C++ 커널 기반 수동 VJP 연산** |  **0.15 sec** | **0.00015 sec** |

---
### 프로젝트 목차

>0. 기초 이론 정리
>
>1. Chain Rule과 Jacobian Contraction을 직접 이용해 구현하기
>   - 토이 데이터 준비
>   - 최적화를 고려하지 않은 Jacobian contraction 기반의 backward 구현
>
>2. 연산 그래프와 Vector-Jacobi Product(VJP)를 이용해 역전파 최적화하기
>    - 연산 그래프 확인 및 common subgraph 추출
>    - 연산별 VJP의 유도 및 구현
>    - PyTorch의 Autograd & SGD와의 성능 비교
>
>3. C++ 소스를 작성하여 backward pass 최적화하기
>
>4. 결론
