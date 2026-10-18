# -*- coding: utf-8 -*-
"""정통 DeepONet 재설계.
CurveOperatorV2 : branch=CNN(곡선)+vol·corr, trunk=MLP(계약). 데이터기반. stage-1 앵커.
MarginOperator  : stage-2 잔차 DeepONet(내적). branch=시장상태, trunk=계약 → 내적으로 가격.
PIStateDeepONet : CurveOperatorV2 + trunk 에 경로상태(스팟·러닝최소·잔존만기·KI) 증강. PI 앵커."""
import torch
import torch.nn as nn


def mlp(d, p=128):
    """branch·trunk 공용 3층 MLP 를 만든다.

    마지막 층에 활성함수가 **없다** — 출력이 내적에 들어가는 잠재벡터이므로 부호와 크기를
    자유롭게 둬야 한다.

    입력
      d : int — 입력 차원
      p : int — 은닉 폭 = 출력 차원 (잠재 차원 P)
    출력
      nn.Sequential — Linear(d,p) → Tanh → Linear(p,p) → Tanh → Linear(p,p)
    """
    return nn.Sequential(nn.Linear(d, p), nn.Tanh(), nn.Linear(p, p), nn.Tanh(), nn.Linear(p, p))


class MarginOperator(nn.Module):
    """stage-2 잔차 DeepONet: branch=MLP(vol·corr·sig_eff), trunk=MLP(곡선+계약). 내적으로 잔차 회귀."""

    def __init__(self, nb, nt, P):
        """branch·trunk MLP 한 쌍과 학습 가능한 편향을 만든다.

        입력
          nb : int — branch 입력 차원 (vol·corr·sig_eff)
          nt : int — trunk 입력 차원 (곡선 + 계약)
          P  : int — 내적 잠재 차원
        출력
          없음 (파라미터 등록만 한다)
        """
        super().__init__()
        self.b = mlp(nb, P)  # branch: 리스크(vol·corr) + 금리
        self.t = mlp(nt, P)  # trunk: 나머지(계약+발행+범주형 one-hot)
        self.b0 = nn.Parameter(torch.zeros(1))

    def V(self, bx, tx):
        """branch·trunk 잠재벡터를 내적해 잔차를 낸다.

        입력
          bx : Tensor(n, nb) — 표준화된 branch 입력
          tx : Tensor(n, nt) — 표준화된 trunk 입력
        출력
          Tensor(n,) — 표준화 공간의 잔차. 원본 스케일 복원은 호출부 몫이다
        """
        return (self.b(bx) * self.t(tx)).sum(-1) + self.b0


class CurveOperatorV2(nn.Module):
    """DeepONet-Curve: branch=1D-CNN(곡선)+vol·corr 융합, trunk=계약. 스팟 없음."""

    def __init__(self, nvc, ncon, P):
        """곡선 CNN · 융합 branch · 계약 trunk · 편향을 만든다.

        CNN 은 10노드 곡선을 32차원으로 요약한다 (Conv→ReLU→MaxPool→Conv→ReLU→전역평균).
        곡선을 MLP 에 평평하게 넣지 않는 이유는 노드 순서가 만기축이라 국소 형태(기울기·휘어짐)가
        의미를 갖기 때문이다.

        입력
          nvc  : int — vol·corr 블록 차원
          ncon : int — 계약 차원 (정본은 51)
          P    : int — 내적 잠재 차원 (정본은 192)
        출력
          없음 (파라미터 등록만 한다)
        """
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.branch = nn.Sequential(
            nn.Linear(32 + nvc, 128), nn.Tanh(), nn.Linear(128, P), nn.Tanh(), nn.Linear(P, P)
        )
        self.t = mlp(ncon, P)  # trunk: 계약
        self.b0 = nn.Parameter(torch.zeros(1))

    def V(self, curve, vc, con):
        """시장상태(곡선+vol·corr)와 계약을 각각 잠재공간으로 보내 내적한다.

        입력
          curve : Tensor(n, 10) — 표준화된 제로커브 노드
          vc    : Tensor(n, nvc) — 표준화된 vol·corr 블록
          con   : Tensor(n, ncon) — 표준화된 계약 블록
        출력
          Tensor(n,) — 표준화 공간의 가격. 스팟은 입력에 없다 (발행시점 평가라 전부 1.0)
        """
        hc = self.cnn(curve.unsqueeze(1)).squeeze(-1)  # (n, 32)
        branch = self.branch(torch.cat([hc, vc], -1))  # (n, P)  시장상태(곡선+vol·corr)
        return (branch * self.t(con)).sum(-1) + self.b0


class PIStateDeepONet(nn.Module):
    """PI 앵커: CurveOperatorV2 와 같은 branch(곡선 CNN + vol·corr) + trunk 에 **경로상태 10차원** 증강.

    데이터항만 쓰면 CurveOperatorV2 와 사실상 동등하지만(발행시점 상태로 고정 평가),
    trunk 가 임의의 (S, M, tau) 를 받으므로 PDE 잔차·이벤트 경계조건을 걸 수 있다.
    상태 규약은 module.pi_physics 의 docstring 참고 (S1..3, M1..3, tau, KI, event_pre, obs_frac)."""

    state_dim = 10

    def __init__(self, nvc, ncon, P):
        """CurveOperatorV2 와 같은 branch + trunk 입력만 state_dim 만큼 넓힌 망을 만든다.

        trunk fan_in 이 ncon+10 으로 커져 초기화 분산이 작아진다 — 물리항을 끄고 state 를
        마스킹해도 **시드 분산이 줄어드는** 부수효과가 있다 (최종 R² 시드 sd 0.0118 → 0.0031).

        입력
          nvc  : int — vol·corr 블록 차원
          ncon : int — 계약 차원. trunk 는 여기에 state_dim(10) 을 더해 받는다
          P    : int — 내적 잠재 차원
        출력
          없음 (파라미터 등록만 한다)
        """
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.branch = nn.Sequential(
            nn.Linear(32 + nvc, 128), nn.Tanh(), nn.Linear(128, P), nn.Tanh(), nn.Linear(P, P)
        )
        self.t = mlp(ncon + self.state_dim, P)  # trunk: 계약 + 경로상태
        self.b0 = nn.Parameter(torch.zeros(1))

    def V(self, curve, vc, con, state):
        """계약에 경로상태를 이어 붙여 trunk 에 넣는다 — 임의의 (S, M, tau) 에서 평가 가능하다.

        입력
          curve : Tensor(n, 10) — 표준화된 제로커브 노드
          vc    : Tensor(n, nvc) — 표준화된 vol·corr 블록
          con   : Tensor(n, ncon) — 표준화된 계약 블록
          state : Tensor(n, 10) — 경로상태 (S1~3, M1~3, tau, KI, event_pre, obs_frac).
                  규약은 `module.pi_physics` 참고. 0 으로 마스킹하면 기능적으로 CurveOperatorV2 다
        출력
          Tensor(n,) — 표준화 공간의 가격
        """
        hc = self.cnn(curve.unsqueeze(1)).squeeze(-1)  # (n, 32)
        branch = self.branch(torch.cat([hc, vc], -1))  # (n, P)  시장상태
        return (branch * self.t(torch.cat([con, state], -1))).sum(-1) + self.b0


class CurveOperatorMoE(nn.Module):
    """arm A4b (XPINN 게이팅): CurveOperatorV2 공유 branch/trunk + K 전문가 헤드 + 배리어 게이트(soft routing).
    불연속(낙인 배리어)에서 영역 특화하되 전량 데이터로 학습(하드분할의 데이터굴주 회피). ib=계약 내 배리어 인덱스."""

    def __init__(self, nvc, ncon, P, K=3, ib=0):
        """공유 branch/trunk 위에 K 전문가 헤드와 배리어 기준 soft 게이트를 올린다.

        입력
          nvc  : int — vol·corr 블록 차원
          ncon : int — 계약 차원
          P    : int — 공유 상호작용 차원
          K    : int — 전문가 수
          ib   : int — 계약 블록 안의 배리어 열 인덱스. 게이트가 이 한 열만 본다
        출력
          없음 (파라미터 등록만 한다)
        """
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.branch = nn.Sequential(
            nn.Linear(32 + nvc, 128), nn.Tanh(), nn.Linear(128, P), nn.Tanh(), nn.Linear(P, P)
        )
        self.t = mlp(ncon, P)
        self.heads = nn.Linear(P, K)  # K 전문가 (공유 상호작용 z 위)
        self.b0 = nn.Parameter(torch.zeros(K))
        self.gate = nn.Sequential(
            nn.Linear(1, 16), nn.Tanh(), nn.Linear(16, K)
        )  # 배리어 기준 soft 게이트
        self.ib = ib

    def V(self, curve, vc, con):
        """전문가별 가격을 배리어 게이트로 가중평균한다.

        하드 분할이 아니라 soft routing 이다 — 전 데이터로 학습하므로 영역별 데이터 굶주림이
        생기지 않는다.

        입력
          curve : Tensor(n, 10) — 표준화된 제로커브 노드
          vc    : Tensor(n, nvc) — 표준화된 vol·corr 블록
          con   : Tensor(n, ncon) — 표준화된 계약 블록
        출력
          Tensor(n,) — 표준화 공간의 가격 (전문가 K 개의 게이트 가중합)
        """
        hc = self.cnn(curve.unsqueeze(1)).squeeze(-1)
        branch = self.branch(torch.cat([hc, vc], -1))
        z = branch * self.t(con)  # (n, P) 공유 상호작용
        experts = self.heads(z) + self.b0  # (n, K) 전문가별 가격
        g = torch.softmax(self.gate(con[:, self.ib : self.ib + 1]), -1)  # (n, K) 배리어 게이트
        return (g * experts).sum(-1)
