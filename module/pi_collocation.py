# -*- coding: utf-8 -*-
"""PI 앵커용 collocation 샘플러 — **fold train 계약만** 사용.

물리항은 라벨이 필요 없으므로 어디서든 뽑을 수 있는데, 그게 곧 누출 경로다.
그래서 (1) 학습 인덱스만 받고 (2) 매 스텝 assert_train_only 로 검증하며
(3) 샘플 범위(스팟·러닝최소 하한)도 fold train 계약의 행사가·배리어 분위수에서 도출한다.

4종을 분리해 뽑는다:
  sample_interior      PDE 잔차용 내부점 (이벤트 시각 근방은 회피 — 그 지점은 불연속)
  sample_event         관측일 pre/post 상태 쌍 (정규상환 / 리자드 / 연속 3모드)
  sample_terminal      만기 (tau=0)
  sample_ki_boundary   배리어 ±eps 양측 상태 쌍 (KI 지시자만 다른 쌍)
"""
import numpy as np
import torch

from .pi_physics import event_tau, feasible_ki_states


class PICollocationSampler:
    """D.CON 레이아웃 의존: [0:12]=strk, [12:24]=pmt, [24:36]=lz_barr, [36:48]=lz_pmt, D.iBARR=배리어."""

    def __init__(self, D, train_idx, seed=0):
        """학습 계약만으로 샘플 범위를 도출한다 — 범위 자체가 누출 경로이기 때문이다.

        스팟 상·하한과 러닝최소 하한을 **fold train 계약**의 행사가·배리어 분위수에서 뽑는다.
        학습분포 밖에서 물리를 강제하면 노이즈만 늘어난다.

        입력
          D         : Dataset — D.CON · D.ITEM · D.iBARR 를 쓴다
          train_idx : array-like[int] — 이 폴드의 학습 행. 비어 있으면 거부한다
          seed      : int — 추첨 시드
        출력
          없음 (spot_lo/spot_hi/run_lo 와 train_items 를 채운다)
        예외
          ValueError — train_idx 가 비었을 때
        """
        self.D = D
        self.train_idx = np.asarray(train_idx, dtype=int)
        if not len(self.train_idx):
            raise ValueError("collocation 학습 유니버스가 비었습니다")
        self.rng = np.random.default_rng(seed)
        con = D.CON[self.train_idx]
        ks = con[:, :12].ravel()
        ks = ks[(ks > 0.2) & (ks < 1.2)]
        bs = con[:, D.iBARR]
        bs = bs[(bs > 0.2) & (bs < 1.0)]
        # 계약 행사가/배리어가 실제로 사는 구간 + 여유. 학습분포 밖에서 물리를 강제하면 노이즈만 늘어난다.
        self.spot_lo = float(max(0.35, np.quantile(ks, 0.01) - 0.15))
        self.spot_hi = float(min(1.30, np.quantile(ks, 0.99) + 0.25))
        self.run_lo = float(max(0.25, np.quantile(bs, 0.01) - 0.12)) if len(bs) else 0.35
        self.train_items = set(D.ITEM[self.train_idx].astype(str))

    # ---- 내부 유틸 ----
    def _contracts(self, n):
        """학습 계약에서 n 개를 **복원추출**한다.

        입력
          n : int — 뽑을 개수
        출력
          ndarray(n,) int — 행 인덱스. 중복이 있을 수 있다
        """
        return self.train_idx[self.rng.integers(0, len(self.train_idx), n)]

    def _state(self, idx, spot, run, tau, ki, event_pre=0.0, frac=0.0):
        """(n, 10) 상태행렬을 조립한다. 열 규약은 `module.pi_physics` 머리말 참고.

        입력
          idx       : ndarray[int] — 길이로만 쓴다 (행 수를 맞추기 위한 자리)
          spot      : ndarray(n, 3) — 스팟/기준가
          run       : ndarray(n, 3) — 러닝 최소
          tau       : ndarray(n,) — 잔존만기 [년]
          ki        : ndarray(n,) — KI 지시자 (0/1)
          event_pre : float — 1=이벤트 직전, 0=직후. 전 행에 같은 값이 들어간다
          frac      : float | ndarray(n,) — 관측 진행률 (0=발행, 1=만기)
        출력
          ndarray(n, 10) float32
        """
        return np.c_[
            spot, run, tau, ki, np.full(len(idx), event_pre), np.asarray(frac) + np.zeros(len(idx))
        ].astype("float32")

    def _ki(self, idx, run):
        """러닝최소와 계약 배리어로 **물리적으로 가능한** KI 지시자를 만든다.

        배리어를 이미 깬 상태에 KI=0 을 붙이면 모순이므로 `feasible_ki_states` 로 교정한다.

        입력
          idx : ndarray[int] — 행 인덱스 (배리어를 여기서 읽는다)
          run : ndarray(n, 3) — 러닝 최소
        출력
          ndarray(n,) — KI 지시자 (0/1)
        """
        barr = self.D.CON[idx, self.D.iBARR]
        return feasible_ki_states(torch.tensor(run), torch.tensor(barr)).numpy()

    # ---- 4종 샘플러 ----
    def sample_interior(self, n, device):
        """PDE 잔차용 내부점. 관측일에서 만기의 2.5% 이상 떨어진 tau 만 채택한다.

        관측일은 가치가 점프하는 불연속점이라 PDE 가 성립하지 않는다. 재추첨은 최대 100회이고,
        그 안에 못 찾으면 **마지막 추첨값을 그대로 쓴다** (회차가 촘촘한 계약에서 일어난다).

        입력
          n      : int — 뽑을 점의 수
          device : str — torch 장치
        출력
          (idx, state) — idx   : ndarray(n,) int, 계약 행 인덱스
                         state : Tensor(n, 10), 상태행렬
        """
        idx = self._contracts(n)
        spot = self.rng.uniform(self.spot_lo, self.spot_hi, (n, 3)).astype("float32")
        run = np.minimum(spot, self.rng.uniform(self.run_lo, 1.0, (n, 3))).astype("float32")
        tau = np.empty(n, "float32")
        frac = np.empty(n, "float32")
        for q, i in enumerate(idx):
            nobs = int(self.D.ml.iloc[i].nobs)
            ten = float(self.D.TEN[i])
            events = ten - np.arange(1, nobs + 1) * ten / nobs
            for _ in range(100):  # 이벤트 근방이면 재추첨 (최대 100회)
                x = self.rng.uniform(0.03 * ten, 0.97 * ten)
                if np.min(np.abs(events - x)) > 0.025 * ten:
                    break
            tau[q] = x
            frac[q] = 1 - x / ten
        return idx, torch.tensor(
            self._state(idx, spot, run, tau, self._ki(idx, run), 0.0, frac), device=device
        )

    def sample_event(self, n, device):
        """관측일의 pre/post 상태 쌍을 뽑는다 — 세 모드를 균등 추첨한다.

        mode 0=정규상환(worst ≥ K), 1=리자드, 2=연속(둘 다 실패).
        리자드 모드는 **실현 가능한 계약에서만** 성립한다 (배리어가 1.15 미만이고 행사가와
        0.03 이상 떨어진 경우). 안 되면 조용히 모드 2 로 떨어진다.

        입력
          n      : int — 뽑을 쌍의 수
          device : str — torch 장치
        출력
          (idx, j, pre, post) — idx  : ndarray(n,) int, 계약 행 인덱스
                                j    : ndarray(n,) int, 관측 회차 (0-based)
                                pre  : Tensor(n, 10), event_pre=1
                                post : Tensor(n, 10), event_pre=0.
                                       두 상태는 event_pre 열만 다르다
        """
        idx = self._contracts(n)
        nobs = self.D.ml.iloc[idx].nobs.to_numpy(int)
        j = np.array([self.rng.integers(0, x) for x in nobs])
        K = self.D.CON[idx, j]
        lzb = self.D.CON[idx, 24 + j]
        mode = self.rng.integers(0, 3, n)
        spot = np.empty((n, 3), "float32")
        run = np.empty((n, 3), "float32")
        for q in range(n):
            if mode[q] == 0:  # 정규 상환: worst >= K
                spot[q] = self.rng.uniform(K[q] + 0.005, K[q] + 0.12, 3)
                run[q] = np.minimum(spot[q], self.rng.uniform(self.run_lo, 1.0, 3))
            elif (
                mode[q] == 1 and lzb[q] < 1.15 and K[q] > lzb[q] + 0.03
            ):  # 리자드가 실현 가능한 계약만
                lo = max(lzb[q] + 0.005, self.spot_lo)
                hi = K[q] - 0.005
                spot[q] = self.rng.uniform(lo, hi, 3)
                spot[q, 0] = K[q] - 0.01
                run[q] = self.rng.uniform(lzb[q] + 0.002, np.minimum(spot[q], 1.0))
            else:  # 연속: 정규·리자드 모두 실패
                spot[q] = self.rng.uniform(self.spot_lo, max(self.spot_lo + 0.01, K[q] - 0.01), 3)
                spot[q, 0] = K[q] - 0.03
                run[q] = np.minimum(spot[q], self.rng.uniform(self.run_lo, 1.0, 3))
                run[q, 0] = min(run[q, 0], spot[q, 0], lzb[q] - 0.01)
        ki = self._ki(idx, run)
        tau = event_tau(torch.tensor(self.D.TEN[idx]), torch.tensor(nobs), torch.tensor(j)).numpy()
        frac = (j + 1) / nobs
        pre = self._state(idx, spot, run, tau, ki, 1.0, frac)
        post = self._state(idx, spot, run, tau, ki, 0.0, frac)
        return idx, j, torch.tensor(pre, device=device), torch.tensor(post, device=device)

    def sample_terminal(self, n, device):
        """만기(tau=0) 경계점을 뽑는다.

        입력
          n      : int — 뽑을 점의 수
          device : str — torch 장치
        출력
          (idx, state) — idx   : ndarray(n,) int, 계약 행 인덱스
                         state : Tensor(n, 10), tau=0 · 진행률 1.0
        """
        idx = self._contracts(n)
        spot = self.rng.uniform(self.spot_lo, self.spot_hi, (n, 3)).astype("float32")
        run = np.minimum(spot, self.rng.uniform(self.run_lo, 1.0, (n, 3))).astype("float32")
        state = self._state(idx, spot, run, np.zeros(n), self._ki(idx, run), 0.0, 1.0)
        return idx, torch.tensor(state, device=device)

    def sample_ki_boundary(self, n, device, eps=0.002):
        """배리어 바로 위(KI=0) vs 바로 아래(KI=1) 상태 쌍을 뽑는다.

        낙인 계약(B<1) 만 대상이다 — 노낙인은 KI 전이가 없다. 그래서 **반환 길이가 n 보다
        작을 수 있다** (요청한 n 개 중 낙인 계약만 남는다).

        입력
          n      : int — 추첨 개수 (필터 전)
          device : str — torch 장치
          eps    : float — 배리어에서 띄울 거리
        출력
          (idx, s0, s1) — idx : ndarray(m,) int, m ≤ n
                          s0  : Tensor(m, 10), 러닝최소 = B+eps, KI=0
                          s1  : Tensor(m, 10), 러닝최소 = B−eps, KI=1
        """
        idx = self._contracts(n)
        b = self.D.CON[idx, self.D.iBARR]
        keep = b < 1.0
        idx = idx[keep]
        b = b[keep]
        n = len(idx)
        spot = np.maximum(b[:, None] + 0.05, self.rng.uniform(0.75, 1.1, (n, 3))).astype("float32")
        above = np.repeat((b + eps)[:, None], 3, 1).astype("float32")
        below = np.repeat((b - eps)[:, None], 3, 1).astype("float32")
        tau = (self.rng.uniform(0.1, 0.9, n) * self.D.TEN[idx]).astype("float32")
        frac = 1 - tau / self.D.TEN[idx]
        s0 = self._state(idx, spot, above, tau, np.zeros(n), 0.0, frac)
        s1 = self._state(idx, spot, below, tau, np.ones(n), 0.0, frac)
        return idx, torch.tensor(s0, device=device), torch.tensor(s1, device=device)

    def assert_train_only(self, idx):
        """collocation 이 비학습 계약을 건드리면 즉시 실패시킨다 (누출 방지 가드).

        물리항은 라벨이 필요 없어 어디서든 뽑을 수 있고, 그게 곧 누출 경로다 — 매 스텝 부른다.

        입력
          idx : array-like[int] — 이번 스텝이 쓴 계약 행 인덱스
        출력
          없음 (통과하면 아무 일도 하지 않는다)
        예외
          AssertionError — 학습 유니버스 밖 계약이 하나라도 섞였을 때
        """
        if not set(self.D.ITEM[np.asarray(idx)].astype(str)).issubset(self.train_items):
            raise AssertionError("collocation 이 학습 외 계약을 사용했습니다")
