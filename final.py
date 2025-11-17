import numpy as np
##initialization=============================================================
# 1) Source
X = [0, 1]
pX = [0.5, 0.5]

# 2) Distortion target
epsilon = 0.5
D = 0.3 /(1+epsilon)

# 3) Test channel p(hatX|X)
pX_hat_given_X = [
    [1 - D, D],   # X=0 → hatX=0/1
    [D, 1 - D]    # X=1 → hatX=0/1
]

# 4) Marginal p(hatX)
X_hat = [0, 1]
pX_hat = [0.5, 0.5]

# 5) Joint distribution p(X, hatX)
pXY = [
    [0.5 * (1 - D), 0.5 * D],
    [0.5 * D,       0.5 * (1 - D)]
]

print("Distortion D =", D)
print("Source distribution pX =", pX)
print("Reconstruction distribution pX_hat =", pX_hat)
print("Test channel pX_hat_given_X =\n", np.array(pX_hat_given_X))
print("Joint distribution pXY =\n", np.array(pXY))
##initialization=============================================================

#function definition=============================================================

def Hamming_distortion(X, Xhat):
    assert X.shape == Xhat.shape, "X와 Xhat shape이 다름"
    mism = (X != Xhat)
    cnt = np.count_nonzero(mism)
    return cnt

import numpy as np

def random_bit(size=None):
    return np.random.randint(0, 2, size=size)

print("Example random bit (scalar) =", random_bit())          # 단일 0 또는 1
print("Example random bit array (size 10) =", random_bit(10)) # [0 1 0 1 1 0 0 1 0 1]

import math
def mutual_information(X, Xhat):
    hap = 0
    for x in X:
        for x_hat in X_hat:
            hap += (pX[x] * pX_hat_given_X[x_hat][x] * 
                    math.log2(pX[x] * pX_hat_given_X[x_hat][x]/(pX[x]*
                                                                pX_hat[x_hat]))
                   )
    return hap        

def D_entropy(D):
    return -D*math.log2(D) -(1-D)*math.log2(1-D)

print("Mutual information I(X; X_hat) =", mutual_information(X, X_hat))
print("Rate bound 1 - H(D) =", 1-D_entropy(D))


def codebook_generation(n, R, distribution):
    M = int(2 ** (n*R))
    codebook = np.random.choice([0, 1], size=(M, n), p=distribution)
    return codebook

n=80
R=mutual_information(X, X_hat) + 0.01
codebook = codebook_generation(n, R, pX_hat)
codebook.shape
np.sum(codebook, axis = 1).shape

def joint_empirical_pmf(x_seq: np.ndarray, y_seq: np.ndarray, X_alphabet, Y_alphabet):
    x_seq = np.asarray(x_seq)                     # 예: array([0,1,1,0])
    y_seq = np.asarray(y_seq)                     # 예: array([0,1,1,0])
    assert x_seq.shape == y_seq.shape and x_seq.ndim == 1

    # map symbols to indices
    x_to_idx = {sym:i for i, sym in enumerate(X_alphabet)}   # {0:0, 1:1}
    y_to_idx = {sym:j for j, sym in enumerate(Y_alphabet)}   # {0:0, 1:1}
    kx, ky = len(X_alphabet), len(Y_alphabet)                # kx=2, ky=2

    # encode to indices
    xi = np.vectorize(x_to_idx.__getitem__)(x_seq)           # [0,1,1,0]
    yj = np.vectorize(y_to_idx.__getitem__)(y_seq)           # [0,1,1,0]
    # joint counts
    idx = xi * ky + yj                                       
    # ky=2 이므로 xi*2 + yj → [0*2+0, 1*2+1, 1*2+1, 0*2+0] = [0,3,3,0]

    counts = np.bincount(idx, minlength=kx*ky).astype(float).reshape(kx, ky)
    # idx=[0,3,3,0] 이므로 bincount: [2,0,0,2] → reshape(2x2)=[[2,0],[0,2]]
    return counts / x_seq.size 
    # x_seq.size = 4 → [[2/4, 0/4],[0/4, 2/4]]
    # 결과: array([[0.5, 0.0], [0.0, 0.5]])

def is_jointly_typical(x_seq,
                       y_seq,
                       pXY,
                       X_alphabet,
                       Y_alphabet,
                       eps: float):
    """
    (x^n, y^n)이 주어졌을 때,
    정의:
      |π(x,y) - pXY(x,y)| <= eps * pXY(x,y)  (pXY(x,y) > 0인 모든 (x,y))
    이고,
      pXY(x,y) = 0 인 자리에서는 π(x,y)도 0 이어야 함
    을 만족하면 True (joint ε-typical), 아니면 False.
    """
    # 경험적 joint pmf π(x,y)
    pi = joint_empirical_pmf(x_seq, y_seq, X_alphabet, Y_alphabet)

    # 이론 joint pmf
    pXY = np.asarray(pXY, dtype=float)
    assert pi.shape == pXY.shape, "π와 pXY의 shape이 다릅니다."

    # (1) pXY(x,y) > 0 인 위치: 상대 오차 조건 검사
    pos = pXY > 0
    if np.any(np.abs(pi[pos] - pXY[pos]) > eps * pXY[pos]):
        return False

    # (2) pXY(x,y) = 0 인 위치: 실제로도 안 나와야 함
    zero = ~pos
    if np.any(pi[zero] > 0):
        return False

    return True

    
#function definition=============================================================

## Distortion =============================================================

# ----- 3. 샘플 수 -----
N = 1000000

# ----- 4. X 샘플링 -----
X_samples = np.random.choice(X, size=N, p=pX)

# ----- 5. 조건부 분포에 따라 X_hat 샘플링 -----
Xhat_samples = np.empty(N, dtype=int)
for i, x in enumerate(X_samples):
    Xhat_samples[i] = np.random.choice(X_hat, p=pX_hat_given_X[x])

# ----- 6. Joint 분포 추정 -----
joint_counts = np.zeros((2, 2))
for x, xhat in zip(X_samples, Xhat_samples):
    joint_counts[x, xhat] += 1

joint_probs = joint_counts / N

# ----- 7. 평균 왜곡 측정 -----
distortion = np.mean(X_samples != Xhat_samples)

print("Target distortion D =", D)
print("Conditional probability p(X_hat | X):")
print(pX_hat_given_X)
print("\nEmpirical joint probability p(X, X_hat):")
print(joint_probs)
print(f"\nAverage Hamming distortion over {N} samples ~= {distortion:.4f}")
## Distortion =============================================================




count = 1000 
zero_cnt = 0
block_distortions = []
for i in range(count):
    xn = random_bit(n)  # source block X^n
    found_index = None
    for idx, tempt in enumerate(codebook):   # tempt = codeword candidate
        if is_jointly_typical(xn, tempt, pXY, X, X_hat, eps=epsilon):
            found_index = idx
            break
    chosen = codebook[found_index]
    # print("source = " , xn)
    # print("codeword = ", chosen)
    dist = Hamming_distortion(xn, chosen)
    block_distortions.append(dist)
    if found_index is None:
        found_index = 0  # 못 찾았을 때 기본값 (논문에서 m=1)
        zero_cnt += 1
    #print(f"Block {i}: chosen codeword index = {found_index}")
print("Typicality threshold epsilon =", epsilon, "| Number of blocks with no typical codeword (error count) =", zero_cnt)
print(f"- Mean distortion per block (raw count/100) over {count} blocks = {np.mean(block_distortions)/n:.4f}")