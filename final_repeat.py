import numpy as np
import math

## initialization (epsilon에 독립적인 부분) ================================
# 1) Source
X = [0, 1]
pX = [0.5, 0.5]

# 4) Marginal p(hatX)
X_hat = [0, 1]
pX_hat = [0.5, 0.5]

# 블록 길이, 샘플 수, 블록 개수
n = 80
N = 1000000   # for empirical distortion estimation
count = 1000  # number of blocks for typicality-based coding experiment

## function definition =====================================================

def Hamming_distortion(X_seq, Xhat_seq):
    assert X_seq.shape == Xhat_seq.shape, "X와 Xhat shape이 다름"
    mism = (X_seq != Xhat_seq)
    cnt = np.count_nonzero(mism)
    return cnt

def random_bit(size=None):
    return np.random.randint(0, 2, size=size)

def mutual_information(X, Xhat):
    hap = 0
    for x in X:
        for x_hat in X_hat:
            hap += (pX[x] * pX_hat_given_X[x_hat][x] * 
                    math.log2(
                        pX[x] * pX_hat_given_X[x_hat][x] / (pX[x] * pX_hat[x_hat])
                    )
                   )
    return hap        

def D_entropy(D):
    return -D*math.log2(D) -(1-D)*math.log2(1-D)

def codebook_generation(n, R, distribution):
    M = int(2 ** (n*R))
    codebook = np.random.choice([0, 1], size=(M, n), p=distribution)
    return codebook

def joint_empirical_pmf(x_seq: np.ndarray, y_seq: np.ndarray, X_alphabet, Y_alphabet):
    x_seq = np.asarray(x_seq)
    y_seq = np.asarray(y_seq)
    assert x_seq.shape == y_seq.shape and x_seq.ndim == 1

    # map symbols to indices
    x_to_idx = {sym:i for i, sym in enumerate(X_alphabet)}
    y_to_idx = {sym:j for j, sym in enumerate(Y_alphabet)}
    kx, ky = len(X_alphabet), len(Y_alphabet)

    # encode to indices
    xi = np.vectorize(x_to_idx.__getitem__)(x_seq)
    yj = np.vectorize(y_to_idx.__getitem__)(y_seq)

    # joint counts
    idx = xi * ky + yj
    counts = np.bincount(idx, minlength=kx*ky).astype(float).reshape(kx, ky)
    return counts / x_seq.size

def is_jointly_typical(x_seq,
                       y_seq,
                       pXY,
                       X_alphabet,
                       Y_alphabet,
                       eps: float):
    """
    (x^n, y^n)이 주어졌을 때,
      |π(x,y) - pXY(x,y)| <= eps * pXY(x,y)  (pXY(x,y) > 0인 모든 (x,y))
    이고,
      pXY(x,y) = 0 인 자리에서는 π(x,y)도 0 이어야 함
    을 만족하면 True (joint ε-typical), 아니면 False.
    """
    # empirical joint pmf π(x,y)
    pi = joint_empirical_pmf(x_seq, y_seq, X_alphabet, Y_alphabet)

    # theoretical joint pmf
    pXY = np.asarray(pXY, dtype=float)
    assert pi.shape == pXY.shape, "π와 pXY의 shape이 다릅니다."

    # (1) pXY(x,y) > 0: check relative error
    pos = pXY > 0
    if np.any(np.abs(pi[pos] - pXY[pos]) > eps * pXY[pos]):
        return False

    # (2) pXY(x,y) = 0: empirical prob must be zero as well
    zero = ~pos
    if np.any(pi[zero] > 0):
        return False

    return True

## simple sanity print (한 번만) ==========================================
print("Example random bit (scalar) =", random_bit())
print("Example random bit array (size 10) =", random_bit(10))

## epsilon sweep ===========================================================
# epsilon 값을 여기서 천천히 줄여가며 지정하면 됨
epsilon_list = [0.5, 0.4, 0.3, 0.2, 0.1]

for epsilon in epsilon_list:
    print("\n" + "="*70)
    print(f"Running experiment with typicality epsilon = {epsilon}")
    print("="*70)

    # 2) Distortion target
    D = 0.3 / (1 + epsilon)

    # 3) Test channel p(hatX|X)
    pX_hat_given_X = [
        [1 - D, D],   # X=0 → hatX=0/1
        [D, 1 - D]    # X=1 → hatX=0/1
    ]

    # 5) Joint distribution p(X, hatX)
    pXY = [
        [0.5 * (1 - D), 0.5 * D],
        [0.5 * D,       0.5 * (1 - D)]
    ]

    # --- initialization prints -------------------------------------------
    print("Distortion D =", D)
    print("Source distribution pX =", pX)
    print("Reconstruction distribution pX_hat =", pX_hat)
    print("Test channel pX_hat_given_X =\n", np.array(pX_hat_given_X))
    print("Joint distribution pXY =\n", np.array(pXY))

    # --- mutual information & rate bound ---------------------------------
    I = mutual_information(X, X_hat)
    print("Mutual information I(X; X_hat) =", I)
    print("Rate bound 1 - H(D) =", 1 - D_entropy(D))

    # --- codebook generation ---------------------------------------------
    R = I + 0.01
    codebook = codebook_generation(n, R, pX_hat)

    ## Distortion (single-letter experiment) ==============================
    # 4) X sampling
    X_samples = np.random.choice(X, size=N, p=pX)

    # 5) conditional sampling of X_hat given X
    Xhat_samples = np.empty(N, dtype=int)
    for i, x in enumerate(X_samples):
        Xhat_samples[i] = np.random.choice(X_hat, p=pX_hat_given_X[x])

    # 6) joint empirical distribution
    joint_counts = np.zeros((2, 2))
    for x, xhat in zip(X_samples, Xhat_samples):
        joint_counts[x, xhat] += 1
    joint_probs = joint_counts / N

    # 7) average Hamming distortion
    distortion = np.mean(X_samples != Xhat_samples)

    print("Target distortion D =", D)
    print("Conditional probability p(X_hat | X):")
    print(pX_hat_given_X)
    print("\nEmpirical joint probability p(X, X_hat):")
    print(joint_probs)
    print(f"\nAverage Hamming distortion over {N} samples ~= {distortion:.4f}")

    ## Block coding & typicality experiment ===============================
    zero_cnt = 0
    block_distortions = []
    for i in range(count):
        xn = random_bit(n)  # source block X^n
        found_index = None
        for idx, tempt in enumerate(codebook):   # tempt = codeword candidate
            if is_jointly_typical(xn, tempt, pXY, X, X_hat, eps=epsilon):
                found_index = idx
                break

        # if no typical codeword is found, use default index 0 (as in the paper)
        if found_index is None:
            found_index = 0
            zero_cnt += 1

        chosen = codebook[found_index]
        dist = Hamming_distortion(xn, chosen)
        block_distortions.append(dist)

        #print(f"Block {i}: chosen codeword index = {found_index}")

    print("Typicality threshold epsilon =", epsilon,
          "| Number of blocks with no typical codeword (error count) =", zero_cnt)
    print(f"- Mean distortion per block (normalized by block length n={n}) "
          f"over {count} blocks = {np.mean(block_distortions)/n:.4f}")
