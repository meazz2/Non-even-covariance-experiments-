import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from numpy.polynomial import chebyshev
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm

_EPS = 1e-12
d = 2 
p = 1 

# ============================================================
# PSD and Covariance Functions
# ============================================================
def make_psd(C, eps=1e-7): # Increased eps for better stability
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, eps)
    return (V * w) @ V.T

def G_even(h, a):
    return np.exp(-(h @ h) / (a * a + _EPS))

def G_odd(h, a, u0):
    hnorm = (h @ h) ** 0.5
    if hnorm < 1e-10: return 0
    Tp = chebyshev.Chebyshev.basis(p)
    return ((-1) ** ((p - 1) / 2)) * (a ** (1 - p)) * Tp((h @ u0) / hnorm) * np.exp(
        -(h @ h) / (2.0 * a + _EPS)
    )

def C2_block(h, a11, a22, a12, rho, s1, s2, u0):
    C11 = s1 * s1 * G_even(h, a11)
    C22 = s2 * s2 * G_even(h, a22)
    odd_part = G_odd(h, a12, u0)
    C12 = rho * s1 * s2 * odd_part
    C21 = -rho * s1 * s2 * odd_part
    return np.array([[C11, C12], [C21, C22]])

def build_cov_C2(loc, a11, a22, a12, rho, s1, s2, u0):
    N = len(loc)
    C = np.zeros((2 * N, 2 * N))
    for i in range(N):
        for j in range(N):
            h = loc[j] - loc[i]
            C[2 * i:2 * i + 2, 2 * j:2 * j + 2] = C2_block(
                h, a11, a22, a12, rho, s1, s2, u0
            )
    # Add a small nugget for numerical stability
    return make_psd(C) + np.eye(2*N) * 1e-8

def total_nll_full_params(params, loc, y_list, s1, s2, u0):
    a11, a22, a12, rho = params
    if a11 <= 0.1 or a22 <= 0.1 or a12 <= 0.1 or abs(rho) > 0.99:
        return 1e20
    C = build_cov_C2(loc, a11, a22, a12, rho, s1, s2, u0)
    try:
        cF = cho_factor(C, check_finite=False)
        logdet = 2.0 * np.sum(np.log(np.diag(cF[0])))
        quad_sum = sum(float(y @ cho_solve(cF, y, check_finite=False)) for y in y_list)
        return 0.5 * (quad_sum + len(y_list) * logdet + len(y_list) * C.shape[0] * np.log(2.0 * np.pi))
    except:
        return 1e20

# ============================================================
# Worker Function
# ============================================================
def run_single_iteration(iteration_id, loc, L_true, true_params, s1, s2, u0, replicates):
    np.random.seed(iteration_id) # Important for multiprocessing randomness
    y_sim = [L_true @ np.random.normal(size=L_true.shape[0]) for _ in range(replicates)]
    random_start = [p * np.random.uniform(0.9, 1.1) for p in true_params[:]]
    
    res = minimize(
        total_nll_full_params,
        x0=random_start,
        args=(loc, y_sim, s1, s2, u0),
        bounds=[(0.5, 10), (0.5, 10), (0.5, 15), (-0.95, 0.95)],
        method="L-BFGS-B",
        options={'maxiter': 150}
    )
    if not res.success:
        print(f"Iteration {iteration_id} failed: {res.message}")
    return res.x if res.success else None

# ============================================================
# Main Execution
# ============================================================
if __name__ == "__main__":
    TRUE_PARAMS = [3.0, 3.0, 4.0, 0.6] 
    S1, S2, U0 = 1.0, 1.0, np.array([1.0, 0.0])
    N_ITERATIONS, REPLICATES = 50, 20

    grid_side = np.linspace(0, 10, 8)
    x, y = np.meshgrid(grid_side, grid_side)
    loc = np.column_stack([x.ravel(), y.ravel()])

    C_true = build_cov_C2(loc, *TRUE_PARAMS, S1, S2, U0)
    L_true = np.linalg.cholesky(C_true)

    results = []
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(run_single_iteration, i, loc, L_true, TRUE_PARAMS, S1, S2, U0, REPLICATES): i for i in range(N_ITERATIONS)}
        for future in tqdm(as_completed(futures), total=N_ITERATIONS, desc="Processing"):
            res = future.result()
            if res is not None: results.append(res)

    df_results = pd.DataFrame(results, columns=['a11', 'a22', 'a12', 'rho'])
    df_results.to_csv("simulation_results_50_rows_g1.csv", index=False)

    # Independent Plotting
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    for i, col in enumerate(df_results.columns):
        col_data = df_results[col]
        Q1, Q3 = col_data.quantile(0.25), col_data.quantile(0.75)
        IQR = Q3 - Q1
        filtered = col_data[(col_data >= Q1 - 1.5*IQR) & (col_data <= Q3 + 1.5*IQR)]
        
        axes[i].boxplot(filtered)
        axes[i].axhline(TRUE_PARAMS[i], color='red', linestyle='--')
        axes[i].set_xticks([1])
        axes[i].set_xticklabels([col])
        axes[i].set_title(f"Mean: {filtered.mean():.2f}")

    plt.tight_layout()
    plt.show()
    print(f"Total rows saved: {len(results)}")