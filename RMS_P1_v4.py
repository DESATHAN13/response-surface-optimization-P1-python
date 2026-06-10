import os
import itertools
import numpy as np
import pandas as pd
from scipy.linalg import hadamard as scipy_hadamard
import generators

def get_robust_hadamard(n):
    if (n & (n - 1)) == 0 and n > 0: return scipy_hadamard(n)
    if n == 12: return np.kron(np.array([[1,1,1],[1,-1,1],[1,1,-1]]), scipy_hadamard(4))
    elif n == 20: return np.kron(np.ones((5,5)) - 2*np.eye(5), scipy_hadamard(4))
    elif n in [24, 40]: return np.kron(get_robust_hadamard(n // 2), scipy_hadamard(2))
    elif n == 28: return np.kron(np.ones((7,7)) - 2*np.eye(7), scipy_hadamard(4))
    else: raise ValueError(f"Hadamard framework for order {n} not found.")

def MakeModelMatrix(X):
    n, m = X.shape
    D = np.hstack((X, X * X))
    interactions = [X[:, i] * X[:, j] for i in range(m - 1) for j in range(i + 1, m)]
    if interactions: D = np.hstack((D, np.column_stack(interactions)))
    return np.hstack((np.ones((n, 1)), D))

def WriteDesignToFile(filename, D, Qstar, D_value):
    with open(filename, 'a') as f:
        f.write(f"Qstar: {Qstar:.6f} | D-value: {D_value:.10f}\n")
        np.savetxt(f, D, fmt='%+.2f')
        f.write("="*40 + "\n")

def process_results(results, target_dir, f, n, h, cm, od, best_designs):
    if not results:
        print("No designs met the criteria.")
        return
    df = pd.DataFrame(results)
    df = df.sort_values('D-value', ascending=False).drop_duplicates('Type')
    print("\n--- Optimization Results ---")
    print(df.to_string(index=False))
    
    choice = input("\nDo you want to create the Excel report and save the text files? (y/n): ").lower().strip()
    if choice == 'y':
        filename = f"F{f}_N{n}_H{h}_CM{cm}_OD{od}_Report.xlsx"
        df.to_excel(os.path.join(target_dir, filename), index=False)
        print(f"Excel report generated: {filename}")
        for design_type, data in best_designs.items():
            WriteDesignToFile(os.path.join(target_dir, f"{design_type}.txt"), data['D'], data['Qstar'], data['Dval'])
        print("Text files (Evang.txt, Nguyen.txt) saved.")
    else:
        print("Files were not saved.")

def Construction3g2(Xf, XA, nc, s, fname, Qbest, Dbest, dynamic_type, critical_D, results_list, m, h, cm, od, best_designs):
    n_runs, m_factors = XA.shape
    C = np.zeros((nc, s))
    if s <= m_factors:
        for idx in itertools.combinations(range(m_factors), s):
            T = XA[:, list(idx)]
            D = np.vstack((Xf, T, C, -T))
            Xm = MakeModelMatrix(D)
            nn, p = Xm.shape
            try: det_val = np.abs(np.linalg.det(Xm.T @ Xm))
            except np.linalg.LinAlgError: det_val = 0.0
            D_value = ((10**3) * (det_val**(1/p))) / nn if dynamic_type == 2 else (10**9) * det_val / (nn**p)
            Qstar = 0.85 # Placeholder
            if ((Qstar > Qbest) or (D_value > Dbest)) and (D_value > critical_D):
                if Qstar > Qbest: Qbest = Qstar
                if D_value > Dbest:
                    Dbest = D_value
                    design_type = 'Evang' if 'Evang' in fname else 'Nguyen'
                    best_designs[design_type] = {'D': D, 'Qstar': Qstar, 'Dval': D_value}
                    results_list.append({'Type': design_type, 'F': m, 'N': D.shape[0], 'H': h, 'CM': cm, 'OD': od, 'D-value': D_value, 'Qstar': Qstar})
    return [Qbest, Dbest]

def main():
    target_dir = r"C:\Users\zacde\Desktop\Des\PYTHON\P1"
    results_data, best_designs = [], {}
    print("==================================================\n      RSM METRIC MATRIX DESIGN CONFIGURATOR       \n==================================================")
    m = int(input("\n[1/4] Enter number of factors (m): "))
    print("\n[2/4] Options: 8, 12, 16, 20, 24, 32, 40")
    h_order = int(input("Hadamard Order choice: "))
    print("\n[3/4] Options: 0 (None), 6, 8, 10, 12")
    conf_order = int(input("Conference Matrix choice: "))
    print("\n[4/4] Options: 2, 4, 6, 8, 10, 12, 16, 20, 22")
    od_choice = int(input("OD Type choice: "))
    od_variant = 'a'
    if od_choice in [6, 12]: od_variant = input(f"Select sub-variant for X{od_choice} ('a', 'b', or 'c'): ").lower().strip()

    Xf = get_robust_hadamard(h_order)[:, 1:m+1]
    if conf_order > 0:
        C = generators.get_conference_matrix(conf_order)
        Xf = np.vstack((Xf, np.vstack((C, -C))[:, :m]))
    
    Qbest, Dbest1, Dbest2 = -1.0, -1.0, -1.0
    print("\nRunning optimization loops...")
    for p_set in itertools.product([0, 1], repeat=11):
        x1, x2, x3, x4, x5, x6, x7, x8, xb, x12_val, xa = p_set
        if od_choice == 2: XT = generators.X2(x1, x2)
        elif od_choice == 4: XT = generators.X4(x1, x2, x3, x4)
        elif od_choice == 6: XT = (generators.X6b(xa, xb) if od_variant == 'b' else generators.X6c(x1, x2) if od_variant == 'c' else generators.X6a(xa, xb))
        elif od_choice == 8: XT = generators.X8(x1, x2, x3, x4, x5, x6, x7, x8)
        elif od_choice == 12: XT = (generators.X12b(x1, x2, x3, x4) if od_variant == 'b' else generators.X12c(x1, x2, x3, x4) if od_variant == 'c' else generators.X12a(x1, x2, x3, x4))
        elif od_choice == 16: XT = generators.X16(x1, x2, x3, x4, x5, x6, x7, x8)
        elif od_choice == 20: XT = generators.X20(x1, x2, x3, x4)
        elif od_choice == 22: XT = np.block([[generators.X20(x1, x2, x3, x4), np.zeros((20, 2))], [np.zeros((2, 20)), generators.X2(xb, x12_val)]])
        else: XT = generators.X2(x1, x2)

        XA = XT[:, :m]
        res1 = Construction3g2(Xf, XA, 0, m, "Evang.txt", Qbest, Dbest1, 1, 10e-300, results_data, m, h_order, conf_order, od_choice, best_designs)
        Qbest, Dbest1 = res1[0], res1[1]
        res2 = Construction3g2(Xf, XA, 0, m, "Nguyen.txt", Qbest, Dbest2, 2, 10e-300, results_data, m, h_order, conf_order, od_choice, best_designs)
        Qbest, Dbest2 = res2[0], res2[1]

    N_count = np.vstack((Xf, XA, -XA)).shape[0]
    process_results(results_data, target_dir, m, N_count, h_order, conf_order, od_choice, best_designs)

if __name__ == "__main__":
    main()