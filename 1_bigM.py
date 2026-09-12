import numpy as np

def big_m_method(c, A, b, constraint_types, maximize=True, M=10000.0):
    """
    Uses: 
        c: List or array of objective function coefficients.
        A: 2D List or array of constraint coefficients (LHS).
        b: List or array of constraint bounds (RHS).
        constraint_types: List of strings ('<=', '>=', '=').
        maximize: Boolean, True for maximization, False for minimization.
        M: Large positive penalty value for artificial variables.
    Returns:
        solution: Array containing the optimal values of the decision variables.
        optimal_z: The optimal objective function value.
    """
    c = np.array(c, dtype=float)
    A = np.array(A, dtype=float)
    b = np.array(b, dtype=float)
    
    num_vars = len(c)
    num_constraints = len(b)
    
    if not maximize:
        c = -c
        
    num_slacks = sum(1 for t in constraint_types if t in ['<=', '>='])
    num_artificials = sum(1 for t in constraint_types if t in ['>=', '='])
    
    total_vars = num_vars + num_slacks + num_artificials
    
    tableau = np.zeros((num_constraints + 1, total_vars + 1))
    
    slack_idx = num_vars
    art_idx = num_vars + num_slacks
    art_rows = []
    
    for i in range(num_constraints):
        tableau[i, :num_vars] = A[i]
        tableau[i, -1] = b[i]
        
        if constraint_types[i] == '<=':
            tableau[i, slack_idx] = 1
            slack_idx += 1
        elif constraint_types[i] == '>=':
            tableau[i, slack_idx] = -1
            slack_idx += 1
            tableau[i, art_idx] = 1
            art_rows.append((i, art_idx))
            art_idx += 1
        elif constraint_types[i] == '=':
            tableau[i, art_idx] = 1
            art_rows.append((i, art_idx))
            art_idx += 1

    tableau[-1, :num_vars] = -c
    
    for r, col in art_rows:
        tableau[-1, col] = M
        
    for r, col in art_rows:
        tableau[-1, :] -= M * tableau[r, :]
        
    # Simplex Iterations
    iteration = 0
    while True:
        z_row = tableau[-1, :-1]
        
        if np.all(z_row >= -1e-7):
            break 
            
        entering_col = np.argmin(z_row)
        
        ratios = []
        for i in range(num_constraints):
            if tableau[i, entering_col] > 1e-7:
                ratios.append(tableau[i, -1] / tableau[i, entering_col])
            else:
                ratios.append(np.inf)
                
        if np.all(np.array(ratios) == np.inf):
            raise ValueError("Problem is unbounded. No optimal solution exists.")
            
        leaving_row = np.argmin(ratios)
        
        pivot_val = tableau[leaving_row, entering_col]
        tableau[leaving_row, :] /= pivot_val
        
        for i in range(num_constraints + 1):
            if i != leaving_row:
                factor = tableau[i, entering_col]
                tableau[i, :] -= factor * tableau[leaving_row, :]
                
        iteration += 1

    solution = np.zeros(num_vars)
    for j in range(num_vars):
        col = tableau[:-1, j]
        if np.sum(np.isclose(col, 1)) == 1 and np.sum(np.isclose(col, 0)) == num_constraints - 1:
            row_idx = np.where(np.isclose(col, 1))[0][0]
            solution[j] = tableau[row_idx, -1]
            
    art_start_col = num_vars + num_slacks
    for col_idx in range(art_start_col, total_vars):
        col = tableau[:-1, col_idx]
        if np.sum(np.isclose(col, 1)) == 1 and np.sum(np.isclose(col, 0)) == num_constraints - 1:
            row_idx = np.where(np.isclose(col, 1))[0][0]
            if tableau[row_idx, -1] > 1e-7:
                raise ValueError("Problem is infeasible. An artificial variable remains in the basis.")

    optimal_z = tableau[-1, -1]
    if not maximize:
        optimal_z = -optimal_z
        
    return solution, optimal_z

if __name__ == "__main__":
    # Example Problem:
    # Minimize Z = 4x_1 + x_2
    # Subject to:
    #   3x_1 + x_2 = 3
    #   4x_1 + 3x_2 >= 6
    #   x_1 + 2x_2 <= 4
    #   x_1, x_2 >= 0
    
    c = [4, 1]
    A = [
        [3, 1],
        [4, 3],
        [1, 2]
    ]
    b = [3, 6, 4]
    constraint_types = ['=', '>=', '<=']
    
    try:
        sol, z_opt = big_m_method(c, A, b, constraint_types, maximize=False)
        print("Optimal Decision Variables (x):", sol)
        print("Optimal Objective Value (Z):", z_opt)
    except ValueError as e:
        print("Error:", e)


"""
Output:

Optimal Decision Variables (x): [0.4 1.8]
Optimal Objective Value (Z): 3.400000000003274
"""