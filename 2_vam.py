import numpy as np

def vogels_approximation_method(costs, supply, demand):
    """
    Solves the transportation problem using Vogel's Approximation Method (VAM).
    """
    cost_matrix = np.array(costs, dtype=float)
    s = np.array(supply, dtype=float)
    d = np.array(demand, dtype=float)

    if s.sum() != d.sum():
        print("Warning: Problem is unbalanced. You may need a dummy row/col.")

    rows, cols = cost_matrix.shape
    alloc = np.zeros((rows, cols))

    while s.sum() > 0 and d.sum() > 0:
        row_penalties = np.zeros(rows)
        col_penalties = np.zeros(cols)

        for i in range(rows):
            if s[i] > 0:  
                valid_costs = np.sort(cost_matrix[i, :])
                valid_costs = valid_costs[~np.isinf(valid_costs)]
                
                if len(valid_costs) > 1:
                    row_penalties[i] = valid_costs[1] - valid_costs[0]
                else:
                    row_penalties[i] = 0 
            else:
                row_penalties[i] = -1 

        for j in range(cols):
            if d[j] > 0:  
                valid_costs = np.sort(cost_matrix[:, j])
                valid_costs = valid_costs[~np.isinf(valid_costs)]
                
                if len(valid_costs) > 1:
                    col_penalties[j] = valid_costs[1] - valid_costs[0]
                else:
                    col_penalties[j] = 0
            else:
                col_penalties[j] = -1

        max_row_penalty = np.max(row_penalties)
        max_col_penalty = np.max(col_penalties)

        if max_row_penalty >= max_col_penalty:
            i = np.argmax(row_penalties)
            j = np.argmin(cost_matrix[i, :])
        else:
            j = np.argmax(col_penalties)
            i = np.argmin(cost_matrix[:, j])

        xij = min(s[i], d[j])
        alloc[i, j] = xij

        s[i] -= xij
        d[j] -= xij

        if s[i] == 0:
            cost_matrix[i, :] = np.inf
        elif d[j] == 0:
            cost_matrix[:, j] = np.inf

    return alloc

if __name__ == "__main__":
    costs = [
        [3, 1, 7, 4],
        [2, 6, 5, 9],
        [8, 3, 3, 2]
    ]
    
    supply = [300, 400, 500]
    demand = [250, 350, 400, 200]
    
    allocated_matrix = vogels_approximation_method(costs, supply, demand)
    
    print("VAM Final Allocation Matrix:")
    print(allocated_matrix)
    
    original_costs = np.array(costs)
    total_cost = np.sum(allocated_matrix * original_costs)
    print(f"\nTotal Transportation Cost (VAM): {total_cost}")


"""
Output:

VAM Final Allocation Matrix:
[[  0. 300.   0.   0.]
 [250.   0. 150.   0.]
 [  0.  50. 250. 200.]]

Total Transportation Cost (VAM): 2850.0
"""