import numpy as np

def least_cost_method(costs, supply, demand):
    cost_matrix = np.array(costs, dtype=float)
    s = np.array(supply, dtype=float)
    d = np.array(demand, dtype=float)

    if s.sum()!=d.sum():
        print("Warning: Problem is unbalanced. You may need a dummy row/col.")

    rows, cols = cost_matrix.shape
    alloc = np.zeros((rows, cols))

    while s.sum()>0 and d.sum()>0:
        min_idx = np.unravel_index(np.argmin(cost_matrix), cost_matrix.shape)
        i, j = min_idx[0], min_idx[1]

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
    # Example Cost Matrix
    costs = [
        [3, 1, 7, 4],
        [2, 6, 5, 9],
        [8, 3, 3, 2]
    ]
    
    supply = [300, 400, 500]
    
    demand = [250, 350, 400, 200]
    
    allocated_matrix = least_cost_method(costs, supply, demand)
    
    print("Final Allocation Matrix:")
    print(allocated_matrix)
    
    original_costs = np.array(costs)
    total_cost = np.sum(allocated_matrix * original_costs)
    print(f"\nTotal Transportation Cost: {total_cost}")
