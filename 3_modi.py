import numpy as np

def get_uv(costs, alloc):
    """Calculates the dual variables u (rows) and v (columns)."""
    rows, cols = alloc.shape
    u = [None] * rows
    v = [None] * cols
    
    u[0] = 0
    basic_cells = list(zip(*np.where(alloc > 0)))
    
    resolved_something = True
    while resolved_something:
        resolved_something = False
        for r, c in basic_cells:
            if u[r] is not None and v[c] is None:
                v[c] = costs[r, c] - u[r]
                resolved_something = True
            elif v[c] is not None and u[r] is None:
                u[r] = costs[r, c] - v[c]
                resolved_something = True
                
        if not resolved_something and (None in u or None in v):
            if None in u:
                u[u.index(None)] = 0
            else:
                v[v.index(None)] = 0
            resolved_something = True
            
    return u, v

def find_loop(start_cell, basic_cells):
    """Finds the closed loop of basic cells starting at the entering cell."""
    cells = set(basic_cells)
    cells.add(start_cell)
    
    while True:
        to_remove = []
        for r, c in cells:
            row_count = sum(1 for x, y in cells if x == r)
            col_count = sum(1 for x, y in cells if y == c)
            if row_count == 1 or col_count == 1:
                to_remove.append((r, c))
                
        if not to_remove:
            break
        for cell in to_remove:
            cells.remove(cell)
            
    def dfs(curr, path, move_horizontal):
        if len(path) > 3 and curr == start_cell:
            return path
        
        for r, c in cells:
            if (r, c) not in path or ((r, c) == start_cell and len(path) >= 3):
                if move_horizontal and r == curr[0] and c != curr[1]:
                    res = dfs((r, c), path + [(r, c)], False)
                    if res: return res
                elif not move_horizontal and c == curr[1] and r != curr[0]:
                    res = dfs((r, c), path + [(r, c)], True)
                    if res: return res
        return None
        
    loop = dfs(start_cell, [start_cell], True)
    if not loop:
        loop = dfs(start_cell, [start_cell], False)
        
    return loop

def modi_method(costs, initial_alloc):
    """
    Optimizes a transportation problem using the MODI method.
    """
    costs = np.array(costs, dtype=float)
    alloc = np.array(initial_alloc, dtype=float)
    rows, cols = alloc.shape
    iteration = 1
    
    while True:
        u, v = get_uv(costs, alloc)
        
        min_penalty = 0
        entering_cell = None
        
        for i in range(rows):
            for j in range(cols):
                if alloc[i, j] == 0:
                    penalty = costs[i, j] - (u[i] + v[j])
                    if penalty < min_penalty:
                        min_penalty = penalty
                        entering_cell = (i, j)
        
        if min_penalty >= -1e-7:
            print(f"\nOptimal solution found after {iteration - 1} iterations.")
            break
            
        print(f"Iteration {iteration}: Entering cell {entering_cell} with penalty {min_penalty}")
        
        basic_cells = list(zip(*np.where(alloc > 0)))
        loop_path = find_loop(entering_cell, basic_cells)
        
        if not loop_path:
            raise ValueError("Degeneracy prevents loop formation. Requires epsilon perturbation.")
            
        loop_corners = loop_path[:-1] 
        
        minus_cells = loop_corners[1::2]
        plus_cells = loop_corners[0::2]
        
        theta = min(alloc[r, c] for r, c in minus_cells)
        
        for r, c in plus_cells:
            alloc[r, c] += theta
        for r, c in minus_cells:
            alloc[r, c] -= theta
            
        iteration += 1
        
    return alloc

if __name__ == "__main__":
    costs = [
        [3, 1, 7, 4],
        [2, 6, 5, 9],
        [8, 3, 3, 2]
    ]
    
    initial_allocation = [
        [250, 50,  0,   0],
        [  0, 300, 100, 0],
        [  0,  0,  300, 200]
    ]
    
    print("Initial Allocation Matrix:")
    print(np.array(initial_allocation))
    initial_cost = np.sum(np.array(initial_allocation) * np.array(costs))
    print(f"Initial Cost: {initial_cost}\n")
    
    optimal_allocation = modi_method(costs, initial_allocation)
    
    print("\nFinal Optimal Allocation Matrix:")
    print(optimal_allocation)
    
    final_cost = np.sum(optimal_allocation * np.array(costs))
    print(f"\nFinal Minimized Transportation Cost: {final_cost}")