from astar import KNIGHT_MOVES


def is_knight_move(a,b):
    dx = abs(a[0]-b[0]); dy = abs(a[1]-b[1])
    return (dx,dy) in [(1,2),(2,1)]


def validate_path(path, grid):
    if path is None:
        return False, 'sem caminho'
    total = 0
    for i in range(len(path)):
        x,y = path[i]
        cost = grid[y][x]
        if cost==float('inf'):
            return False, f'ponto {i} é uma barreira'
        total += cost
        if i>0:
            if not is_knight_move(path[i-1], path[i]):
                return False, f'ponto {i} não é movimento de cavalo'
    return True, total
