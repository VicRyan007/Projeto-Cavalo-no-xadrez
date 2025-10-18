import heapq
import math
from collections import deque

KNIGHT_MOVES = [(2,1),(1,2),(-1,2),(-2,1),(-2,-1),(-1,-2),(1,-2),(2,-1)]

INF = float('inf')


def in_bounds(x,y):
    return 0 <= x < 8 and 0 <= y < 8


def min_knight_moves(start,goal):
    # BFS em um tabuleiro 8x8 para obter o número mínimo de movimentos de cavalo entre duas casas
    sx,sy = start
    gx,gy = goal
    if (sx,sy)==(gx,gy):
        return 0
    visited = [[False]*8 for _ in range(8)]
    q = deque()
    q.append((sx,sy,0))
    visited[sy][sx] = True
    while q:
        x,y,d = q.popleft()
        for dx,dy in KNIGHT_MOVES:
            nx,ny = x+dx, y+dy
            if not in_bounds(nx,ny):
                continue
            if visited[ny][nx]:
                continue
            if (nx,ny)==(gx,gy):
                return d+1
            visited[ny][nx]=True
            q.append((nx,ny,d+1))
    return INF


def heuristic_h1(a,b,min_cost):
    # H1: distância Euclidiana dividida pela maior deslocamento do cavalo (sqrt(5)), vezes min_cost.
    (x1,y1),(x2,y2)= (a,b)
    euclid = math.hypot(x2-x1,y2-y1)
    return (euclid / math.sqrt(5.0)) * min_cost


def heuristic_h2(a,b,min_cost,cache=None):
    # H2: número mínimo de movimentos de cavalo multiplicado por min_cost. Usa cache (dict) para acelerar.
    key = (a,b)
    if cache is not None and key in cache:
        moves = cache[key]
    else:
        moves = min_knight_moves(a,b)
        if cache is not None:
            cache[key]=moves
    return moves * min_cost


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def a_star(start, goal, terrain, heuristic_func, min_cost, record_progress=False):
    # terrain: lista 8x8 de custos (float ou INF)
    # start, goal: tuplas (x,y)
    sx,sy = start
    gx,gy = goal
    start_cost = terrain[sy][sx]
    if start_cost==INF or terrain[gy][gx]==INF:
        return None, None, 0, []

    open_heap = []  # elementos: (f, g, x, y)
    g_score = [[INF]*8 for _ in range(8)]
    came_from = {}
    open_set = set()
    closed_set = set()
    snapshots = []

    start_g = start_cost
    h0 = heuristic_func(start, goal, min_cost)
    heapq.heappush(open_heap,(start_g + h0, start_g, sx, sy))
    g_score[sy][sx]=start_g
    open_set.add((sx,sy))

    nodes_expanded = 0

    while open_heap:
        f,g,x,y = heapq.heappop(open_heap)
        if (x,y) in closed_set:
            continue
        open_set.discard((x,y))
        closed_set.add((x,y))
        nodes_expanded += 1

        if record_progress:
            # registra um snapshot raso: cópias do open e closed
            snapshots.append((set(open_set), set(closed_set), [row[:] for row in g_score]))

        if (x,y)==(gx,gy):
            path = reconstruct_path(came_from,(x,y))
            return path, g_score, nodes_expanded, snapshots

        for dx,dy in KNIGHT_MOVES:
            nx,ny = x+dx, y+dy
            if not in_bounds(nx,ny):
                continue
            cost = terrain[ny][nx]
            if cost==INF:
                continue
            tentative_g = g_score[y][x] + cost
            if tentative_g < g_score[ny][nx]:
                g_score[ny][nx] = tentative_g
                came_from[(nx,ny)] = (x,y)
                h = heuristic_func((nx,ny), (gx,gy), min_cost)
                heapq.heappush(open_heap,(tentative_g + h, tentative_g, nx, ny))
                open_set.add((nx,ny))

    return None, g_score, nodes_expanded, snapshots
