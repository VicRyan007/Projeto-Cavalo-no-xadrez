import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from maps import example_map_obstacles, min_terrain_cost
from astar import a_star, heuristic_h1, heuristic_h2
from validate_path import validate_path

if __name__=='__main__':
    grid = example_map_obstacles()
    start = (0,0)
    goal = (7,7)
    min_cost = min_terrain_cost(grid)

    path1, g1, nodes1, _ = a_star(start, goal, grid, lambda a,b,mc: heuristic_h1(a,b,mc), min_cost, record_progress=False)
    path2, g2, nodes2, _ = a_star(start, goal, grid, lambda a,b,mc: heuristic_h2(a,b,mc,{}), min_cost, record_progress=False)

    valid1 = validate_path(path1, grid)
    valid2 = validate_path(path2, grid)

    print('H1 nós expandidos:', nodes1)
    print('H2 nós expandidos:', nodes2)
    print('H1 caminho válido:', valid1)
    print('H2 caminho válido:', valid2)

    if path1 is not None:
        print('H1 comprimento do caminho (passos):', len(path1))
        print('H1 custo total G reportado (gscore do objetivo):', g1[goal[1]][goal[0]])
    if path2 is not None:
        print('H2 comprimento do caminho (passos):', len(path2))
        print('H2 custo total G reportado (gscore do objetivo):', g2[goal[1]][goal[0]])
