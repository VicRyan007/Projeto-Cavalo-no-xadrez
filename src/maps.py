import random
from math import inf

INF = float('inf')

# Custos de terreno
TERRA = 1.0
ESTRADA = 0.5
LAMA = 5.0
BARREIRA = INF


def example_map_easy():
    # Principalmente terra com algumas estradas e alguns pontos de lama
    grid = [[TERRA for _ in range(8)] for _ in range(8)]
    for y in range(8):
        for x in range(8):
            if random.random() < 0.1:
                grid[y][x] = ESTRADA
            if random.random() < 0.05:
                grid[y][x] = LAMA
    return grid


def example_map_obstacles():
    # Mapa com uma barreira central (impossível atravessar) e alguns custos altos
    grid = [[TERRA for _ in range(8)] for _ in range(8)]
    # adicionar bloco de barreiras no centro
    for x in range(2, 6):
        grid[3][x] = BARREIRA
        grid[4][x] = BARREIRA
    # custos variados próximos
    grid[2][4] = LAMA
    grid[5][3] = LAMA
    for x in range(8):
        if x % 3 == 0:
            grid[0][x] = ESTRADA
    return grid


def example_map_high_cost():
    # Mapa com muitas casas de alto custo (lama) e um corredor de estrada
    grid = [[LAMA if random.random() < 0.2 else TERRA for _ in range(8)] for _ in range(8)]
    # criar um corredor de baixo custo
    for y in range(8):
        grid[y][1] = ESTRADA
    grid[0][0] = ESTRADA
    return grid


def min_terrain_cost(grid):
    m = min(min(row) for row in grid)
    return m
