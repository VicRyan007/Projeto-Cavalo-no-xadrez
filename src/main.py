import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors
import numpy as np
import time
import os
import sys
import argparse

from maps import example_map_obstacles, min_terrain_cost
from astar import a_star, heuristic_h1, heuristic_h2


def show_side_by_side(grid, start, goal):
    min_cost = min_terrain_cost(grid)
    # prepare heuristics: H1 weak, H2 strong (uses BFS internally)
    h2_cache = {}
    def h1(a,b,mc):
        return heuristic_h1(a,b,mc)
    def h2(a,b,mc):
        return heuristic_h2(a,b,mc,h2_cache)

    # run with recording
    path1, g1, nodes1, snaps1 = a_star(start,goal,grid,h1,min_cost,record_progress=True)
    path2, g2, nodes2, snaps2 = a_star(start,goal,grid,h2,min_cost,record_progress=True)

    print(f"H1 nós expandidos: {nodes1}")
    print(f"H2 nós expandidos: {nodes2}")

    # Convert grid to numpy for plotting
    arr = np.array(grid)
    cmap = plt.cm.get_cmap('viridis')

    # Preparar legendas (pré-calculadas) e normalização de cores
    finite_vals = arr[np.isfinite(arr)]
    if finite_vals.size > 0:
        vmin = float(np.nanmin(finite_vals))
        vmax = float(np.nanmax(finite_vals))
    else:
        vmin, vmax = 0.0, 1.0
    norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    terrain_samples = [(0.5, 'Estrada (0.5)'), (1.0, 'Terra (1.0)'), (5.0, 'Lama (5.0)')]
    terrain_patches = []
    for val,label in terrain_samples:
        color = plt.cm.terrain(norm(val))
        terrain_patches.append(Patch(facecolor=color, edgecolor='black', label=label))
    terrain_patches.append(Patch(facecolor='black', edgecolor='black', label='Barreira (∞)'))
    marker_handles = [Line2D([0],[0], marker='s', color='w', markerfacecolor='cyan', markersize=10, label='Open'),
                      Line2D([0],[0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Closed'),
                      Line2D([0],[0], marker='*', color='w', markerfacecolor='green', markersize=15, label='Início'),
                      Line2D([0],[0], marker='X', color='w', markerfacecolor='yellow', markersize=12, label='Objetivo'),
                      Line2D([0],[0], color='black', lw=2, label='Caminho ótimo')]

    # Show dynamic progression side-by-side
    max_steps = max(len(snaps1), len(snaps2))
    # Usar GridSpec para reservar uma coluna à esquerda para legendas grandes
    from matplotlib.gridspec import GridSpec
    fig = plt.figure(figsize=(12,9))
    gs = GridSpec(2, 3, width_ratios=[0.6, 1, 1], figure=fig, wspace=0.3, hspace=0.4)
    axs = np.empty((2,2), dtype=object)
    axs[0,0] = fig.add_subplot(gs[0,1])
    axs[0,1] = fig.add_subplot(gs[0,2])
    axs[1,0] = fig.add_subplot(gs[1,1])
    axs[1,1] = fig.add_subplot(gs[1,2])
    for i in range(max_steps):
        axs[0,0].clear(); axs[0,1].clear(); axs[1,0].clear(); axs[1,1].clear()
        axs[0,0].set_title('H1 Terrain + Search')
        axs[0,1].set_title('H2 Terrain + Search')
        axs[1,0].set_title('H1 Final Path G map')
        axs[1,1].set_title('H2 Final Path G map')

        # plot terrain
        im0 = axs[0,0].imshow(arr, cmap='terrain', origin='lower')
        im1 = axs[0,1].imshow(arr, cmap='terrain', origin='lower')

        if i < len(snaps1):
            open1, closed1, gmap1 = snaps1[i]
            ox = [p[0] for p in open1]; oy=[p[1] for p in open1]
            cx = [p[0] for p in closed1]; cy=[p[1] for p in closed1]
            axs[0,0].scatter(ox,oy, c='cyan', marker='s', s=80, label='open')
            axs[0,0].scatter(cx,cy, c='red', marker='s', s=80, label='closed')
        if i < len(snaps2):
            open2, closed2, gmap2 = snaps2[i]
            ox = [p[0] for p in open2]; oy=[p[1] for p in open2]
            cx = [p[0] for p in closed2]; cy=[p[1] for p in closed2]
            axs[0,1].scatter(ox,oy, c='cyan', marker='s', s=80)
            axs[0,1].scatter(cx,cy, c='red', marker='s', s=80)

        # final G maps
        if path1 is not None:
            gmap1 = np.array(g1)
            gmap1[gmap1==float('inf')] = np.nan
            axs[1,0].imshow(gmap1, cmap='hot', origin='lower')
            px=[p[0] for p in path1]; py=[p[1] for p in path1]
            axs[1,0].plot(px,py, c='black')
        if path2 is not None:
            gmap2 = np.array(g2)
            gmap2[gmap2==float('inf')]=np.nan
            axs[1,1].imshow(gmap2, cmap='hot', origin='lower')
            if path2 is not None:
                px=[p[0] for p in path2]; py=[p[1] for p in path2]
                axs[1,1].plot(px,py, c='black')

        # mark start and goal
        for ax in axs.flatten():
            ax.scatter([start[0]],[start[1]],c='green',s=120,marker='*')
            ax.scatter([goal[0]],[goal[1]],c='yellow',s=120,marker='X')
            ax.set_xticks(range(8)); ax.set_yticks(range(8))

        # (legenda global adicionada após o loop)

        # draw one frame to the figure canvas
        fig.canvas.draw()

    # adicionar legendas específicas por subplot (contextuais)
    # Handles separados
    search_handles = [Line2D([0],[0], marker='s', color='w', markerfacecolor='cyan', markersize=10, label='Open'),
                      Line2D([0],[0], marker='s', color='w', markerfacecolor='red', markersize=10, label='Closed')]
    start_goal_handles = [Line2D([0],[0], marker='*', color='w', markerfacecolor='green', markersize=12, label='Início'),
                          Line2D([0],[0], marker='X', color='w', markerfacecolor='yellow', markersize=10, label='Objetivo'),
                          Line2D([0],[0], color='black', lw=2, label='Caminho ótimo')]

    # Top-left: mostrar terreno + open/closed + início/objetivo
    # Colocar a legenda fora, à esquerda do respectivo subplot
    # Posicionar legendas grandes fora da área do subplot, na coluna esquerda reservada
    # Usar deslocamento maior para garantir não sobreposição
    axs[0,0].legend(handles=terrain_patches + search_handles + start_goal_handles,
                    loc='center left', bbox_to_anchor=(-1.0, 0.5), bbox_transform=axs[0,0].transAxes,
                    fontsize=10, framealpha=0.9)
    # Top-right: mostrar open/closed + início/objetivo (o mapa H2 foca na busca)
    axs[0,1].legend(handles=search_handles + start_goal_handles + terrain_patches,
                    loc='center left', bbox_to_anchor=(-1.0, 0.5), bbox_transform=axs[0,1].transAxes,
                    fontsize=10, framealpha=0.9)

    # Bottom maps: adicionar colorbars que indicam o custo G acumulado
    # e posicionar legenda de início/objetivo à esquerda do mapa de G
    # H1 G map: colorbar + legenda de início/objetivo/caminho
    if path1 is not None:
        im_h1 = axs[1,0].images[0]
        cb1 = fig.colorbar(im_h1, ax=axs[1,0], fraction=0.046, pad=0.04)
        cb1.set_label('G (custo acumulado) — soma dos custos de terreno')
    axs[1,0].legend(handles=start_goal_handles, loc='center left', bbox_to_anchor=(-1.0, 0.5), bbox_transform=axs[1,0].transAxes, fontsize=10, framealpha=0.9)
    # H2 G map
    if path2 is not None:
        im_h2 = axs[1,1].images[0]
        cb2 = fig.colorbar(im_h2, ax=axs[1,1], fraction=0.046, pad=0.04)
        cb2.set_label('G (custo acumulado) — soma dos custos de terreno')
    axs[1,1].legend(handles=start_goal_handles, loc='center left', bbox_to_anchor=(-1.0, 0.5), bbox_transform=axs[1,1].transAxes, fontsize=10, framealpha=0.9)

    # pequena anotação geral sobre modelo (opcional)
    # Cabeçalho explicativo
    fig.text(0.5, 0.955, 'Tabuleiro 8x8 — custos: Estrada 0.5, Terra 1.0, Lama 5.0, Barreira ∞. G = soma dos custos das casas visitadas.', ha='center', fontsize=10)

    # Ajustar layout: a GridSpec já reserva coluna esquerda; garantir que nada sobreponha
    plt.subplots_adjust(left=0.12, right=0.98, top=0.93, bottom=0.05)

    # retorna figura e dados para possível salvamento
    return fig, (path1, g1, nodes1), (path2, g2, nodes2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--headless', action='store_true', help='Run without interactive GUI and save outputs to output/')
    return p.parse_args()


if __name__=='__main__':
    args = parse_args()
    grid = example_map_obstacles()
    start = (0,0)
    goal = (7,7)
    out = show_side_by_side(grid,start,goal)
    if args.headless:
        fig, r1, r2 = out
        os.makedirs('output', exist_ok=True)
        fig.savefig(os.path.join('output','comparison.png'))
        # save metrics
        (p1,g1,n1) = r1
        (p2,g2,n2) = r2
        with open(os.path.join('output','metrics.txt'),'w') as f:
            f.write(f'H1 nós_expandidos: {n1}\n')
            f.write(f'H2 nós_expandidos: {n2}\n')
        print('Salvo em output/comparison.png e output/metrics.txt')
    else:
        plt.show()
