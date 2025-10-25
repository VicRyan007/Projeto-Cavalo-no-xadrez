import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches 

from busca_a_estrela import (
    Tabuleiro, 
    BuscaAEstrela, 
    gerar_matriz_aleatoria, 
    TERRAIN_VALUES
)
from matplotlib.colors import ListedColormap, BoundaryNorm 

def plotar_resultados(tabuleiro, caminho, closed_list, g_map, nome_heuristica):
    
    largura = tabuleiro.largura
    altura = tabuleiro.altura
    
    mapa_terreno = np.zeros((altura, largura))
    mapa_g_cost = np.full((altura, largura), np.nan)
    
    for y in range(altura):
        for x in range(largura):
            casa = tabuleiro.get_casa(x, y)
            mapa_terreno[y, x] = casa.custo_terreno
            if casa in g_map:
                mapa_g_cost[y, x] = g_map[casa]

    mapa_terreno[mapa_terreno == np.inf] = -1 
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9))
    fig.suptitle(f'Resultados da Busca A* com {nome_heuristica}', fontsize=16)

    cores_terreno = ['#000000', '#6BBE6B', '#E3C18E', '#8B4513']
    limites_terreno = [-1.5, -0.5, 0.75, 1.5, 5.5]
    cmap_terreno = ListedColormap(cores_terreno)
    norm_terreno = BoundaryNorm(limites_terreno, cmap_terreno.N)

    im1 = ax1.imshow(mapa_terreno, cmap=cmap_terreno, norm=norm_terreno)
    ax1.set_title(f"Mapa de Terreno e Caminho Ótimo")


    cmap_busca = plt.cm.get_cmap('viridis')
    cmap_busca.set_bad('lightgray')
    
    im2 = ax2.imshow(mapa_g_cost, cmap=cmap_busca)
    ax2.set_title(f"Área de Busca (Nós Expandidos: {len(closed_list)})")
    fig.colorbar(im2, ax=ax2, label="Custo G (Custo para Chegar)")

    if caminho:
        x_coords = [casa.x for casa in caminho]
        y_coords = [casa.y for casa in caminho]
        
        ax1.plot(x_coords, y_coords, marker='o', color='red', markersize=5,
                 linewidth=2, label='Caminho Ótimo')
        ax1.legend(loc='upper left')
        
        ax2.plot(x_coords, y_coords, marker='x', color='red', markersize=5,
                 linewidth=2, label='Caminho Ótimo')
        ax2.legend(loc='upper left')

    for ax in [ax1, ax2]:
        ax.set_xticks(np.arange(largura))
        ax.set_yticks(np.arange(altura))
        ax.set_xticklabels(np.arange(largura))
        ax.set_yticklabels(np.arange(altura))

        ax.set_xticks(np.arange(largura) - 0.5, minor=True)
        ax.set_yticks(np.arange(altura) - 0.5, minor=True)
        ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    patch_estrada = mpatches.Patch(color='#6BBE6B', label='Estrada (0.5)')
    patch_terra = mpatches.Patch(color='#E3C18E', label='Terra (1.0)')
    patch_lama = mpatches.Patch(color='#8B4513', label='Lama (5.0)')
    patch_barreira = mpatches.Patch(color='#000000', label='Barreira (Inf)')
    
    fig.legend(handles=[patch_estrada, patch_terra, patch_lama, patch_barreira],
               loc='lower center', ncol=4, bbox_to_anchor=(0.5, 0.01))

    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    plt.show()

inf = float('inf')

print("Gerando novo tabuleiro aleatório...")

CUSTOS_TABULEIRO_ALEATORIO = gerar_matriz_aleatoria(largura=8, altura=8)

origem_coords = (0, 0)  
destino_coords = (7, 7) 

print(f"Garantindo que origem {origem_coords} e destino {destino_coords} não são barreiras.")
CUSTOS_TABULEIRO_ALEATORIO[origem_coords[1]][origem_coords[0]] = TERRAIN_VALUES['terra']
CUSTOS_TABULEIRO_ALEATORIO[destino_coords[1]][destino_coords[0]] = TERRAIN_VALUES['terra']

tabuleiro_teste = Tabuleiro(CUSTOS_TABULEIRO_ALEATORIO)
origem = tabuleiro_teste.get_casa(origem_coords[0], origem_coords[1])
destino = tabuleiro_teste.get_casa(destino_coords[0], destino_coords[1])

buscaaestrela_cavalo = BuscaAEstrela(tabuleiro_teste, destino)

print("--- Rodando com Heurística H1 (Fraca) ---")
caminho_h1, custo_h1, closed_h1, g_map_h1 = buscaaestrela_cavalo.busca_a_estrela(
    origem, tipo_heuristica='h1'
)

if caminho_h1:
    print(f'H1: Caminho encontrado! Custo: {custo_h1:.2f}, Nós Expandidos: {len(closed_h1)}')
    plotar_resultados(tabuleiro_teste, caminho_h1, closed_h1, g_map_h1, "H1 (Fraca)")
else:
    print('H1: Caminho não encontrado.')

print("\n" + "="*30 + "\n")

print("--- Rodando com Heurística H2 (Forte) ---")
caminho_h2, custo_h2, closed_h2, g_map_h2 = buscaaestrela_cavalo.busca_a_estrela(
    origem, tipo_heuristica='h2'
)

if caminho_h2:
    print(f'H2: Caminho encontrado! Custo: {custo_h2:.2f}, Nós Expandidos: {len(closed_h2)}')
    plotar_resultados(tabuleiro_teste, caminho_h2, closed_h2, g_map_h2, "H2 (Forte)")
else:
    print('H2: Caminho não encontrado.')