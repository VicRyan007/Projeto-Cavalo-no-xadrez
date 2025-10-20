# Salve como: visualizador_cavalo.py

import matplotlib.pyplot as plt
import numpy as np
from busca_a_estrela import Tabuleiro, BuscaAEstrela

# --- FUNÇÃO DE PLOTAGEM (Onde a mágica acontece) ---

def plotar_resultados(tabuleiro, caminho, closed_list, g_map, nome_heuristica):
    """
    Gera a visualização gráfica dos resultados usando Matplotlib.
    """
    
    # --- 1. Preparar os dados para os mapas ---
    largura = tabuleiro.largura
    altura = tabuleiro.altura
    
    # Mapa de Terreno (Req A.A)
    mapa_terreno = np.zeros((altura, largura))
    
    # Mapa de Custo G / Área de Busca (Req A.C / B.B)
    # Começa com 'NaN' (Not a Number) para casas não visitadas
    mapa_g_cost = np.full((altura, largura), np.nan)
    
    for y in range(altura):
        for x in range(largura):
            casa = tabuleiro.get_casa(x, y)
            mapa_terreno[y, x] = casa.custo_terreno
            if casa in g_map:
                mapa_g_cost[y, x] = g_map[casa]

    # Lida com barreiras (inf) para melhor visualização
    mapa_terreno[mapa_terreno == np.inf] = -1 # Marcar barreiras com -1
    
    
    # --- 2. Criar os Gráficos (Figura com 2 subplots) ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle(f'Resultados da Busca A* com {nome_heuristica}', fontsize=16)

    # --- SUBPLOT 1: Mapa de Terreno e Caminho Ótimo (Req A.A, A.B) ---
    cmap_terreno = plt.cm.get_cmap('terrain_r')
    cmap_terreno.set_under('black') # Barreiras (-1) ficarão pretas
    
    im1 = ax1.imshow(mapa_terreno, cmap=cmap_terreno, vmin=0)
    ax1.set_title(f"Mapa de Terreno e Caminho Ótimo")
    fig.colorbar(im1, ax=ax1, label="Custo do Terreno")

    # --- SUBPLOT 2: Área de Busca e Mapa de Custo G (Req A.C, B.B) ---
    cmap_busca = plt.cm.get_cmap('viridis')
    cmap_busca.set_bad('lightgray') # Casas não visitadas (NaN) ficarão cinza
    
    im2 = ax2.imshow(mapa_g_cost, cmap=cmap_busca)
    ax2.set_title(f"Área de Busca (Nós Expandidos: {len(closed_list)})")
    fig.colorbar(im2, ax=ax2, label="Custo G (Custo para Chegar)")

    # --- 3. Desenhar o Caminho Ótimo em ambos os gráficos ---
    if caminho:
        x_coords = [casa.x for casa in caminho]
        y_coords = [casa.y for casa in caminho]
        
        # Caminho no gráfico 1
        ax1.plot(x_coords, y_coords, marker='o', color='red', markersize=5,
                 linewidth=2, label='Caminho Ótimo')
        ax1.legend(loc='upper left')
        
        # Caminho no gráfico 2
        ax2.plot(x_coords, y_coords, marker='x', color='red', markersize=5,
                 linewidth=2, label='Caminho Ótimo')
        ax2.legend(loc='upper left')

    # Ajusta os eixos
    for ax in [ax1, ax2]:
        ax.set_xticks(np.arange(largura))
        ax.set_yticks(np.arange(altura))
        ax.set_xticklabels(np.arange(largura))
        ax.set_yticklabels(np.arange(altura))

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


# --- DEFINIÇÃO DO TABULEIRO 8x8 ---
inf = float('inf')

# <<< USE ESTE NOVO MAPA DE CUSTOS >>>
CUSTOS_TABULEIRO_EXEMPLO = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 0.5, 0.5, 0.5, 0.5, 0.5, 1.0], # "Estrada" fácil...
    [1.0, 1.0, 0.5, 5.0, 5.0, 5.0, 0.5, 1.0], # ...que vira "Lama"
    [1.0, 1.0, 0.5, 5.0, 5.0, 5.0, 0.5, 1.0], # "Lama" no meio
    [1.0, 1.0, 0.5, 5.0, 5.0, 5.0, 0.5, 1.0], 
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
]

# --- EXECUÇÃO DA BUSCA E VISUALIZAÇÃO ---

# 1. Configuração inicial
tabuleiro_teste = Tabuleiro(CUSTOS_TABULEIRO_EXEMPLO)

# <<< MUDE A ORIGEM E O DESTINO >>>
origem = tabuleiro_teste.get_casa(2, 2)   # Começa na "estrada"
destino = tabuleiro_teste.get_casa(6, 4)  # Objetivo do outro lado da lama

buscaaestrela_cavalo = BuscaAEstrela(tabuleiro_teste, destino)

# 2. Executar e Plotar com Heurística 1 (Fraca)
# (O restante do código permanece o mesmo)
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

# 3. Executar e Plotar com Heurística 2 (Forte)
print("--- Rodando com Heurística H2 (Forte) ---")
caminho_h2, custo_h2, closed_h2, g_map_h2 = buscaaestrela_cavalo.busca_a_estrela(
    origem, tipo_heuristica='h2'
)

if caminho_h2:
    print(f'H2: Caminho encontrado! Custo: {custo_h2:.2f}, Nós Expandidos: {len(closed_h2)}')
    plotar_resultados(tabuleiro_teste, caminho_h2, closed_h2, g_map_h2, "H2 (Forte)")
else:
    print('H2: Caminho não encontrado.')