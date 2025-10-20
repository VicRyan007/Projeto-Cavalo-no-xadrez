# Salve como: animador_comparativo.py

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from busca_a_estrela import Tabuleiro, BuscaAEstrelaAnimada 
from matplotlib.colors import ListedColormap, BoundaryNorm
import itertools

# --- DEFINIÇÃO DO TABULEIRO ---
inf = float('inf')
CUSTOS_TABULEIRO_EXEMPLO = [
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 0.5, 0.5, 0.5, 1.0, 1.0],
    [1.0, 1.0, 1.0, 0.5, inf, 0.5, 1.0, 1.0],
    [1.0, 1.0, 1.0, 0.5, inf, 5.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, inf, 5.0, 1.0, 1.0],
    [1.0, 5.0, 5.0, 5.0, 5.0, 5.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
]

# --- CONFIGURAÇÃO DA BUSCA ---
tabuleiro = Tabuleiro(CUSTOS_TABULEIRO_EXEMPLO)
origem = tabuleiro.get_casa(0, 0)
destino = tabuleiro.get_casa(7, 4)

# --- INICIALIZA OS DOIS GERADORES ---
print("Inicializando geradores (H1 e H2)...")
busca_h1 = BuscaAEstrelaAnimada(tabuleiro, destino)
gerador_h1 = busca_h1.busca_passo_a_passo(origem, tipo_heuristica='h1')

busca_h2 = BuscaAEstrelaAnimada(tabuleiro, destino)
gerador_h2 = busca_h2.busca_passo_a_passo(origem, tipo_heuristica='h2')

# --- CONFIGURAÇÃO DA ANIMAÇÃO (LADO A LADO) ---
# Cria 1 figura com 2 subplots (1 linha, 2 colunas)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

# Prepara o mapa de terreno base (será usado em ambos)
largura = tabuleiro.largura
altura = tabuleiro.altura
mapa_terreno = np.zeros((altura, largura))
for y in range(altura):
    for x in range(largura):
        mapa_terreno[y, x] = tabuleiro.get_casa(x, y).custo_terreno
mapa_terreno[mapa_terreno == np.inf] = -1

cmap_terreno = plt.cm.get_cmap('terrain_r')
cmap_terreno.set_under('black')
ax1.imshow(mapa_terreno, cmap=cmap_terreno, vmin=0)
ax2.imshow(mapa_terreno, cmap=cmap_terreno, vmin=0)

# Prepara mapas de cores para listas (Open/Closed)
cmap_listas = ListedColormap(['#FF8C0090', '#00FF0090']) 
norm = BoundaryNorm([1.5, 2.5, 3.5], cmap_listas.N)
mapa_listas_h1 = np.full((altura, largura), np.nan)
mapa_listas_h2 = np.full((altura, largura), np.nan)

# --- CRIA OS ARTISTAS DE ANIMAÇÃO ---

# Artistas para H1 (Gráfico da Esquerda)
im_listas_h1 = ax1.imshow(mapa_listas_h1, cmap=cmap_listas, norm=norm, interpolation='none')
linha_caminho_h1, = ax1.plot([], [], marker='o', color='red', markersize=5, linewidth=2)
status_text_h1 = ax1.text(0.01, 0.01, '', transform=ax1.transAxes, 
                          color='white', backgroundcolor='black', fontsize=12)
ax1.set_title("H1 (Fraca / Dijkstra, H=0)")

# Artistas para H2 (Gráfico da Direita)
im_listas_h2 = ax2.imshow(mapa_listas_h2, cmap=cmap_listas, norm=norm, interpolation='none')
linha_caminho_h2, = ax2.plot([], [], marker='o', color='red', markersize=5, linewidth=2)
status_text_h2 = ax2.text(0.01, 0.01, '', transform=ax2.transAxes, 
                          color='white', backgroundcolor='black', fontsize=12)
ax2.set_title("H2 (Forte / A*)")


# --- FUNÇÃO GERADORA COMBINADA ---
def gerador_combinado():
    """ Puxa um frame de cada gerador (H1 e H2) a cada passo """
    for frame_h1, frame_h2 in itertools.zip_longest(gerador_h1, gerador_h2, fillvalue=None):
        yield frame_h1, frame_h2

# --- FUNÇÃO DE ATUALIZAÇÃO (para cada frame) ---
def update(frame_data):
    frame_h1, frame_h2 = frame_data
    
    artistas_retorno = []
    
    # --- Atualiza o Gráfico H1 (Esquerda) ---
    if frame_h1:
        open_list, closed_list, caminho_atual, terminado = frame_h1
        mapa_listas_h1.fill(np.nan)
        for casa in closed_list: mapa_listas_h1[casa.y, casa.x] = 2
        for casa in open_list: mapa_listas_h1[casa.y, casa.x] = 3
        im_listas_h1.set_data(mapa_listas_h1)
        artistas_retorno.append(im_listas_h1)
        
        if caminho_atual:
            x_coords = [c.x for c in caminho_atual]
            y_coords = [c.y for c in caminho_atual]
            linha_caminho_h1.set_data(x_coords, y_coords)
            artistas_retorno.append(linha_caminho_h1)

        if terminado:
            status_text_h1.set_text(f'TERMINADO!\nNós: {len(closed_list)}')
            if caminho_atual: linha_caminho_h1.set_color('lime')
        else:
            status_text_h1.set_text(f'Fechada: {len(closed_list)}\nAberta: {len(open_list)}')
        artistas_retorno.append(status_text_h1)

    # --- Atualiza o Gráfico H2 (Direita) ---
    if frame_h2:
        open_list, closed_list, caminho_atual, terminado = frame_h2
        mapa_listas_h2.fill(np.nan)
        for casa in closed_list: mapa_listas_h2[casa.y, casa.x] = 2
        for casa in open_list: mapa_listas_h2[casa.y, casa.x] = 3
        im_listas_h2.set_data(mapa_listas_h2)
        artistas_retorno.append(im_listas_h2)
        
        if caminho_atual:
            x_coords = [c.x for c in caminho_atual]
            y_coords = [c.y for c in caminho_atual]
            linha_caminho_h2.set_data(x_coords, y_coords)
            artistas_retorno.append(linha_caminho_h2)

        if terminado:
            status_text_h2.set_text(f'TERMINADO!\nNós: {len(closed_list)}')
            if caminho_atual: linha_caminho_h2.set_color('lime')
        else:
            status_text_h2.set_text(f'Fechada: {len(closed_list)}\nAberta: {len(open_list)}')
        artistas_retorno.append(status_text_h2)
        
    return artistas_retorno

# --- RODA A ANIMAÇÃO ---
print("Criando animação comparativa...")
# blit=False é mais fácil para animações complexas com subplots
ani = animation.FuncAnimation(fig, update, frames=gerador_combinado, 
                              blit=False, repeat=False, interval=200)

fig.suptitle("Comparação Visual: H1 (Fraca) vs. H2 (Forte)", fontsize=16)
plt.show()

print("Animação concluída.")