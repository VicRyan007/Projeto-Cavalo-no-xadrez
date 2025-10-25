import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import matplotlib.patches as mpatches 

from busca_a_estrela import (
    Tabuleiro, 
    BuscaAEstrelaAnimada, 
    gerar_matriz_aleatoria,  
    TERRAIN_VALUES           
)
from matplotlib.colors import ListedColormap, BoundaryNorm
import itertools

print("Gerando novo tabuleiro aleatório...")
CUSTOS_TABULEIRO_ALEATORIO = gerar_matriz_aleatoria(largura=8, altura=8)
inf = float('inf') 

origem_coords = (0, 0)
destino_coords = (7, 7)

print(f"Garantindo que origem {origem_coords} e destino {destino_coords} não são barreiras.")
CUSTOS_TABULEIRO_ALEATORIO[origem_coords[1]][origem_coords[0]] = TERRAIN_VALUES['terra']
CUSTOS_TABULEIRO_ALEATORIO[destino_coords[1]][destino_coords[0]] = TERRAIN_VALUES['terra']

tabuleiro = Tabuleiro(CUSTOS_TABULEIRO_ALEATORIO)
origem = tabuleiro.get_casa(origem_coords[0], origem_coords[1])
destino = tabuleiro.get_casa(destino_coords[0], destino_coords[1])


print("Inicializando geradores (H1 e H2)...")
busca_h1 = BuscaAEstrelaAnimada(tabuleiro, destino)
gerador_h1 = busca_h1.busca_passo_a_passo(origem, tipo_heuristica='h1')

busca_h2 = BuscaAEstrelaAnimada(tabuleiro, destino)
gerador_h2 = busca_h2.busca_passo_a_passo(origem, tipo_heuristica='h2')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

largura = tabuleiro.largura
altura = tabuleiro.altura
mapa_terreno = np.zeros((altura, largura))
for y in range(altura):
    for x in range(largura):
        mapa_terreno[y, x] = tabuleiro.get_casa(x, y).custo_terreno
mapa_terreno[mapa_terreno == np.inf] = -1

cores_terreno = ['#000000', '#6BBE6B', '#E3C18E', '#8B4513']
limites_terreno = [-1.5, -0.5, 0.75, 1.5, 5.5]
cmap_terreno = ListedColormap(cores_terreno)
norm_terreno = BoundaryNorm(limites_terreno, cmap_terreno.N)

ax1.imshow(mapa_terreno, cmap=cmap_terreno, norm=norm_terreno)
ax2.imshow(mapa_terreno, cmap=cmap_terreno, norm=norm_terreno)

for ax in [ax1, ax2]:
    ax.set_xticks(np.arange(largura))
    ax.set_yticks(np.arange(altura))
    ax.set_xticklabels(np.arange(largura))
    ax.set_yticklabels(np.arange(altura))
    ax.set_xticks(np.arange(largura) - 0.5, minor=True)
    ax.set_yticks(np.arange(altura) - 0.5, minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

cor_fechada = '#9400D3'
cor_aberta = '#00BFFF' 
marker_size = 50        

scatter_fechada_h1 = ax1.scatter([], [], s=marker_size, c=cor_fechada, marker='s')
scatter_aberta_h1 = ax1.scatter([], [], s=marker_size, c=cor_aberta, marker='s')
linha_caminho_h1, = ax1.plot([], [], marker='o', color='red', markersize=5, linewidth=2)
status_text_h1 = ax1.text(0.01, 0.01, '', transform=ax1.transAxes, 
                          color='white', backgroundcolor='black', fontsize=12)
ax1.set_title("H1 (Fraca / Dijkstra, H=0)")

scatter_fechada_h2 = ax2.scatter([], [], s=marker_size, c=cor_fechada, marker='s')
scatter_aberta_h2 = ax2.scatter([], [], s=marker_size, c=cor_aberta, marker='s')
linha_caminho_h2, = ax2.plot([], [], marker='o', color='red', markersize=5, linewidth=2)
status_text_h2 = ax2.text(0.01, 0.01, '', transform=ax2.transAxes, 
                          color='white', backgroundcolor='black', fontsize=12)
ax2.set_title("H2 (Forte / A*, H >0)")

patch_estrada = mpatches.Patch(color='#6BBE6B', label='Estrada (0.5)')
patch_terra = mpatches.Patch(color='#E3C18E', label='Terra (1.0)')
patch_lama = mpatches.Patch(color='#8B4513', label='Lama (5.0)')
patch_barreira = mpatches.Patch(color='#000000', label='Barreira (Inf)')
patch_fechada = mpatches.Patch(color=cor_fechada, label='Lista Fechada')
patch_aberta = mpatches.Patch(color=cor_aberta, label='Lista Aberta')


def gerador_combinado():
    for frame_h1, frame_h2 in itertools.zip_longest(gerador_h1, gerador_h2, fillvalue=None):
        yield frame_h1, frame_h2

def update(frame_data):
    frame_h1, frame_h2 = frame_data
    
    artistas_retorno = []
    
    if frame_h1:
        open_list, closed_list, caminho_atual, terminado = frame_h1
        
        if closed_list:
            closed_coords = np.array([(c.x, c.y) for c in closed_list])
            scatter_fechada_h1.set_offsets(closed_coords)
        else:
            scatter_fechada_h1.set_offsets(np.empty((0, 2)))
            
        if open_list:
            open_coords = np.array([(c.x, c.y) for c in open_list])
            scatter_aberta_h1.set_offsets(open_coords)
        else:
            scatter_aberta_h1.set_offsets(np.empty((0, 2)))
        artistas_retorno.extend([scatter_fechada_h1, scatter_aberta_h1])
        
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

    if frame_h2:
        open_list, closed_list, caminho_atual, terminado = frame_h2
        
        if closed_list:
            closed_coords = np.array([(c.x, c.y) for c in closed_list])
            scatter_fechada_h2.set_offsets(closed_coords)
        else:
            scatter_fechada_h2.set_offsets(np.empty((0, 2)))
            
        if open_list:
            open_coords = np.array([(c.x, c.y) for c in open_list])
            scatter_aberta_h2.set_offsets(open_coords)
        else:
            scatter_aberta_h2.set_offsets(np.empty((0, 2)))
        artistas_retorno.extend([scatter_fechada_h2, scatter_aberta_h2])
        
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

print("Criando animação comparativa...")
ani = animation.FuncAnimation(fig, update, frames=gerador_combinado, 
                              blit=False, repeat=False, interval=200)

fig.suptitle("Comparação Visual: H1 (Fraca) vs. H2 (Forte / A* Informado)", fontsize=16)

fig.legend(handles=[patch_estrada, patch_terra, patch_lama, 
                    patch_barreira, patch_aberta, patch_fechada],
           loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.01))

fig.tight_layout(rect=[0, 0.1, 1, 0.95])

plt.show()

print("Animação concluída.")