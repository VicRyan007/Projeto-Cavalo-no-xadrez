import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
# Importa a nova classe 'BuscaAEstrelaAnimada'
from busca_a_estrela import Tabuleiro, BuscaAEstrelaAnimada 
from matplotlib.colors import ListedColormap, BoundaryNorm

# --- DEFINIÇÃO DO TABULEIRO (mesmo de antes) ---
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

# --- INICIALIZA O GERADOR DA BUSCA ---
# Vamos animar a H2 (Forte) para ver a busca focada
# Para ver a H1 (Dijkstra), mude para 'h1'
print("Inicializando gerador da busca (H2)...")
busca_animada = BuscaAEstrelaAnimada(tabuleiro, destino)
gerador_busca = busca_animada.busca_passo_a_passo(origem, tipo_heuristica='h2')

# --- CONFIGURAÇÃO DA ANIMAÇÃO ---
fig, ax = plt.subplots(figsize=(8, 8))

# 1. Prepara o mapa de terreno (base)
largura = tabuleiro.largura
altura = tabuleiro.altura
mapa_terreno = np.zeros((altura, largura))
for y in range(altura):
    for x in range(largura):
        mapa_terreno[y, x] = tabuleiro.get_casa(x, y).custo_terreno
mapa_terreno[mapa_terreno == np.inf] = -1

# 2. Desenha o mapa de terreno base
cmap_terreno = plt.cm.get_cmap('terrain_r')
cmap_terreno.set_under('black') # Barreiras
ax.imshow(mapa_terreno, cmap=cmap_terreno, vmin=0)

# 3. Prepara matrizes de cores para Open e Closed lists
# Usaremos 'nan' para vazio, 2 para Fechada, 3 para Aberta
mapa_listas = np.full((altura, largura), np.nan)

# 4. cmap customizado para as listas
# Laranja (Lista Fechada), Verde (Lista Aberta)
cmap_listas = ListedColormap(['#FF8C0090', '#00FF0090']) 
norm = BoundaryNorm([1.5, 2.5, 3.5], cmap_listas.N)

# 5. Objetos de imagem que serão atualizados
# Começa com dados vazios
im_listas = ax.imshow(mapa_listas, cmap=cmap_listas, norm=norm, interpolation='none')

# Esta linha será atualizada a cada frame
linha_caminho, = ax.plot([], [], marker='o', color='red', 
                         markersize=5, linewidth=2, label='Caminho Atual')

# 6. Texto de status
status_text = ax.text(0.01, 0.01, '', transform=ax.transAxes, 
                        color='white', backgroundcolor='black', fontsize=12)

# --- FUNÇÃO DE ATUALIZAÇÃO (para cada frame) ---
def update(frame_data):
    open_list, closed_list, caminho_atual, terminado = frame_data
    
    # Limpa a matriz de listas
    mapa_listas.fill(np.nan)
    
    # Pinta a Lista Fechada (Valor 2 = Laranja)
    for casa in closed_list:
        mapa_listas[casa.y, casa.x] = 2
    
    # Pinta a Lista Aberta (Valor 3 = Verde)
    for casa in open_list:
        mapa_listas[casa.y, casa.x] = 3
        
    # Atualiza a imagem das listas
    im_listas.set_data(mapa_listas)

    # Atualiza os dados (posição) da linha do caminho
    if caminho_atual:
        x_coords = [casa.x for casa in caminho_atual]
        y_coords = [casa.y for casa in caminho_atual]
        linha_caminho.set_data(x_coords, y_coords)
    else:
        # Se não houver caminho (ex: frame inicial antes da busca),
        # define a linha como vazia
        linha_caminho.set_data([], [])

    # <<< MUDANÇA AQUI: Atualiza os dados da linha do caminho >>>
    if terminado:
        if caminho_atual:
            status_text.set_text(f'CAMINHO ENCONTRADO!\n'
                                 f'Nós Expandidos: {len(closed_list)}')
            linha_caminho.set_color('lime') # Muda a cor no final
        else:
            status_text.set_text(f'CAMINHO NÃO ENCONTRADO!\n'
                                 f'Nós Expandidos: {len(closed_list)}')
    else:
        status_text.set_text(f'Nós Expandidos (Fechada): {len(closed_list)}\n'
                             f'Nós na Fila (Aberta): {len(open_list)}')
    return [im_listas, status_text, linha_caminho]
    

# --- RODA A ANIMAÇÃO ---
print("Criando animação...")
# interval=50 -> 50ms por frame. Aumente para 200 para mais lento.
ani = animation.FuncAnimation(fig, update, frames=gerador_busca, 
                              blit=True, repeat=False, interval=200)

ax.set_title("Visualização Dinâmica A* (H2)\nVerde=Lista Aberta, Laranja=Lista Fechada")
plt.show()

print("Animação concluída.")