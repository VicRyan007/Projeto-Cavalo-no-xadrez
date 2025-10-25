import heapq
from collections import deque
import random

TERRAIN_VALUES = {
    'estrada': 0.5,
    'terra': 1.0,
    'lama': 5.0,
    'barreira': float('inf')
}

def gerar_matriz_aleatoria(largura=8, altura=8, probs=None):
    if probs is None:
        probs = {'estrada': 0.15, 'terra': 0.7, 'lama': 0.12, 'barreira': 0.03}
    tipos = list(probs.keys())
    pesos = [probs[t] for t in tipos]
    matriz = []
    for y in range(altura):
        linha = []
        for x in range(largura):
            tipo = random.choices(tipos, weights=pesos, k=1)[0]
            linha.append(TERRAIN_VALUES[tipo])
        matriz.append(linha)
    return matriz

class Casa:
    def __init__(self, x, y, custo_terreno):
        self.x = x
        self.y = y
        self.custo_terreno = custo_terreno
        self.visitado = False

    def __str__(self):
        return f'({self.x}, {self.y})'
    
    def __repr__(self):
        return f'Casa({self.x}, {self.y}, Custo:{self.custo_terreno})'

    def __lt__(self, other):
        return (self.x, self.y) < (other.x, other.y)

class Tabuleiro:
    def __init__(self, matriz_custos):
        self.altura = len(matriz_custos)
        self.largura = len(matriz_custos[0])
        
        self.min_custo_terreno = float('inf')
        for linha in matriz_custos:
            for custo in linha:
                if custo < self.min_custo_terreno:
                    self.min_custo_terreno = custo

        self.grid = []
        for y in range(self.altura):
            linha_grid = []
            for x in range(self.largura):
                linha_grid.append(Casa(x, y, matriz_custos[y][x]))
            self.grid.append(linha_grid)
            
        self.movimentos_cavalo = [
            (1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2)
        ]

    def get_casa(self, x, y):
        if 0 <= x < self.largura and 0 <= y < self.altura:
            return self.grid[y][x]
        return None

    def get_vizinhos(self, casa):
        vizinhos = []
        for dx, dy in self.movimentos_cavalo:
            novo_x, novo_y = casa.x + dx, casa.y + dy
            vizinho_casa = self.get_casa(novo_x, novo_y)
            if vizinho_casa and vizinho_casa.custo_terreno != float('inf'):
                vizinhos.append(vizinho_casa)
        return vizinhos

    def limpar_tabuleiro(self):
        for y in range(self.altura):
            for x in range(self.largura):
                self.grid[y][x].visitado = False

class BuscaAEstrela:
    
    def _criar_tabela_saltos_bfs(self, largura, altura):
        distancias = [[-1 for _ in range(largura)] for _ in range(altura)]
        fila = deque([(0, 0, 0)])
        distancias[0][0] = 0
        
        movimentos_cavalo = [
            (1, 2), (2, 1), (2, -1), (1, -2),
            (-1, -2), (-2, -1), (-2, 1), (-1, 2)
        ]
        
        while fila:
            x, y, dist = fila.popleft()
            for dx_mov, dy_mov in movimentos_cavalo:
                nx, ny = x + dx_mov, y + dy_mov
                if 0 <= nx < largura and 0 <= ny < altura:
                    if distancias[ny][nx] == -1:
                        distancias[ny][nx] = dist + 1
                        fila.append((nx, ny, dist + 1))
        return distancias

    def __init__(self, tabuleiro, objetivo):
        self.tabuleiro = tabuleiro
        self.objetivo = objetivo
        self.encontrado = False
        self.tabela_saltos_cavalo = self._criar_tabela_saltos_bfs(
            tabuleiro.largura, tabuleiro.altura
        )

    # --- Heurística H1 Dijkstra ---
    def h1_fraca(self, casa_atual):
        return 0.0 

    # --- Heurística H2 baseado no movimento do cavalo ---
    def h2_forte(self, casa_atual):
        dx = abs(casa_atual.x - self.objetivo.x)
        dy = abs(casa_atual.y - self.objetivo.y)
        
        num_min_saltos = self.tabela_saltos_cavalo[dy][dx]

        return num_min_saltos * self.tabuleiro.min_custo_terreno


    # --- Algoritmo A* ---
    def busca_a_estrela(self, casa_origem, tipo_heuristica='h1'):
        self.tabuleiro.limpar_tabuleiro()
        self.encontrado = False
        
        closed_list = set() 

        if tipo_heuristica == 'h1':
            calcular_h = self.h1_fraca
        else:
            calcular_h = self.h2_forte

        g_inicial = 0 
        h_inicial = calcular_h(casa_origem)
        f_inicial = g_inicial + h_inicial

        fila_prioridade = [(f_inicial, g_inicial, casa_origem, [casa_origem])]
        custo_g_visitado = {casa_origem: 0} 

        while fila_prioridade:
            f_score, g_atual, casa_atual, caminho = heapq.heappop(fila_prioridade)
            
            closed_list.add(casa_atual)

            if casa_atual == self.objetivo:
                self.encontrado = True
                return caminho, g_atual, closed_list, custo_g_visitado

            if g_atual > custo_g_visitado.get(casa_atual, float('inf')):
                continue

            for vizinho in self.tabuleiro.get_vizinhos(casa_atual):
                novo_custo_g = g_atual + vizinho.custo_terreno
                if novo_custo_g < custo_g_visitado.get(vizinho, float('inf')):
                    custo_g_visitado[vizinho] = novo_custo_g
                    custo_h = calcular_h(vizinho)
                    f_score_vizinho = novo_custo_g + custo_h
                    novo_caminho = caminho + [vizinho]
                    heapq.heappush(fila_prioridade,
                        (f_score_vizinho, novo_custo_g, vizinho, novo_caminho))

        return None, 0, closed_list, custo_g_visitado

# --- ANIMAÇÃO ---
class BuscaAEstrelaAnimada(BuscaAEstrela):
    
    def busca_passo_a_passo(self, casa_origem, tipo_heuristica='h1'):
        self.tabuleiro.limpar_tabuleiro()

        if tipo_heuristica == 'h1':
            calcular_h = self.h1_fraca
        else:
            calcular_h = self.h2_forte

        closed_list = set()
        custo_g_visitado = {casa_origem: 0}
        
        g_inicial = 0 
        h_inicial = calcular_h(casa_origem)
        f_inicial = g_inicial + h_inicial
        
        fila_prioridade = [(f_inicial, g_inicial, casa_origem, [casa_origem])]
        open_list_dict = {casa_origem: f_inicial}

        yield set(open_list_dict.keys()), set(closed_list), [casa_origem], False

        while fila_prioridade:
            f_score, g_atual, casa_atual, caminho = heapq.heappop(fila_prioridade)
            
            if casa_atual in open_list_dict:
                del open_list_dict[casa_atual]

            if g_atual > custo_g_visitado.get(casa_atual, float('inf')):
                continue

            closed_list.add(casa_atual)

            yield set(open_list_dict.keys()), set(closed_list), caminho, False

            if casa_atual == self.objetivo:
                yield set(open_list_dict.keys()), set(closed_list), caminho, True 
                return 

            for vizinho in self.tabuleiro.get_vizinhos(casa_atual):
                if vizinho in closed_list:
                    continue
                
                novo_custo_g = g_atual + vizinho.custo_terreno
                
                if novo_custo_g >= custo_g_visitado.get(vizinho, float('inf')):
                    continue

                custo_g_visitado[vizinho] = novo_custo_g
                custo_h = calcular_h(vizinho)
                f_score_vizinho = novo_custo_g + custo_h
                novo_caminho = caminho + [vizinho]
                
                heapq.heappush(fila_prioridade,
                    (f_score_vizinho, novo_custo_g, vizinho, novo_caminho))
                
                open_list_dict[vizinho] = f_score_vizinho

        yield set(open_list_dict.keys()), set(closed_list), None, True