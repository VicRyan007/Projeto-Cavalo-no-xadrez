import heapq
import math
from collections import deque

# --- 1. CLASSE 'CASA' (Seu "Estado" ou "Vértice") ---
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


# --- 2. CLASSE 'TABULEIRO' (Seu "Grafo Implícito") ---
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


# --- 3. CLASSE 'BuscaAEstrela' (Lógica Principal) ---
class BuscaAEstrela:
    
    # --- Cria tabela de saltos mínimos do cavalo (BFS) ---
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
        # Pré-calcula a tabela de saltos do cavalo (para H2)
        self.tabela_saltos_cavalo = self._criar_tabela_saltos_bfs(
            tabuleiro.largura, tabuleiro.altura
        )

    # --- Heurística H1 (Fraca, Genérica, ADMISSÍVEL) ---
    def h1_fraca(self, casa_atual):
        """
        H1: Heurística Fraca (Busca de Custo Uniforme / Dijkstra)
        H = 0. Esta é a heurística "ignorante" mais fraca possível.
        É admissível (0 <= Custo Real).
        """
        return 0.0 # Retorna sempre 0


    # --- Heurística H2 (Forte, Específica, ADMISSÍVEL) ---
    def h2_forte(self, casa_atual):
        """
        H2: Heurística Forte (Baseada nos saltos reais do Cavalo)
        Usa a tabela pré-calculada pelo BFS.
        """
        dx = abs(casa_atual.x - self.objetivo.x)
        dy = abs(casa_atual.y - self.objetivo.y)
        
        # 1. Pega o N° Mínimo de Saltos da tabela (BFS)
        num_min_saltos = self.tabela_saltos_cavalo[dy][dx]

        # 2. Multiplica pelo CUSTO MÍNIMO para garantir admissibilidade
        return num_min_saltos * self.tabuleiro.min_custo_terreno


    # --- Algoritmo A* ---
    def busca_a_estrela(self, casa_origem, tipo_heuristica='h1'):
        self.tabuleiro.limpar_tabuleiro()
        self.encontrado = False
        
        # <<< MUDANÇA 1: Mudar de contador para um CONJUNTO >>>
        closed_list = set() # Substitui 'nos_expandidos = 0'

        if tipo_heuristica == 'h1':
            calcular_h = self.h1_fraca
        else:
            calcular_h = self.h2_forte

        g_inicial = 0 
        h_inicial = calcular_h(casa_origem)
        f_inicial = g_inicial + h_inicial

        fila_prioridade = [(f_inicial, g_inicial, casa_origem, [casa_origem])]
        custo_g_visitado = {casa_origem: 0} # Este mapa é crucial para o Req A.C

        while fila_prioridade:
            f_score, g_atual, casa_atual, caminho = heapq.heappop(fila_prioridade)
            
            # <<< MUDANÇA 2: Adiciona a casa à lista fechada >>>
            closed_list.add(casa_atual)

            if casa_atual == self.objetivo:
                self.encontrado = True
                # <<< MUDANÇA 3: Retorna os novos dados >>>
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

        # <<< MUDANÇA 4: Retorna os novos dados (em caso de falha) >>>
        return None, 0, closed_list, custo_g_visitado