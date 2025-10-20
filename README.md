PRECISA APENAS INSTALAR O MATPLOTLIB E O NUMPY:

abre o terminal e:
```
python -m pip install matplotlib numpy
```
No terminal dentro da pasta dos dois arquivos:
 ```
	python teste cavalo.py
 ```

Instalação, configuração e execução (mínimo)

Pré-requisito: Python 3.8+ instalado.

1) Criar ambiente virtual e instalar dependências ( dentro da pasta do projeto):

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

2) Executar validação rápida (sem interface gráfica):

```powershell
.\.venv\Scripts\python.exe run_validate_direct.py
```

3) Executar visualização (interativa):

```powershell
.\.venv\Scripts\python.exe src\main.py
```

4) Executar em modo headless (salva imagem e métricas em `output/`):

```powershell
.\.venv\Scripts\python.exe src\main.py --headless
```

Arquivos úteis:
- `src/astar.py` — implementação do A* e heurísticas.
- `src/maps.py` — mapas de exemplo.
- `src/main.py` — visualização (use `--headless` para salvar).
- `run_validate_direct.py` — runner sem GUI para métricas.
