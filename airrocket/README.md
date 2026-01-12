# airrocket

Módulos compartilhados (reutilizáveis) entre:
- geração de dataset (RocketPy)
- otimização (Fast/Hard)

**Ideia:** a simulação e as derivações/constraints ficam aqui para evitar divergência entre quem gera o banco e quem otimiza.

Arquivos (primeira versão):
- `env.py`: criação/caching do `Environment`
- `design_space.py`: amostragem do DNA (parâmetros livres)
- `derive.py`: parâmetros derivados (ex.: `tube_d` derivado)
- `motor_param.py`: motor paramétrico + geração de curva de empuxo sintética
- `constraints.py`: validações físicas básicas (hard fail)
- `simulate.py`: build do motor/foguete e extração de métricas do `Flight`
- `dataset.py`: helpers de RNG/ID/amostragem
