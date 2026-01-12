# AIRocket — Dicionário de variáveis do dataset

Este documento descreve as colunas dos arquivos CSV em `dataset/` (ex.: `rocket_design_dataset5000.csv`).

## Convenções e unidades

- O projeto usa majoritariamente **SI**.
- Comprimentos/raios: **m**
- Massa: **kg**
- Densidade: **kg/m³**
- Tempo: **s**
- Força/empuxo: **N**
- Impulso total: **N·s**
- Pressão dinâmica: **Pa** (N/m²)
- Números adimensionais: Mach, razões geométricas e parâmetros de curva.

> Resposta direta: **`grain_h` está em metros (m)** e representa a **altura/comprimento axial de cada grão** do propelente.
> Faixa típica no gerador: 0.05–0.15 m (veja `airrocket/design_space.py`).

## Colunas (entrada de design)

| Coluna | Significado | Unidade | Observações / como é usada |
|---|---:|---:|---|
| `id` | Identificador curto do design | — | UUID truncado (8 chars). |
| `num_grains` | Número de grãos (segmentos) do grão/propelente | — | Inteiro. |
| `grain_h` | Altura (comprimento axial) de **cada** grão | m | Volume do propelente: $V = \pi( r_o^2 - r_i^2 )\,h\,n$. |
| `grain_out_r` | Raio externo do grão | m | Deve ser > `grain_in_r`. |
| `grain_in_r` | Raio interno inicial do grão (canal) | m | Define o “web” $w = r_o - r_i$. |
| `throat_r` | Raio da garganta do bocal | m | Controla área de garganta e junto com `expansion_ratio` define saída. |
| `expansion_ratio` | Razão de expansão do bocal $\varepsilon = A_e/A_t$ | — | Em `derive.py`: $r_e = r_t\sqrt{\varepsilon}$. |
| `tube_l` | Comprimento do corpo/tubo (airframe) | m | Também limita `motor_casing_length`. |
| `nose_l` | Comprimento do nariz | m | Usado em `rocket.add_nose`. |
| `nose_type` | Tipo de nariz | — | Enum: `conical`, `ogive`, `vonKarman`. |
| `fin_n` | Número de aletas | — | Geralmente 3 ou 4. |
| `fin_root` | Corda na raiz da aleta | m | `root_chord`. |
| `fin_tip` | Corda na ponta da aleta | m | `tip_chord` (clamp para ≤ `fin_root`). |
| `fin_span` | Envergadura (span) da aleta | m | `span`. |
| `fin_sweep` | Varredura (sweep length) | m | `sweep_length` (>= 0). |
| `prop_density` | Densidade do propelente | kg/m³ | Usada para massa do propelente: $m = V\rho$. |
| `isp_eff_s` | Isp efetivo (modelo) | s | Entra no impulso total. |
| `eta_thrust` | Eficiência/fator de empuxo | — | Multiplicativo no impulso total. |
| `burn_rate` | Taxa de queima radial (modelo) | m/s | Em `motor_param.py`: $t_b = (r_o-r_i)/\dot r$. |
| `curve_alpha` | Parâmetro α da curva de empuxo (Beta) | — | Controla formato temporal do empuxo. |
| `curve_beta` | Parâmetro β da curva de empuxo (Beta) | — | Controla formato temporal do empuxo. |

## Colunas derivadas (calculadas)

| Coluna | Significado | Unidade | Observações / fórmula |
|---|---:|---:|---|
| `burn_time_s` | Tempo de queima | s | $t_b = \mathrm{clip}((r_o-r_i)/\dot r,\ 1.5,\ 6.0)$. |
| `prop_mass_kg` | Massa de propelente | kg | $m_p = \pi(r_o^2-r_i^2)\,h\,n\,\rho$. |
| `total_impulse_ns` | Impulso total | N·s | $I = \eta\,m_p\,g_0\,Isp$. |
| `nozzle_exit_r` | Raio de saída do bocal | m | $r_e = r_t\sqrt{\varepsilon}$. |
| `motor_dry_mass_kg` | Massa seca do motor (estimada) | kg | Modelo simples: $m_{dry} = 0.35\,m_p + 0.5$. |
| `max_thrust_n` | Empuxo máximo do perfil | N | Máximo da curva de empuxo gerada. |
| `thrust_tau05_n` | Empuxo em ~5% do tempo de queima | N | Ajuda a filtrar motores que “demoram a subir”. |
| `tube_d` | Diâmetro externo do foguete (derivado) | m | Em `derive.py`: $d = 2(r_o + clearance + wall)$. |
| `tube_r` | Raio do tubo | m | `tube_d / 2`. |
| `rocket_mass_est_kg` | Massa do foguete sem motor/prop (estimada) | kg | Modelo simples com massa por comprimento + payload. |
| `motor_casing_length` | Comprimento do invólucro do motor (estimado) | m | `num_grains*grain_h + (num_grains-1)*grain_sep + 2*end_margin`. |

## Colunas de saída da simulação de voo

| Coluna | Significado | Unidade | Observações |
|---|---:|---:|---|
| `apogee` | Apogeu | m | Altitude máxima retornada pelo RocketPy. |
| `max_mach` | Mach máximo | — | Adimensional. |
| `max_acceleration` | Aceleração máxima | m/s² | Conforme `flight.max_acceleration`. |
| `max_dynamic_pressure` | Pressão dinâmica máxima | Pa | Conforme `flight.max_dynamic_pressure`. |
| `rail_exit_speed` | Velocidade de saída do trilho | m/s | `flight.out_of_rail_velocity`. |
| `stability_initial` | Margem estática no início | calibers (diâmetros do corpo) | `rocket.static_margin(0)`. |
| `stability_min_over_flight` | Menor margem estática durante o voo | calibers | Fonte: `flight.min_stability_margin` (quando disponível). |
| `stability_at_max_q` | Margem estática no instante de $q$ máximo | calibers | Calculada via `rocket.static_margin(t_max_q)`. |
| `min_stability_margin` | Métrica redundante da margem mínima | calibers | Atualmente copia `flight.min_stability_margin`. |

## Colunas de status/qualidade

| Coluna | Significado | Unidade | Observações |
|---|---:|---:|---|
| `ok` | Se o design passou (constraints + voo) | — | Boolean. |
| `fail_reason` | Motivo da falha (quando `ok=False`) | — | Pode ser `no_liftoff_or_no_rail_exit`, `exception:...` ou lista `;` de constraints. |

### Principais `fail_reason` (constraints)

O validador em `airrocket/constraints.py` pode gerar, por exemplo:

- `grain_in_r>=grain_out_r`
- `tube_radius<=grain_out_r`
- `motor_casing_length>0.95*tube_l`
- `nozzle_exit_r>tube_radius`
- `insufficient_thrust_to_weight`
- `insufficient_early_thrust`

---

Se você quiser, eu também posso gerar automaticamente esse dicionário lendo o header do CSV e os símbolos do código (para não ficar manual) e incluir as colunas de Monte Carlo (`mc_*`) quando você habilita `mc_enabled=True` no gerador.
