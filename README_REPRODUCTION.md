# H-ACO sobre GRETEL — reproducción de los experimentos de la tesis

Fork de GRETEL con el explicador `HypercubeACOExplainer` (ACO sobre hipercubo
binario para explicaciones contrafactuales en grafos). Release citada en la
tesis: **`v1.0-tesis`**.

## Entorno
```bash
conda env create -f environment.yml   # crea el entorno GRTL (Python 3.9.23)
conda activate GRTL
# alternativa reproducible: usar el .devcontainer / dockerfile incluidos
```

## Datos
- `TreeCycles`: se genera al vuelo (`TreeCyclesRand`, 128 instancias, 28 nodos).
- `ASD`: incluido en `data/datasets/autism/` (49 `asd/` + 52 `td/` = 101 grafos;
  conectividad funcional de-identificada del consorcio público ABIDE).

## Cómo reproducir cada tabla del Capítulo 4
| Tabla | Bloque | Driver / notebook |
|------|--------|-------------------|
| 4.3 | Sensibilidad a `beta_compatibility` (TreeCycles) | `lab/.../ACO/hypercube_aco_pipeline_treecycles.jsonc` (β ∈ {0,1,2,3}) |
| 4.4 | Comparación con referencias (TreeCycles) | idem, con OBS/IRand/CF2/ACO |
| 4.5 | Comparación principal (ASD, 3 semillas) | `lab/.../ACO/hypercube_aco_pipeline_asd.jsonc` |
| 4.6–4.8 | Presupuesto de consultas (ASD) | idem + `max_oracle_calls` ∈ {350,500,800} |
| Ablación | Refinamiento on/off | idem + `enable_backward_refinement: false` |
| Paralelización | `num_workers` ∈ {1,4} | idem (subconjunto de 10 casos) |
| Caché | Aciertos de caché (101 casos) | idem, leyendo `HypercubeACOExplainer._cache_log` |

Ejecutar mediante los notebooks `lab/evaluation_pipeline_hypercube_ACO*.ipynb`.

## Ablación del refinamiento
El parámetro `enable_backward_refinement` (por defecto `true`) activa/desactiva
la fase de refinamiento hacia atrás bajo el mismo límite de llamadas al oráculo.

## Métricas de caché
`HypercubeACOExplainer._cache_log` acumula `(cache_hits, oracle_calls_used)` por
instancia (a través de todas las instancias del experimento). Limpiar con
`HypercubeACOExplainer._cache_log.clear()` antes de cada corrida; al terminar,
promediar los `cache_hits`, y calcular tasa global = Σhits / Σ(hits+calls) y
tasa media por instancia = media de hits/(hits+calls).