# Surface-conditioned GA for Laves-assisted self-healing ferritic steels

This repository contains the Python implementation of a surface-conditioned genetic algorithm (GA) for the design and prioritisation of ferritic steel candidates consistent with Laves-phase-assisted self-healing.

The workflow combines machine-learning (ML) pre-screening, Monte Carlo (MC) surface-segregation calculations, Thermo-Calc-based go/no-go criteria and a coarsening-based fitness function. The objective is to compare bulk-dominant and surface-enhanced Laves precipitation pathways and to identify candidate alloy-processing conditions for subsequent experimental validation.

## Workflow

Each candidate alloy is encoded as composition and heat-treatment genes. After decoding, the workflow proceeds through:

1. RF-based ML pre-screening.
2. MC prediction of the surface-conditioned composition.
3. Thermo-Calc go/no-go evaluation of matrix stability, Laves availability and competing phases.
4. Driving-force comparison between bulk and surface-conditioned states.
5. Coarsening-based ranking using the predicted Laves radius after a 100 h ageing window.

The ML model is used only as a conservative rejection filter. Final candidate acceptance depends on MC surface conditioning, Thermo-Calc-derived quantities and the fitness function.

## Design variables

Fe is calculated by mass balance and C is fixed at 0.05 wt.%.

| Variable | Lower | Upper | Step | Unit |
|---|---:|---:|---:|---|
| Cr | 14.0 | 20.0 | 0.25 | wt.% |
| Mn | 0.0 | 2.0 | 0.1 | wt.% |
| Si | 0.0 | 2.0 | 0.1 | wt.% |
| W | 0.0 | 5.0 | 0.1 | wt.% |
| Mo | 0.0 | 5.0 | 0.1 | wt.% |
| Nb | 0.0 | 3.0 | 0.1 | wt.% |
| Ti | 0.0 | 3.0 | 0.1 | wt.% |
| Ni | 0.0 | 6.0 | 0.1 | wt.% |
| Cu | 0.0 | 3.0 | 0.1 | wt.% |
| T_sol | 900 | 1300 | 20 | °C |
| T_aging | 550 | 750 | 20 | °C |

The discretised search space contains approximately 1.19 × 10^16 candidate alloy-processing conditions.

## Repository structure

```text
config/      Configuration files
data/        Input data and ML models
ga_sh/       Main GA, encoding, go/no-go, MC and fitness modules
legacy_ga/   Earlier scripts retained for traceability
ml_df/       ML-related utilities
scripts/     Execution and diagnostic scripts
tests/       Test scripts
outputs/     Generated outputs, ignored by Git
```

## Installation

Create and activate a Python environment, then install the dependencies:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Full Thermo-Calc evaluation requires local Thermo-Calc Python API access and valid thermodynamic or mobility databases. Licensed database files are not included if restricted by licence.

## Outputs

Generated runs, caches and result files are written to `outputs/`, which is excluded from version control.

Typical outputs include:

- evaluated candidate tables;
- cached candidate evaluations;
- GA history files;
- convergence summaries;
- prioritised candidate lists.

## Scope

This repository provides a computational candidate-prioritisation framework. It does not directly simulate:

- creep-cavity nucleation;
- cavity growth;
- incubation time for Laves nucleation;
- complete precipitation kinetics;
- long-term creep rupture life;
- experimental self-healing efficiency.

The 100 h ageing window is intended for controlled surface-ageing studies. Future versions should include a kinetic incubation-time descriptor to distinguish premature matrix precipitation from damage-triggered surface or cavity precipitation.

## Reproducibility notes

For publication-quality runs, record:

- configuration file;
- random seed;
- Thermo-Calc version;
- thermodynamic and mobility database versions;
- commit hash;
- Python environment.

Exact numerical reproducibility may depend on Thermo-Calc version, database version, package versions and local system configuration.

## Citation

If using this repository, cite the associated thesis chapter or manuscript when available.

## Author

Diego Wackerling
