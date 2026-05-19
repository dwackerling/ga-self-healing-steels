from __future__ import annotations

import importlib
import sys
from pathlib import Path


def check_import(name: str) -> None:
    module = importlib.import_module(name)
    version = getattr(module, "__version__", "unknown")
    print(f"{name}: {version}")


def main() -> None:
    print("Python:", sys.version)

    for name in [
        "numpy",
        "pandas",
        "yaml",
        "joblib",
        "sklearn",
        "tqdm",
        "deap",
    ]:
        check_import(name)

    try:
        import tc_python
        print("tc_python: import OK")
    except Exception as exc:
        raise RuntimeError(f"tc_python import failed: {exc!r}") from exc

    project_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(project_root / "ga_sh"))

    from config import load_config
    from ml_filter import load_models

    cfg = load_config(project_root / "config" / "config.yaml")

    bulk_model, surf_model = load_models(
        bulk_model_path=cfg["ml"]["bulk_model_path"],
        surf_model_path=cfg["ml"]["surface_model_path"],
    )

    print("bulk_model:", type(bulk_model))
    print("surf_model:", type(surf_model))
    print("Environment smoke test OK")


if __name__ == "__main__":
    main()