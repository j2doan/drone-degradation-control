"""Execute the baseline notebook with this project's Python and save outputs."""

import argparse
import asyncio
import os
from pathlib import Path
import sys


def main():
    project = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, nargs="?", default=project / "notebook.ipynb")
    args = parser.parse_args()
    for variable, relative in {
        "JUPYTER_CONFIG_DIR": ".cache/jupyter/config",
        "JUPYTER_DATA_DIR": ".cache/jupyter/data",
        "JUPYTER_RUNTIME_DIR": ".cache/jupyter/runtime",
        "IPYTHONDIR": ".cache/ipython",
    }.items():
        location = project / relative
        location.mkdir(parents=True, exist_ok=True)
        os.environ[variable] = str(location)

    try:
        import nbformat
        from nbclient import NotebookClient
        from jupyter_client import AsyncKernelManager
    except ImportError as error:
        raise SystemExit("Install notebook dependencies: .venv/bin/python -m pip install -r requirements-notebook.txt") from error

    notebook = nbformat.read(args.path, as_version=4)
    nbformat.validate(notebook)
    manager = AsyncKernelManager(kernel_name="python3")
    manager.kernel_spec.argv[0] = sys.executable
    client = NotebookClient(notebook, km=manager, timeout=90, resources={"metadata": {"path": str(args.path.resolve().parent)}})
    try:
        client.execute(cleanup_kc=True)
    finally:
        if manager.has_kernel:
            asyncio.run(manager.shutdown_kernel(now=True))
    nbformat.validate(notebook)
    nbformat.write(notebook, args.path)
    cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    figures = sum("image/png" in output.get("data", {}) for cell in cells for output in cell.outputs)
    print(f"Executed {len(cells)} code cells; {figures} embedded figures; no errors.")
    print(args.path)


if __name__ == "__main__":
    main()
