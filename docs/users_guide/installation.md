---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---
# Installing VectoRose

Installing VectoRose is very straightforward. You must have Python version
3.10 or higher. VectoRose can be installed from PyPI using `pip`.

```bash
pip install vectorose
```

This should be sufficient for most uses. Alternatively, you can download
the wheel from our GitHub releases page.

You can make sure that VectoRose was properly installed by opening a Python
console:

```{code-cell} ipython3
import vectorose as vr

print(vr.__version__)
```

If there is a `ModuleNotFoundError`, try restarting your Python console. If
it still does not work, then try reinstalling it. If there are error
messages that appear when installing or importing, please read them and
share them on [GitHub Issues](https://github.com/bzrudski/vectorose/issues)
so that we can try to resolve them.

## Jupyter Setup

You can easily use VectoRose in [Jupyter Lab](https://jupyter.org/) to
visualise results interactively. Make sure that you have `jupyter` and
`jupyterlab` installed. Try to run your code in Jupyter Lab. If it does not
work, add the following to the first cell in your notebook:

```python
import pyvista as pv
pv.start_xvfb()
pv.set_jupyter_backend("trame")
```

```{attention}
The line `pv.start_xvfb()`{l=python} is only required if running on a
Unix-like or Unix-based operating system (Linux or macOS). For techincal
reasons, it is not required when running on Windows (and will, in fact,
produce an error).
```

For more details on using the PyVista plotting packing with Jupyter Lab,
check out [this page](https://docs.pyvista.org/user-guide/jupyter/).

## Development Setup

To set up VectoRose for development, clone our GitHub repository:

```bash
git clone https://github.com/bzrudski/vectorose.git
```

We use [Poetry](https://python-poetry.org/) to manage our dependencies and
to help with package development. To install Poetry, read the [online
documentation](https://python-poetry.org/docs/#installation).

We also suggest creating a new environment to store all the dependencies
for VectoRose. This may be a virtual environment, a Conda environment, a
Poetry environment or any equivalent.

With this new environment activated, switch into the cloned git repository
and install all the dependencies using:

```bash
poetry install
```

All dependencies, as well as VectoRose itself, should now be installed in
your environment.

## Packaging Acknowledgement

A great resource that we used to develop VectoRose is *Python Packages* by
{cite:t}`beuzenPythonPackages2022`. We recommend consulting this book for
any assistance required on the basics of developing Python packages.
