---
jupytext:
  formats: ipynb,md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.18.1
kernelspec:
  display_name: festim-workshop
  language: python
  name: python3
---


# The FESTIM tutorial

Welcome to the FESTIM tutorial!

Comments and corrections to this webpage should be submitted to the issue tracker by going to the relevant page in the tutorial, then click the {fab}`github` repository symbol in the top right corner and either {fas}`lightbulb` “open issue” or {fas}`pencil` "suggest edit".

## Interactive tutorials

You don't have to install FESTIM locally to be able to run these examples yourself. 

Press the {fas}`rocket` button on the toolbar, then the {fas}`play` button to edit and run the code.

```{note}
This might take a while to load after new releases of the book.
```

```{code-cell} ipython3
import festim as F

print(F.__version__)
```

## Clickable API links

You can directly click modules, functions, and classes to have access to their API documentation.

```{code-cell} ipython3
from festim import Mesh1D  # click Mesh1D
import matplotlib  # click matplotlib
from dolfinx import fem  # click fem
```