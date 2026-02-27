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

<div id="carouselExampleAutoplaying" class="carousel slide custom-carousel" data-bs-ride="carousel">
  <div class="carousel-inner" style="padding: 2rem 10%;">
    <div class="carousel-item active">
      <div class="row align-items-center">
        <div class="col-md-4">
          <div class="card text-center" style="height: 100%; background-color: var(--pst-color-surface); border-color: var(--pst-color-border);">
            <div class="card-body">
              <h5 class="card-title" style="color: var(--pst-color-text-base);">Card 1</h5>
              <p class="card-text" style="color: var(--pst-color-text-muted);">This is the first card</p>
            </div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="card text-center" style="height: 100%; background-color: var(--pst-color-surface); border-color: var(--pst-color-border);">
            <div class="card-body">
              <h5 class="card-title" style="color: var(--pst-color-text-base);">Card 2</h5>
              <p class="card-text" style="color: var(--pst-color-text-muted);">This is the second card</p>
            </div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="card text-center" style="height: 100%; background-color: var(--pst-color-surface); border-color: var(--pst-color-border);">
            <div class="card-body">
              <h5 class="card-title" style="color: var(--pst-color-text-base);">Card 3</h5>
              <p class="card-text" style="color: var(--pst-color-text-muted);">This is the third card</p>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="carousel-item">
      <div class="row align-items-center">
        <div class="col-md-4">
          <div class="card text-center" style="height: 100%; background-color: var(--pst-color-surface); border-color: var(--pst-color-border);">
            <div class="card-body">
              <h5 class="card-title" style="color: var(--pst-color-text-base);">Card 4</h5>
              <p class="card-text" style="color: var(--pst-color-text-muted);">This is the fourth card</p>
            </div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="card text-center" style="height: 100%; background-color: var(--pst-color-surface); border-color: var(--pst-color-border);">
            <div class="card-body">
              <h5 class="card-title" style="color: var(--pst-color-text-base);">Card 5</h5>
              <p class="card-text" style="color: var(--pst-color-text-muted);">This is the fifth card</p>
            </div>
          </div>
        </div>
        <div class="col-md-4">
          <div class="card text-center" style="height: 100%; background-color: var(--pst-color-surface); border-color: var(--pst-color-border);">
            <div class="card-body">
              <h5 class="card-title" style="color: var(--pst-color-text-base);">Card 6</h5>
              <p class="card-text" style="color: var(--pst-color-text-muted);">This is the sixth card</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
  <button class="carousel-control-prev" type="button" data-bs-target="#carouselExampleAutoplaying" data-bs-slide="prev" style="width: 10%;">
    <span class="carousel-control-prev-icon" aria-hidden="true" style="filter: drop-shadow(0 0 1px var(--pst-color-text-base));"></span>
    <span class="visually-hidden">Previous</span>
  </button>
  <button class="carousel-control-next" type="button" data-bs-target="#carouselExampleAutoplaying" data-bs-slide="next" style="width: 10%;">
    <span class="carousel-control-next-icon" aria-hidden="true" style="filter: drop-shadow(0 0 1px var(--pst-color-text-base));"></span>
    <span class="visually-hidden">Next</span>
  </button>
</div>

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