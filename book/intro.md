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

```{code-cell} ipython3
:tags: [remove-input]
from IPython.display import HTML

# Developers! Add new cards to this list.
cards_data = [
    {"title": "🧱 Microstructure", "text": "Learn about capturing microstructural effects.", "link": "content/applications/microstructure.html", "image": "https://dummyimage.com/600x400/007bff/ffffff&text=Microstructure"},
    {"title": "🤖 Machine Learning", "text": "Machine learning applications in FESTIM.", "link": "content/applications/ml.html", "image": "https://dummyimage.com/600x400/28a745/ffffff&text=Machine+Learning"},
    {"title": "Card 3", "text": "This is the third card.", "link": "#", "image": "https://dummyimage.com/600x400/dc3545/ffffff&text=Card+3"},
    {"title": "Card 4", "text": "This is the fourth card.", "link": "#", "image": "https://dummyimage.com/600x400/ffc107/ffffff&text=Card+4"},
    {"title": "Card 5", "text": "This is the fifth card.", "link": "#", "image": "https://dummyimage.com/600x400/17a2b8/ffffff&text=Card+5"}
]

# Standard formatting for the carousel and cards
cards_per_slide = 3

html = '''<div id="carouselExampleAutoplaying" class="carousel slide custom-carousel" data-bs-ride="carousel">
  <div class="carousel-inner" style="padding: 2rem 10%;">'''

for i in range(0, len(cards_data), cards_per_slide):
    active_class = " active" if i == 0 else ""
    html += f'\n    <div class="carousel-item{active_class}">\n      <div class="row align-items-stretch">'
    
    for j in range(cards_per_slide):
        if i + j < len(cards_data):
            card = cards_data[i + j]
            html += f'''
        <div class="col-md-4 mb-3 d-flex">
          <div class="card text-center w-100" style="background-color: var(--pst-color-surface); border-color: var(--pst-color-border);">
            <img src="{card['image']}" class="card-img-top" alt="{card['title']}" style="height: 150px; object-fit: cover;">
            <div class="card-body">
              <h5 class="card-title" style="color: var(--pst-color-text-base);">{card['title']}</h5>
              <p class="card-text" style="color: var(--pst-color-text-muted);">{card['text']}</p>
              <a href="{card['link']}" class="stretched-link"></a>
            </div>
          </div>
        </div>'''
            
    html += '\n      </div>\n    </div>'

html += '''
  </div>
  <button class="carousel-control-prev" type="button" data-bs-target="#carouselExampleAutoplaying" data-bs-slide="prev" style="width: 10%;">
    <span class="carousel-control-prev-icon" aria-hidden="true" style="filter: drop-shadow(0 0 1px var(--pst-color-text-base));"></span>
    <span class="visually-hidden">Previous</span>
  </button>
  <button class="carousel-control-next" type="button" data-bs-target="#carouselExampleAutoplaying" data-bs-slide="next" style="width: 10%;">
    <span class="carousel-control-next-icon" aria-hidden="true" style="filter: drop-shadow(0 0 1px var(--pst-color-text-base));"></span>
    <span class="visually-hidden">Next</span>
  </button>
</div>'''

HTML(html)
```

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