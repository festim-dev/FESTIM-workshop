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
    {"title": "Microstructure", "text": "Learn about capturing microstructural effects.", "link": "content/applications/microstructure.html", "image": "_static/microstructure.png"},
    {"title": "Plasma Facing Components", "text": "", "link": "#", "image": "_static/monoblock.png"},
    {"title": "TDS analysis", "text": "Simulate thermo-desorption experiments.", "link": "content/applications/task02.html", "image": "_static/tds.png"},
    {"title": "TDS fit", "text": "Automatically fit a TDS", "link": "content/applications/task10.html", "image": "_static/fitting_tds.png"},
    {"title": "Machine Learning", "text": "Machine learning applications in FESTIM.", "link": "content/applications/ml.html", "image": "https://dummyimage.com/600x400/28a745/ffffff&text=Machine+Learning"},
]

# Standard formatting for the carousel and cards
cards_per_slide = 3

html = '''
<style>
/* Fix to remove the white background from the docutils jupyter cell container in dark mode */
.cell_output:has(.custom-carousel), .output.text_html:has(.custom-carousel) {
    background-color: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

.custom-carousel-card {
    transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
}
.custom-carousel-card:hover {
    transform: scale(1.05);
    box-shadow: 0 10px 20px rgba(0,0,0,0.15);
    z-index: 10;
}
.custom-carousel-img-container {
    height: 200px; 
    width: 100%; 
    display: flex; 
    align-items: center; 
    justify-content: center;
    padding: 0.5rem;
    background-color: var(--pst-color-surface);
}
.custom-carousel-img-container img {
    max-height: 100%;
    max-width: 100%;
    object-fit: contain;
}
</style>

<script>
// Fallback for older browsers to ensure cell background is transparent
document.addEventListener("DOMContentLoaded", function() {
    let carousels = document.querySelectorAll('.custom-carousel');
    carousels.forEach(c => {
        let cellOutput = c.closest('.cell_output');
        if (cellOutput) {
            cellOutput.style.backgroundColor = 'transparent';
            cellOutput.style.border = 'none';
            cellOutput.style.boxShadow = 'none';
        }
        let textHtml = c.closest('.output.text_html');
        if (textHtml) {
            textHtml.style.backgroundColor = 'transparent';
            textHtml.style.border = 'none';
            textHtml.style.boxShadow = 'none';
        }
    });
});
</script>

<div id="carouselExampleAutoplaying" class="carousel slide custom-carousel" data-bs-ride="carousel" style="background-color: var(--pst-color-background); border-radius: 0.5rem;">
  <div class="carousel-inner" style="padding: 2rem 10%;">'''

for i in range(0, len(cards_data), cards_per_slide):
    active_class = " active" if i == 0 else ""
    html += f'\n    <div class="carousel-item{active_class}">\n      <div class="row align-items-stretch" style="min-height: 280px;">'
    
    for j in range(cards_per_slide):
        if i + j < len(cards_data):
            card = cards_data[i + j]
            html += f'''
        <div class="col-md-4 mb-3 d-flex">
          <div class="card text-center w-100 custom-carousel-card" style="background-color: var(--pst-color-surface); border-color: var(--pst-color-border);">
            <div class="custom-carousel-img-container">
              <img src="{card['image']}" class="card-img-top" alt="{card['title']}">
            </div>
            <div class="card-body" style="padding: 0.5rem; display: flex; flex-direction: column; justify-content: center;">
              <h5 class="card-title" style="color: var(--pst-color-text-base); font-size: 1rem; margin-bottom: 0.2rem;">{card['title']}</h5>
              <p class="card-text" style="color: var(--pst-color-text-muted); font-size: 0.8rem; margin-bottom: 0;">{card['text']}</p>
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