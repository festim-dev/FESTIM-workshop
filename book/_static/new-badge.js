// Adds a "New!" badge next to selected pages and sections of the tutorial.
//
// Paths are relative to the book root and carry no file extension, i.e. exactly
// the path used in _toc.yml. Delete an entry once it is no longer new. A path
// that matches nothing logs a warning in the browser console.

// Badges the page's entry in the left-hand navigation sidebar.
const NEW_PAGES = [
  "content/post_process/intro",
  "content/applications/ml",
];

// Badges a single section at its heading and in the right-hand "Contents" panel,
// and badges the page it lives on in the sidebar too, so the section is visible
// before you are already on the page. Listing the page in NEW_PAGES as well is
// therefore redundant, though harmless.
//
// Add "#" and the section anchor, which you can copy from the "#" permalink
// shown next to the heading.
const NEW_SECTIONS = [
  "content/post_process/derived#post-processing-of-derived-quantities",
];

const makeBadge = () => {
  const badge = document.createElement("span");
  badge.className = "new-badge";
  badge.textContent = "New!";
  return badge;
};

const pagePath = (page) => `/${page.replace(/^\/+/, "")}.html`;

document.addEventListener("DOMContentLoaded", () => {
  // A new section implies a new page. The Set keeps a page that is listed twice,
  // or that holds several new sections, down to a single sidebar badge.
  const sidebarPages = new Set([
    ...NEW_PAGES,
    ...NEW_SECTIONS.map((entry) => entry.split("#")[0]),
  ]);

  const sidebarLinks = [...document.querySelectorAll("nav.bd-docs-nav a.reference.internal")];

  sidebarPages.forEach((page) => {
    // Reading link.href rather than the raw attribute lets the browser resolve
    // the relative path for us. That also covers the page being viewed, whose
    // own sidebar link is rendered as "#".
    const links = sidebarLinks.filter(
      (link) => new URL(link.href).pathname.endsWith(pagePath(page))
    );

    if (!links.length) {
      console.warn(`new-badge.js: no page "${page}" in the navigation sidebar`);
      return;
    }
    links.forEach((link) => link.appendChild(makeBadge()));
  });

  NEW_SECTIONS.forEach((entry) => {
    const [page, anchor] = entry.split("#");

    // A section only exists on its own page, so skip quietly everywhere else.
    if (!window.location.pathname.endsWith(pagePath(page))) return;

    const heading = document.querySelector(
      `section[id="${anchor}"] > :is(h1, h2, h3, h4, h5, h6)`
    );
    if (!heading) {
      console.warn(`new-badge.js: no section "${anchor}" on page "${page}"`);
      return;
    }

    // Sits ahead of the trailing "#" permalink.
    heading.insertBefore(makeBadge(), heading.querySelector("a.headerlink"));

    // Scoped to the on-screen panel: an unscoped lookup matches the print-only
    // copy of the same table of contents first.
    const tocEntry = document.querySelector(
      `#pst-page-toc-nav a.nav-link[href="#${anchor}"]`
    );
    if (tocEntry) tocEntry.appendChild(makeBadge());
  });
});
