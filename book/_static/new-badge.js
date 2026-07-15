// Adds a "New!" badge next to selected pages in the sidebar navigation.
//
// To flag a page, add its path below: relative to the book root and without the
// file extension, i.e. exactly the path used in _toc.yml. Delete the entry again
// once the page is no longer new.

const NEW_PAGES = [
  "content/misc/post_process",
];

document.addEventListener("DOMContentLoaded", () => {
  const targets = NEW_PAGES.map((page) => `/${page.replace(/^\/+/, "")}.html`);

  document.querySelectorAll("nav.bd-docs-nav a.reference.internal").forEach((link) => {
    // Reading link.href (rather than the raw attribute) lets the browser resolve
    // the relative path for us. That also covers the page being viewed, whose own
    // sidebar link is rendered as "#".
    const path = new URL(link.href).pathname;
    if (!targets.some((target) => path.endsWith(target))) return;

    const badge = document.createElement("span");
    badge.className = "new-badge";
    badge.textContent = "New!";
    link.appendChild(badge);
  });
});
