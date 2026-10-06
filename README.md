# DIWA paper project page

A self-contained, English research page for **DIWA: Decision-Influential World Abstraction for VLA-WAM Policies**, an anonymous ICLR 2027 submission. The site uses plain HTML, CSS, and JavaScript. No build step, CDN, external fonts, backend, or API key is required.

## Publish with GitHub Pages

1. Push the contents of this directory to a GitHub repository, including `index.html`, `styles.css`, `script.js`, `.nojekyll`, and `assets/`.
2. Open the repository's **Settings > Pages**.
3. Under **Build and deployment**, choose **Deploy from a branch**.
4. Select your publishing branch (usually `main`) and **/ (root)**, then save.
5. Wait for the Pages deployment to finish. Open the URL shown in Settings > Pages.

All site resources use relative URLs, so both `https://USER.github.io/REPOSITORY/` and custom-domain roots work. No repository name needs to be hard-coded. No remote repository or public deployment is configured by this project.

GitHub documentation: [About GitHub Pages](https://docs.github.com/en/pages/getting-started-with-github-pages/what-is-github-pages) and [publishing-source configuration](https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site).

## Local preview

Open `index.html` directly in a browser, or use Node.js for an HTTP preview:

```sh
node scripts/serve.mjs
```

Then visit `http://localhost:4173`. Use `node scripts/serve.mjs 4174` if the port is occupied. This development server binds to localhost only.

## Files and provenance

- `index.html`: paper narrative, original architecture figure, accessible static tables, download links, and no-JavaScript fallbacks.
- `styles.css`: responsive layout, print styles, focus states, and reduced-motion support.
- `script.js`: illustrative query selection, measured budget frontier, result filters, navigation, and canvas animation.
- `assets/results.js`: browser data generated directly from the manuscript ledger.
- `assets/results-ledger.json` and `assets/reported-measurements.json`: copies of the numerical evidence supplied with the paper.
- `assets/diwa-paper.pdf`, `assets/diwa-source.zip`, `assets/main.tex`: manuscript resources. The ZIP is packaged from the current source directory.
- `assets/architecture.jpg`: original manuscript figure.
- `assets/latency.png` and `assets/budget-frontier.png`: rasterized versions of the corresponding original PDF plots, also included.
- `assets/vendor/`: local Motion and Lucide distributions and licenses. See [third-party notices](THIRD_PARTY_NOTICES.md).

To refresh copies and browser data after a manuscript update:

```sh
node scripts/prepare-assets.mjs
python scripts/package-site-source.py
```

The second command uses only Python's standard library and rebuilds the download archive from the current manuscript inputs. Use an available Python interpreter by its full path when necessary. PNG plots should also be re-rendered when their PDF originals change, for example with Poppler:

```sh
pdftoppm -png -singlefile -scale-to 1600 assets/latency.pdf assets/latency
pdftoppm -png -singlefile -scale-to 1600 assets/budget-frontier.pdf assets/budget-frontier
```

Keep the static HTML tables and headline figures aligned with the updated ledger. They intentionally remain readable without JavaScript.

## Data interpretation

The five fixed-budget controls display measured operating points only: 3, 6, 12, 24, and 48 queries. The adaptive setting is separate: a 12-query ceiling, 23.7% mean retention, 73.5% success, and 89 ms latency. The illustrated adaptive grid shows the ceiling, not a claim that every context uses 12 queries. Query rankings and moving packets are schematic; this site does not execute a trained policy or display recorded model activations.

Platform means, OOD success, physical counts, and counterfactual decision-label accuracy have distinct statistical meanings. The evidence section records those distinctions and the manuscript's reporting limits. The supplied materials do not include policy training code, weights, or rollout videos.

## Verification

`scripts/verify-site.mjs` checks local resources and manuscript-data consistency using Node.js. Browser verification can be run with `scripts/check-browser.cjs` when Playwright and a Chromium browser are available; set `PLAYWRIGHT_MODULE` to the absolute Playwright package path if it is not installed locally. It uses `http://localhost:4173` by default and writes screenshots under `.qa/` (ignored by Git).

The page is tested at wide desktop, narrow desktop, and mobile sizes, with all five budgets, adaptive mode, result filters, mobile navigation, image dialog, keyboard focus, no JavaScript, and reduced motion. Local tests verify the static site; they do not reproduce the robotics experiments.

## Rights

Third-party libraries retain their upstream licenses. Paper text, figures, data, and manuscript source retain their original authors' rights; including a download does not grant a new license to those materials.
