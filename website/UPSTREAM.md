# Website provenance

The website is adapted for the same EmoMosaic project from:

- Reference website: https://rudykon.github.io/EmoMosaic_GMCM2026/
- Source repository: https://github.com/rudykon/EmoMosaic_GMCM2026
- Reference commit: `50202237f5f38f1e44ee916c596d9ab57a92d94e`
- Current project: https://github.com/rudykon/EmoMosaic

The original layout, base CSS, favicon, navigation, theme switching, translation runtime, search, application screenshots, frozen experimental data and browser inference implementation are reused. Page labels, project links, reproduction instructions, selected explanations and translations are adapted to the current research project. The build is independent of Android and model-training dependencies.

Research figures use Figures 1–12 from the current manuscript's `paper/figures/` directory, replacing the historical website figures. Original PDFs are copied unchanged; full-page PNG previews are rendered without cropping or redrawing. Source paths, original PDF hashes, preview hashes and figure numbers are recorded in `assets/paper/sources.json`. The homepage uses the manuscript's model overview and error diagnostics; the other figures accompany their corresponding research sections. The existing application screenshots remain specific to the application showcase.

Experimental data retains its source hashes in `data/results.json`. Inference weights retain their original hashes and are downloaded on demand from the reference website. Historical verification reports are identified as such in the website README.

KaTeX, ONNX Runtime and BERT notices are retained alongside their assets. Reuse of a website asset does not change the original terms of data, models or third-party components.
