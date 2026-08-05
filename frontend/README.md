# Results explorer

This is a static React/Vite frontend for exploring the committed drug-target prioritization results. It deliberately exposes the verified headline numbers and figures without pretending that the generated prediction cache is part of the repository.

```bash
npm install
npm run dev
```

To populate the ranked-target view, run the ML evaluation and transform its output:

```bash
python3 ml/train_eval.py --feature-set biology_only
```

Then, from this directory, export the cache for the frontend:

```bash
npm run export-targets
```

The expected frontend data contract is `public/data/targets.json`, with rows containing `symbol`, `score`, `label`, `rank`, and `fold_idx`. The JSON artifact is generated output and is intentionally not committed.
