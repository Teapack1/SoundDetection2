# data_library/

Shared, reusable audio assets across all projects. Anything that's
project-agnostic lives here so you record it once and reuse it everywhere.

## Layout

```
data_library/
├── glasses_clean/<glass_type>/<uuid>.wav    # studio-clean glass clinks
├── ambience/<source_label>/<uuid>.wav       # rooms, bars, restaurants
├── negatives_external/<class>/<uuid>.wav    # cutlery, ice, bottle, music...
└── manifest.jsonl                           # one line per clip, with metadata
```

## How clips get in here

- **Studio glass recordings**: `sed library record-glass --type wine_thin --num 50`
  records ~50 onset-triggered clean clinks of one glass type and registers them.
- **Bulk import**: `sed library import <dir> --category glasses_clean --label wine_thin`
  imports an existing folder of WAVs.
- **Auto-promotion** (manual for now): a clip recorded for one project that
  turns out to be reusable can be moved to the library by editing its sidecar
  and rerunning import.

## How they get used in a project

`projects/<id>/project.toml` can include a `[dataset.library]` block:

```toml
[dataset.library]
"clink_wine_thin"  = [{ category = "glasses_clean", label = "wine_thin"  }]
"clink_wine_thick" = [{ category = "glasses_clean", label = "wine_thick" }]
"_negative"        = [
    { category = "negatives_external", label = "cutlery" },
    { category = "negatives_external", label = "ice"     },
    { category = "negatives_external", label = "bottle"  },
]
```

`sed dataset compose` picks these up alongside on-site recordings.

## Domain mixing pool

The `ambience/` subtree is **also** used at training time as the domain-mixing
pool, regardless of project. So having a deep, varied ambience pool here pays
off for every model you train. Aim for at least 30 minutes of ambience from
3–5 distinct contexts.
