# Pan-zonal HepatoNet FastCORE reduction

This repository contains the files needed to reproduce the pan-zonal FastCORE-reduced HepatoNet network used for PhysiCell-dFBA hepatic lobule simulations.

## Folder structure

```text
config/
  hepatonet_original_13_08.xml
  PhysiCell_settings.xml

notebooks/
  pan_zonal_fastcore_reduction_workflow.ipynb
  run_pan_zonal_reviewer_fastcore.py
  run_pan_zonal_official_fastcore.py
  run_official_fastcore_literature.py
  run_literature_core_tests.py
  fastcore_outputs/
    reviewer_minimal_carbohydrate_core.csv

requirements.txt
```

## Running in Google Colab

Open:

```text
notebooks/pan_zonal_fastcore_reduction_workflow.ipynb
```

Then run the cells from top to bottom.

The first Colab cell clones this GitHub repository. If the repository URL is different, update `REPO_URL` in the notebook.

The dependency cell installs:

```text
cobra
pandas
python-libsbml
swiglpk
```

## Main outputs

The workflow generates:

```text
notebooks/fastcore_outputs/pan_zonal_reviewer_core/pan_zonal_reviewer_core_reactions.csv
notebooks/fastcore_outputs/pan_zonal_reviewer_core/pan_zonal_reviewer_validation.csv
notebooks/fastcore_outputs/pan_zonal_reviewer_core/official_fastcore/hepatonet_pan_zonal_official_fastcore_reduced.xml
notebooks/fastcore_outputs/pan_zonal_reviewer_core/official_fastcore/pan_zonal_official_fastcore_summary.csv
notebooks/fastcore_outputs/pan_zonal_reviewer_core/official_fastcore/pan_zonal_official_fastcore_validation.csv
```

The expected validated network has 185 reactions, 225 metabolites, 73 retained core reactions, and all module validation tests should return `optimal`.

## Optional PhysiCell step

The notebook has an optional cell that copies the validated SBML to:

```text
config/hepatonet_pan_zonal_official_fastcore_reduced.xml
```

and defines `EX_HC00021_s` as the objective reaction for PhysiCell-dFBA coupling.
