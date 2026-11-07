The official LUNA16 `evaluationScript` (2016-06-17) ported to Python 3 with no logic changes (print statements,
dict iteration, csv file modes; bootstrapping switched off by default for speed). Run:

    python lungcad/eval/official/noduleCADEvaluationLUNA16.py annotations/annotations.csv annotations/annotations_excluded.csv annotations/seriesuids.csv <candidates.csv> <outdir>

`lungcad.eval.froc` reproduces its output exactly (tests/test_froc_official.py) and is what the training loops use,
because it is ~100x faster; final thesis tables should additionally be produced with this script for the record.
