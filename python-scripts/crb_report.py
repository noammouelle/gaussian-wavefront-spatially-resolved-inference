"""
crb_report.py -- shared table-printing/saving helpers for the CRB-vs-empirical
scripts (non_phase_shear/analysis/crb_vs_empirical.py and
python-scripts/phase_shear_crb_vs_empirical.py).

The two pipelines being compared (kinematic vs. phase-shear readout) have
different estimator names (best/moments/null vs. raw/fitted-feat/oracle),
but the SHAPE of "how good is this estimator relative to the noise floor"
is identical: one row per quantity, one CRB column, one (value, ratio-to-CRB)
pair per method. Printing both through this one formatter is what makes the
two pipelines' console output directly comparable side by side.
"""
import json


def print_comparison_table(title, rows, methods, unit=''):
    """rows: list of (component_name, crb_value) pairs, in the SAME units
    for every method. methods: dict {method_name: {component_name: value}}
    (a component missing from a method's dict is skipped for that method).
    unit: optional suffix printed after each value, e.g. 'mrad'."""
    print(f'\n=== {title} ===')
    method_names = list(methods.keys())
    header = f'  {"component":12s} {"CRB":>12s}  ' + '  '.join(f'{m:>28s}' for m in method_names)
    print(header)
    for name, crb_val in rows:
        cells = []
        for m in method_names:
            val = methods[m].get(name)
            if val is None:
                cells.append(f'{"--":>28s}')
            else:
                ratio = val / crb_val if crb_val else float('nan')
                cells.append(f'{val:.4e}{unit} ({ratio:5.2f}x)'.rjust(28))
        print(f'  {name:12s} {crb_val:.4e}{unit}  ' + '  '.join(cells))


def save_json(path, payload):
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2, default=float)
    print(f'\nWrote {path}')
