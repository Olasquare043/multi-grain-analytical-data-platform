"""Task 8 (paper repair pass): renumber all 25 figures sequentially by order
of first mention, plus the new Task 7 figure as number 26. Copies (never
moves or edits) the existing 300 dpi PNGs from outputs/figures/ into
outputs/paper/figures_renumbered/ under new fig_NN_<slug>.png names, and
writes outputs/paper/figure_map.csv.

Pure filesystem + csv work; no docker/duckdb/lightgbm needed. Captions are
reproduced verbatim from outputs/figures/figure_manifest.csv (for the A- and
docs-series figures) and from the notebook cells that generated the three
ladder figures (their captions were never written to figure_manifest.csv).

Two orderings are assumptions, stated here because the paper text is not
available to this pass to check against directly:
  - A1a/A1b: assigned in the order src/viz/figures.py's fig_a1() generates
    them (heatmap, then lines).
  - A12a/b/c/d: assigned in the order src/viz/figures.py's fig_a12()
    generates them (national, state_premiums, rank_stability, zones).
Both are flagged in the `label_assumption` column so they can be corrected
against the paper's own captions if the paper orders them differently.
"""
from __future__ import annotations

import shutil

import pandas as pd

from config import settings

FIGURES_DIR = settings.OUTPUTS_DIR / "figures"
PAPER_DIR = settings.OUTPUTS_DIR / "paper"
OUT_DIR = PAPER_DIR / "figures_renumbered"

MANIFEST = FIGURES_DIR / "figure_manifest.csv"

LADDER_CAPTIONS = {
    "fig_model_ladder_nigeria.png": (
        "Nigerian petrol price, one month ahead: MAPE by rung, with repeat spread. "
        "Bars are the mean of 5 repeats (seeds 1-5, varying the model's random_state only; "
        "Task A does no sampling); error bars are one standard deviation across those repeats. "
        "Dashed and dotted lines are reference predictors that fit no model: a random walk, and "
        "the training mean. V3a (hatched) is computed with future months present and is reported "
        "only for comparison against V3b. Which rung-to-rung differences hold their direction "
        "across all repeats is reported in outputs/tables/ladder_paired_comparisons.csv, not "
        "readable from bar heights alone."
    ),
    "fig_model_ladder_nyc.png": (
        "NYC trip duration, pickup-time features: MAPE by rung, with repeat spread. Bars are the "
        "mean of 5 repeats (seeds 1-5, each drawing its own deterministic content-addressed bucket "
        "of about 1,957,054 training rows and using its own model random_state); error bars are one "
        "standard deviation across those repeats, i.e. the amount a rung moves when nothing "
        "meaningful changes. The y-axis is broken and does not start at zero: the training-mean "
        "reference sits far above every rung, and on a single zero-based axis the rungs and their "
        "error bars compress into an unreadable band. Dashed and dotted lines are reference "
        "predictors that fit no model: distance over average speed, and the training mean. V3a "
        "(hatched) is computed with future months present and is reported only for comparison "
        "against V3b. Which rung-to-rung differences hold their direction across all repeats is in "
        "outputs/tables/ladder_paired_comparisons.csv, and is not readable from bar heights alone."
    ),
    "fig_ladder_comparison.png": (
        "Both ladders on one axis, and which steps survived five repeats. Bars are the mean of 5 "
        "repeats, shown as % change in MAPE against each task's own V0, so two incomparable error "
        "scales (naira/litre, seconds) sit on one axis. Markers report whether the step from the "
        "previous rung into this one held its direction across ALL five repeats, taken within seed: "
        "a difference that flipped sign (cross) is not an established effect however large its bar "
        "looks. Bars and markers both show each task's ORIGINAL hyperparameter variant only, so "
        "NYC's single inconsistent comparison, which occurs under the capacity-controlled variant "
        "(V4b to V5, 4/5), is not visible here; the full set for both variants is in "
        "ladder_paired_comparisons.csv. The same fixed model and hyperparameters are used at every "
        "rung of a given ladder; only the input features change. V3a is computed with future "
        "months present and is shown only for contrast with V3b."
    ),
}

# (current_filename, current_label_in_paper, section, slug, label_assumption)
SEQUENCE = [
    ("fig_erd_conceptual.png", "Figure 4.1", "4.1", "erd_conceptual", ""),
    ("fig_erd_star_schema_dimensions.png", "Figure 4.2 (dimensions)", "4.2", "erd_star_schema_dimensions", ""),
    ("fig_erd_star_schema_facts.png", "Figure 4.2 (facts)", "4.2", "erd_star_schema_facts", ""),
    ("fig_architecture_ingestion.png", "Figure 4.4 (ingestion)", "4.4", "architecture_ingestion", ""),
    ("fig_architecture_serving.png", "Figure 4.4 (serving)", "4.4", "architecture_serving", ""),
    ("fig_cloud_architecture.png", "Figure 4.5", "4.5", "cloud_architecture", ""),
    ("A1_demand_profile_heatmap.png", "Figure A1a", "6.2", "a1a_demand_profile_heatmap",
     "ASSUMED: A1a=heatmap, A1b=lines, per src/viz/figures.py fig_a1() generation order -- verify against the paper's own A1a/A1b captions."),
    ("A1_demand_profile_lines.png", "Figure A1b", "6.2", "a1b_demand_profile_lines",
     "ASSUMED: A1a=heatmap, A1b=lines, per src/viz/figures.py fig_a1() generation order -- verify against the paper's own A1a/A1b captions."),
    ("A2_od_corridors.png", "Figure A2", "6.2", "a2_od_corridors", ""),
    ("A3_congestion_proxy.png", "Figure A3", "6.2", "a3_congestion_proxy", ""),
    ("A5_revenue_concentration.png", "Figure A5", "6.2", "a5_revenue_concentration", ""),
    ("A4_tipping_behaviour.png", "Figure A4", "6.2", "a4_tipping_behaviour", ""),
    ("A11_missingness_structure.png", "Figure A11", "6.2", "a11_missingness_structure", ""),
    ("A6_food_price_index.png", "Figure A6", "6.3", "a6_food_price_index", ""),
    ("A8_price_dispersion.png", "Figure A8", "6.3", "a8_price_dispersion", ""),
    ("A12_petrol_price_national.png", "Figure A12a", "6.3", "a12a_petrol_price_national",
     "ASSUMED: A12a-d = national, state_premiums, rank_stability, zones, per src/viz/figures.py fig_a12() generation order -- verify against the paper's own A12a-d captions."),
    ("A12_petrol_price_state_premiums.png", "Figure A12b", "6.3", "a12b_petrol_price_state_premiums",
     "ASSUMED: see A12a note."),
    ("A12_petrol_price_rank_stability.png", "Figure A12c", "6.3", "a12c_petrol_price_rank_stability",
     "ASSUMED: see A12a note."),
    ("A12_petrol_price_zones.png", "Figure A12d", "6.3", "a12d_petrol_price_zones",
     "ASSUMED: see A12a note."),
    ("A7_subsidy_structural_break.png", "Figure A7", "6.4", "a7_subsidy_structural_break", ""),
    ("A9_fuel_food_passthrough.png", "Figure A9", "6.4", "a9_fuel_food_passthrough", ""),
    ("A10_grain_comparison.png", "Figure A10", "6.5", "a10_grain_comparison", ""),
    ("fig_model_ladder_nigeria.png", "Figure 6.1", "6.7.1", "ladder_nigeria", ""),
    ("fig_model_ladder_nyc.png", "Figure 6.2", "6.7.2", "ladder_nyc", ""),
    ("fig_ladder_comparison.png", "Figure 6.3", "6.10", "ladder_comparison", ""),
]


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(MANIFEST).set_index("filename")["caption"].to_dict()

    rows = []
    for i, (fname, label, section, slug, assumption) in enumerate(SEQUENCE, start=1):
        src_path = FIGURES_DIR / fname
        if not src_path.exists():
            raise FileNotFoundError(f"expected figure not found: {src_path}")
        new_name = f"fig_{i:02d}_{slug}.png"
        dest_path = OUT_DIR / new_name
        shutil.copy2(src_path, dest_path)
        caption = LADDER_CAPTIONS.get(fname) or manifest.get(fname, "")
        rows.append({
            "current_filename": fname,
            "current_label_in_paper": label,
            "current_section_first_mentioned": section,
            "new_number": i,
            "new_filename": new_name,
            "caption_as_currently_written": caption,
            "label_assumption": assumption,
        })

    # Task 7's new figure, additional, numbered 26.
    extra_src = PAPER_DIR / "figures" / "fig_extrapolation_ceiling.png"
    if extra_src.exists():
        new_name = "fig_26_extrapolation_ceiling.png"
        shutil.copy2(extra_src, OUT_DIR / new_name)
        rows.append({
            "current_filename": "fig_extrapolation_ceiling.png (new, Task 7 of this repair pass)",
            "current_label_in_paper": "(not previously in the paper)",
            "current_section_first_mentioned": "(new)",
            "new_number": 26,
            "new_filename": new_name,
            "caption_as_currently_written": (
                "Nigeria: predicted vs. actual national mean petrol price, test window. See "
                "outputs/paper/figures/fig_extrapolation_ceiling.png's own embedded caption and "
                "outputs/paper/extrapolation_ceiling.csv for the full underlying series."
            ),
            "label_assumption": "additional figure introduced by this repair pass (Task 7), not a renumbering of an existing one.",
        })
    else:
        print("NOTE: outputs/paper/figures/fig_extrapolation_ceiling.png not found yet -- "
              "run src/paper/extrapolation_ceiling.py first and re-run this script to include "
              "figure 26 in the map and copy it into figures_renumbered/.")

    out = pd.DataFrame(rows)
    out.to_csv(PAPER_DIR / "figure_map.csv", index=False)
    print(f"wrote outputs/paper/figure_map.csv ({len(out)} rows)")
    print(f"copied {len(rows)} PNGs into {OUT_DIR}")


if __name__ == "__main__":
    main()
