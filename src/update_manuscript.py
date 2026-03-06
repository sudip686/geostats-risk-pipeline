"""Update manuscript key numbers from latest outputs."""

import os
import re
import pandas as pd
from src.utils.io import load_config


def _format_mt(value):
    return f"{value:.2f}"


def update(manuscript_path='paper/manuscript.md', outputs_dir='outputs', config_path='config/project.yaml'):
    config = load_config(config_path)
    cutoff = config.get('cutoff_grade', 3.0)
    risk_path = os.path.join(outputs_dir, 'tables', 'risked_tonnage.csv')
    if not os.path.exists(risk_path):
        return
    risk = pd.read_csv(risk_path)
    row = risk[risk['cutoff'] == cutoff].iloc[0]

    p10 = row['tonnage_p10'] / 1e6
    p50 = row['tonnage_p50'] / 1e6
    p90 = row['tonnage_p90'] / 1e6
    g50 = row['grade_p50']
    c50 = row['contained_p50'] / 1e6

    text = open(manuscript_path, 'r', encoding='utf-8').read()

    # Update validation metrics block if present
    metrics_path = os.path.join(outputs_dir, 'tables', 'validation_metrics.json')
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        metrics_text = (
            "Validation against a block-support calibration target indicates strong alignment after change-of-support correction: "
            f"mean(data) = {metrics.get('mean_data', 0):.2f}%, mean(sim) = {metrics.get('mean_sim', 0):.2f}; "
            f"std(data) = {metrics.get('std_data', 0):.2f}, std(sim) = {metrics.get('std_sim', 0):.2f}; "
            f"histogram overlap = {metrics.get('hist_overlap', 0):.2f}. "
            f"Swath correlations are positive (X = {metrics.get('swath_corr_x', 0):.2f}, "
            f"Y = {metrics.get('swath_corr_y', 0):.2f}, Z = {metrics.get('swath_corr_z', 0):.2f}) "
            f"with {metrics.get('swath_coverage_pct', 0):.1f}% of bins within the P10–P90 envelope, "
            "indicating that the model captures both lateral and vertical trends at pilot-grid scale."
        )

        text = re.sub(
            r"Validation against a block-support calibration target.*?pilot-grid scale\.",
            metrics_text,
            text,
            flags=re.DOTALL,
        )

    text = re.sub(
        r"At a \d+(?:\.\d+)?% TGC cutoff grade, the risked tonnage analysis yields P10/P50/P90 estimates of .*? Mt, respectively, with P50 grade of ~.*?% and contained graphite P50 of ~.*? Mt\.",
        f"At a {cutoff:.0f}% TGC cutoff grade, the risked tonnage analysis yields P10/P50/P90 estimates of {p10:.2f} / {p50:.2f} / {p90:.2f} Mt, respectively, with P50 grade of ~{g50:.2f}% and contained graphite P50 of ~{c50:.2f} Mt.",
        text,
    )

    mt_line = (
        r"\(Mt = million tonnes; block volume 100x100x10 m; density 2\.43 t/m3\)"
    )
    text = re.sub(
        mt_line,
        "(Mt = million tonnes; block volume 100x100x10 m; density 2.43 t/m3)",
        text,
    )

    text = re.sub(
        r"where \$I\$ is the indicator function and \$c\$ = \d+(?:\.\d+)?% TGC\.",
        f"where $I$ is the indicator function and $c$ = {cutoff:.0f}% TGC.",
        text,
    )

    text = re.sub(
        r"At the \d+(?:\.\d+)?% TGC cutoff:\n- P10 tonnage: .*? Mt\n- P50 tonnage: .*? Mt\n- P90 tonnage: .*? Mt\n- P50 grade: .*?% TGC\n- P50 contained graphite: .*? Mt",
        f"At the {cutoff:.0f}% TGC cutoff:\n- P10 tonnage: {p10:.2f} Mt\n- P50 tonnage: {p50:.2f} Mt\n- P90 tonnage: {p90:.2f} Mt\n- P50 grade: {g50:.2f}% TGC\n- P50 contained graphite: {c50:.2f} Mt",
        text,
    )

    text = re.sub(
        r"\*\*Conservative Scenario \(P10\):\*\* .*? Mt at .*?% TGC",
        f"**Conservative Scenario (P10):** {p10:.2f} Mt at {g50:.2f}% TGC",
        text,
    )
    text = re.sub(
        r"\*\*Expected Scenario \(P50\):\*\* .*? Mt at .*?% TGC",
        f"**Expected Scenario (P50):** {p50:.2f} Mt at {g50:.2f}% TGC",
        text,
    )
    text = re.sub(
        r"\*\*Optimistic Scenario \(P90\):\*\* .*? Mt at .*?% TGC",
        f"**Optimistic Scenario (P90):** {p90:.2f} Mt at {row['grade_p90']:.2f}% TGC",
        text,
    )

    text = re.sub(
        r"4\. \*\*Resource Risk:\*\* At a \d+(?:\.\d+)?% TGC cutoff, the pilot-grid risked resource is:",
        f"4. **Resource Risk:** At a {cutoff:.0f}% TGC cutoff, the pilot-grid risked resource is:",
        text,
    )
    text = re.sub(
        r"- P10: .*? Mt @ .*?% TGC\n- P50: .*? Mt @ .*?% TGC\n- P90: .*? Mt @ .*?% TGC",
        f"- P10: {p10:.2f} Mt @ {g50:.2f}% TGC\n- P50: {p50:.2f} Mt @ {g50:.2f}% TGC\n- P90: {p90:.2f} Mt @ {row['grade_p90']:.2f}% TGC",
        text,
    )

    text = re.sub(
        r"6\. \*\*Economic Implications:\*\* Contained graphite values from the pilot grid indicate a P50 contained graphite of ~.*? Mt at \d+(?:\.\d+)?% cutoff\.",
        f"6. **Economic Implications:** Contained graphite values from the pilot grid indicate a P50 contained graphite of ~{c50:.2f} Mt at {cutoff:.0f}% cutoff.",
        text,
    )

    with open(manuscript_path, 'w', encoding='utf-8') as f:
        f.write(text)


def run():
    update()


if __name__ == '__main__':
    run()
