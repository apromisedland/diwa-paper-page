"""Build manuscript tables and an arithmetic ledger from recorded summaries."""
from pathlib import Path
import hashlib
import json
import math
import re
import statistics as stats

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "evidence/reported_measurements.json"
SOURCE_TABLES = json.loads(SOURCE.read_text())["tables"]
PAPER = ROOT / "paper" if (ROOT / "paper").is_dir() else ROOT
OUT = PAPER / "tables"
OUT.mkdir(parents=True, exist_ok=True)
checks = []


def table_after(heading):
    return SOURCE_TABLES[heading]


def number(cell):
    return float(re.search(r"-?\d+(?:\.\d+)?", cell).group())


def close(a, b, label, tol=1e-8):
    assert abs(a - b) <= tol, (label, a, b)
    checks.append(label)


def rounded(a, b, places, label):
    close(float(f"{a:.{places}f}"), b, label)


def wilson95(successes, trials):
    """Conditional within-task binomial interval, derived from measured counts."""
    assert trials > 0 and 0 <= successes <= trials
    z = stats.NormalDist().inv_cdf(0.975)
    rate = successes / trials
    denominator = 1 + z * z / trials
    center = (rate + z * z / (2 * trials)) / denominator
    half_width = z * math.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials)) / denominator
    return [100 * max(0, center - half_width), 100 * min(1, center + half_width)]


def write_table(name, caption, label, columns, headers, rows, note="", placement="t"):
    content = [r"\begin{table}[" + placement + "]", r"\centering", "\\caption{" + caption + "}",
               "\\label{" + label + "}", r"\small", "\\begin{tabular}{" + columns + "}",
               r"\toprule", " & ".join(headers) + r"\\", r"\midrule"]
    content += [r"\midrule" if row is None else " & ".join(row) + r"\\" for row in rows]
    content += [r"\bottomrule", r"\end{tabular}"]
    if note:
        content += [r"\par\vspace{3pt}", r"\begin{minipage}{\linewidth}\footnotesize " + note + r"\end{minipage}"]
    content += [r"\end{table}", ""]
    (OUT / name).write_text("\n".join(content))


seed_data = []
for row in table_after("### 3.2 仿真平台三种子结果"):
    method, platform = row[0].split("／")
    values = [number(v) for v in row[1:4]]
    mean, sd = stats.mean(values), stats.stdev(values)
    sem = sd / math.sqrt(3)
    for metric, value, reported, places in zip(["mean", "sd", "sem"], [mean, sd, sem], row[4:], [1, 2, 2]):
        rounded(value, number(reported), places, f"{method}/{platform}/{metric}")
    seed_data.append(dict(method=method, platform=platform, seeds=[42, 43, 44],
                          measured_success_pct=values, mean_pct=mean, sample_sd_pp=sd, sem_pp=sem))
by_seed = {(row["method"], row["platform"]): row for row in seed_data}

real_names = ["Grasping", "Precise placement", "Drawer interaction", "Insertion",
              "Container manipulation", "Multi-stage contact"]
real = []
for name, row in zip(real_names, table_after("### 3.3 真实机器人测试次数与成功计数")[:6]):
    n, dream, diwa = [int(number(v)) for v in row[1:4]]
    close(dream / n * 100, number(row[4]), name + "/DreamVLA rate")
    close(diwa / n * 100, number(row[5]), name + "/DIWA rate")
    real.append(dict(task=name, trials_per_method=n, DreamVLA_successes=dream, DIWA_successes=diwa))
real_total = {"trials_per_method": sum(r["trials_per_method"] for r in real),
              **{m: sum(r[m + "_successes"] for r in real) for m in ["DreamVLA", "DIWA"]}}
close(real_total["DreamVLA"], 196, "DreamVLA physical successes")
close(real_total["DIWA"], 216, "DIWA physical successes")
close(real_total["trials_per_method"], 300, "physical trial count")

main = []
main_rows = []
for row in table_after("### 2.1 四平台成功率"):
    method = row[0]
    sims = [by_seed[method, p] for p in ["LIBERO", "RoboTwin", "RoboCasa"]]
    physical = real_total[method] / real_total["trials_per_method"] * 100 if method in real_total else number(row[4])
    macro = stats.mean([s["mean_pct"] for s in sims] + [physical])
    for value, reported, p in zip([s["mean_pct"] for s in sims] + [physical, macro], row[1:],
                                ["LIBERO", "RoboTwin", "RoboCasa", "Real", "Macro"]):
        rounded(value, number(reported), 1, method + "/main/" + p)
    main.append(dict(method=method, LIBERO=sims[0]["mean_pct"], RoboTwin=sims[1]["mean_pct"],
                     RoboCasa=sims[2]["mean_pct"], real_robot_pct=physical, macro_pct=macro))
    cells = [method] + [f"${s['mean_pct']:.1f} \\pm {s['sample_sd_pp']:.2f}$" for s in sims] + [f"{physical:.1f}", f"{macro:.1f}"]
    if method == "DIWA":
        main_rows.append(None)
        cells = [r"\textbf{DIWA}"] + [r"$\mathbf{" + f"{s['mean_pct']:.1f} \\pm {s['sample_sd_pp']:.2f}" + "}$" for s in sims] + [r"\textbf{" + f"{v:.1f}" + "}" for v in [physical, macro]]
    main_rows.append(cells)
write_table("main.tex", r"Task success (\%) under our evaluation protocol. Simulation columns report mean $\pm$ sample SD across three training seeds. Real-robot results use the seed-42 checkpoint and 300 trials per method. The final column weights the four platform means equally.", "tab:main", "lrrrrr",
            ["Method", "LIBERO", "RoboTwin", "RoboCasa", "Real robot", "Macro"], main_rows)

libero = []
for row in table_after("### 2.2 LIBERO 子集成功率"):
    values = list(map(number, row[1:]))
    close(stats.mean(values[:4]), values[4], row[0] + "/LIBERO suite mean")
    close(values[4], by_seed[row[0], "LIBERO"]["mean_pct"], row[0] + "/LIBERO suite-seed agreement")
    libero.append(dict(method=row[0], **dict(zip(["Spatial", "Object", "Goal", "Long", "Mean"], values))))
write_table("libero.tex", r"LIBERO suite success (\%). Each suite contains ten tasks with equal evaluation counts. Long denotes \texttt{libero\_10}. Values average seeds 42, 43, and 44.", "tab:libero", "lrrrrr",
            ["Method", "Spatial", "Object", "Goal", "Long", "Mean"],
            [[r["method"]] + [f"{r[k]:.1f}" for k in ["Spatial", "Object", "Goal", "Long", "Mean"]] for r in libero])

suite_seeds = []
for row in table_after("与 LIBERO 平台种子成绩对应的子集实测明细如下："):
    method, suite = row[0].split("／")
    values = list(map(number, row[1:]))
    entry = dict(method=method, suite=suite, seeds=[42, 43, 44], measured_success_pct=values,
                 mean_pct=stats.mean(values), sample_sd_pp=stats.stdev(values))
    close(entry["mean_pct"], next(r[suite] for r in libero if r["method"] == method), method + "/" + suite + "/seed mean")
    suite_seeds.append(entry)
for method in ["DreamVLA", "DIWA"]:
    for i in range(3):
        close(stats.mean(r["measured_success_pct"][i] for r in suite_seeds if r["method"] == method),
              by_seed[method, "LIBERO"]["measured_success_pct"][i], f"{method}/seed {42+i}/suite agreement")

ood = []
for name, row in zip(["Appearance", "Background motion", "Contact counterfactual", "Equal-weight mean"], table_after("### 2.3 OOD 与接触反事实结果")):
    dream, diwa, gain = list(map(number, row[1:]))
    close(diwa - dream, gain, name + "/OOD gain")
    ood.append(dict(condition=name, DreamVLA=dream, DIWA=diwa, gain_pp=gain))
for method in ["DreamVLA", "DIWA"]:
    close(stats.mean(r[method] for r in ood[:3]), ood[-1][method], method + "/OOD condition mean")
    close(ood[-1][method], by_seed[method, "OOD"]["mean_pct"], method + "/OOD seed agreement")
write_table("ood.tex", r"Success under visual shifts and contact counterfactuals (\%). Contact episode success differs from the CF decision accuracy in Table~\ref{tab:ablation}.", "tab:ood", "lrrr",
            ["Condition", "DreamVLA", "DIWA", "Gain (points)"],
            [[r["condition"], f"{r['DreamVLA']:.1f}", f"{r['DIWA']:.1f}", f"+{r['gain_pp']:.1f}"] for r in ood])

ablation_names = ["DIWA", "Without influence estimator", "Without cross-episode swaps", "Without regret geometry", "Dense imagination", "Uniform Top-$K$"]
ablations = [dict(variant=name, **dict(zip(["standard_pct", "ood_pct", "cf_accuracy_pct", "latency_ms"], map(number, row[1:]))))
             for name, row in zip(ablation_names, table_after("## 4. 完整消融结果"))]
write_table("ablation.tex", r"Evaluated DIWA variants. Standard success is the four-platform macro-average; OOD success and CF decision accuracy are separate metrics. Rates are percentages; latency is in milliseconds. These point estimates do not establish equal retained counts across variants.", "tab:ablation", "lrrrr",
            ["Variant", "Standard", "OOD", "CF accuracy", "Latency"],
            [[r["variant"]] + [f"{r[k]:.1f}" for k in ["standard_pct", "ood_pct", "cf_accuracy_pct"]] + [f"{r['latency_ms']:.0f}"] for r in ablations])

budget_rows = table_after("## 5. 预算实验：固定比例与自适应上限")
budget = []
for row in budget_rows[:5]:
    ratio, count, success, latency = map(number, row)
    close(48 * ratio / 100, count, f"fixed {ratio:g}% query count")
    budget.append(dict(budget_pct=ratio, selected_queries=int(count), standard_success_pct=success, latency_ms=latency))
adaptive = dict(maximum_budget_pct=25, mean_selected_pct=23.7,
                approximate_mean_queries_from_rounded_ratio=48 * .237,
                standard_success_pct=number(budget_rows[-1][2]), latency_ms=number(budget_rows[-1][3]))
write_table("budget.tex", r"Measured budget sweep with 48 candidate queries. The first five rows fix the query count; the last row adapts it under a 25\% ceiling. The approximate adaptive mean count is derived from the rounded 23.7\% retention rate.", "tab:budget", "lrrr",
            ["Setting", "Queries retained", r"Success (\%)", "Latency (ms)"],
            [[f"Fixed {r['budget_pct']:g}\\%", str(r["selected_queries"]), f"{r['standard_success_pct']:.1f}", f"{r['latency_ms']:.0f}"] for r in budget] + [None, [r"Adaptive, ceiling 25\%", r"$\approx 11.38$ (mean)", "73.5", "89"]])

mechanism_names = ["Influence--action change Spearman", "Intervention Top-12 recall", "Latent--regret distance Spearman"]
controls = ["Without swaps", "Without swaps", "Without regret"]
mechanisms = [dict(metric=name, DIWA=number(row[1]), control_value=number(row[2]), control=control)
              for name, control, row in zip(mechanism_names, controls, table_after("### 8.2 相关性与干预排序结果"))]
write_table("mechanisms.tex", r"Mechanism diagnostics on independent test states or state pairs, reported as point estimates. Correlations use Spearman's $\rho$; Top-12 recall compares the predicted ranking with interventions on all 48 candidates.", "tab:mechanisms", "lrrl",
            ["Diagnostic", "DIWA", "Control", "Control variant"],
            [[name, f"{r['DIWA']:.0f}" if i == 1 else f"{r['DIWA']:.2f}",
              f"{r['control_value']:.0f}" if i == 1 else f"{r['control_value']:.2f}", r["control"]]
             for i, (name, r) in enumerate(zip([r"Influence--action change $\rho$", r"Top-12 recall (\%)",
                                               r"Latent--regret distance $\rho$"], mechanisms))])

write_table("seeds.tex", r"Measured simulation scores for each training seed and recomputed statistics. SD is the sample standard deviation across three training seeds; SEM equals SD$/\sqrt{3}$. Rates are percentages, and SD/SEM are percentage points.", "tab:seeds", "llrrrrrr",
            ["Method", "Platform", "42", "43", "44", "Mean", "SD", "SEM"],
            [[r["method"], r["platform"]] + [f"{v:.1f}" for v in r["measured_success_pct"]] + [f"{r['mean_pct']:.1f}", f"{r['sample_sd_pp']:.2f}", f"{r['sem_pp']:.2f}"] for r in seed_data], placement="htbp")
write_table("suite_seeds.tex", r"Measured LIBERO suite scores by training seed. Averaging the four suite scores within each seed recovers that seed's LIBERO score in Table~\ref{tab:seeds}.", "tab:suite_seeds", "llrrrrr",
            ["Method", "Suite", "42", "43", "44", "Mean", "SD"],
            [[r["method"], r["suite"]] + [f"{v:.1f}" for v in r["measured_success_pct"]] + [f"{r['mean_pct']:.1f}", f"{r['sample_sd_pp']:.2f}"] for r in suite_seeds], placement="htbp")
real_rows = []
for r in real:
    cells = [r["task"]]
    r["within_task_wilson95_pct"] = {}
    for method in ["DreamVLA", "DIWA"]:
        count, n = r[method + "_successes"], r["trials_per_method"]
        interval = wilson95(count, n)
        r["within_task_wilson95_pct"][method] = interval
        cells += [f"{count}/{n}", f"{100*count/n:.1f} [{interval[0]:.1f}, {interval[1]:.1f}]"]
    real_rows.append(cells)
write_table("real_robot.tex", r"Physical success counts and rates for the seed-42 checkpoints. Brackets give conditional within-task Wilson 95\% intervals for success rates (\%), assuming independent Bernoulli trials within a task. They do not measure training-seed variation. The total row reports aggregate rates without imposing a common success probability across tasks.", "tab:real_robot", "lrlrl",
            ["Task", "DreamVLA", r"Rate [95\% CI]", "DIWA", r"Rate [95\% CI]"],
            real_rows + [None, ["Total / mean", "196/300", "65.3", "216/300", "72.0"]], placement="htbp")

ledger = dict(
    revision_date="2026-09-25", status="Author-confirmed measurements; revision recomputes statistics and does not rerun experiments.",
    source=SOURCE.name, source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
    provenance=dict(measured="Platform and suite scores, seed scores, physical successes, ablations, budget measurements, mechanism diagnostics, and phase retention are author-confirmed measurements.",
                    computed="Means, sample SD, SEM, gains, ratios, and rounded adaptive mean query count are arithmetic derivatives.",
                    historical_derivations={"without_influence_standard_pct": "73.5 - 5.3 = 68.2", "without_influence_ood_pct": "75.8 - 9.7 = 66.1", "uniform_standard_pct": "73.5 - 2.9 = 70.6", "uniform_ood_pct": "75.8 - 6.9 = 68.9"}),
    evaluation=dict(training_seeds=[42,43,44], physical_checkpoint_seed=42,
                    LIBERO_episodes_per_method=6000, RoboTwin_episodes_per_method=3000,
                    RoboCasa_episodes_per_method=3000, physical_trials_per_method=300,
                    simulation_episodes_per_task_per_seed=50,
                    aggregation="Task means within platform, then training seeds; equal weight for four platforms. Real-robot value is a single checkpoint.",
                    uncertainty="Sample SD uses ddof=1 across training seeds. No synthetic uncertainty for the mixed four-platform aggregate."),
    platform_results=main, training_seed_results=seed_data, libero_suites=libero,
    libero_suite_seed_results=suite_seeds, ood_conditions=ood, real_robot_tasks=real,
    real_robot_totals=real_total, ablations=ablations, fixed_budget=budget, adaptive_budget=adaptive,
    mechanism_diagnostics=mechanisms,
    latency_ms={"DIWA (sparse)":89,"DIWA (dense)":213,"WorldVLA":241,"RoboDreamer":318},
    derived=dict(latency_reduction_vs_dense_pct=(213-89)/213*100, speedup_vs_dense=213/89,
                 latency_reduction_vs_worldvla_pct=(241-89)/241*100, speedup_vs_worldvla=241/89,
                 latency_reduction_vs_robodreamer_pct=(318-89)/318*100,
                 physical_gain_pp=(216-196)/300*100, future_query_reduction_pct=100-23.7,
                 theoretical_uniform_top12_recall_pct=12/48*100),
    separate_phase_measurements=dict(free_space_pct=15, contact_pct=31,
        manuscript_inclusion=False,
        note="Measured phase values retained in the internal ledger, excluded from the revised manuscript because run configuration and denominator are unspecified. Contact 31% cannot be assigned to the same 48-query hard-25%-ceiling configuration."),
    remaining_run_metadata=["GPU model and actual device count", "Timing boundaries, warm-up, repetitions, batch size, precision, and software versions",
        "Selected task IDs and trajectory-level splits", "OOD and CF label definitions, counts, and pairing",
        "Physical robot, cameras, low-level control, and candidate-return collection protocol",
        "142 ms ablation configuration and uniform Top-K index policy", "Budget sweep checkpoint/retraining mapping",
        "Phase-run ceiling and denominator", "Mechanism diagnostic sample counts and sampling protocol", "Actual RGB/video/heatmap files"],
    arithmetic_checks=dict(passed=len(checks), details=checks))
(ROOT / "evidence/results_ledger.json").write_text(json.dumps(ledger, ensure_ascii=False, indent=2) + "\n")
print(f"Generated 9 tables and results ledger; {len(checks)} arithmetic checks passed.")
