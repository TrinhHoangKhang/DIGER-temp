#!/usr/bin/env python3
import ast
import glob
import os
import re
import sys


TARGETS = {
    ("beauty", "frqud"): {"recall@5": 0.0440, "recall@10": 0.0683, "ndcg@5": 0.0294, "ndcg@10": 0.0372},
    ("beauty", "sdud"): {"recall@5": 0.0442, "recall@10": 0.0657, "ndcg@5": 0.0292, "ndcg@10": 0.0361},
    ("beauty", "both"): {"recall@5": 0.0439, "recall@10": 0.0696, "ndcg@5": 0.0293, "ndcg@10": 0.0376},
    ("instruments", "frqud"): {"recall@5": 0.0915, "recall@10": 0.1138, "ndcg@5": 0.0772, "ndcg@10": 0.0844},
    ("instruments", "sdud"): {"recall@5": 0.0905, "recall@10": 0.1124, "ndcg@5": 0.0753, "ndcg@10": 0.0823},
    ("instruments", "both"): {"recall@5": 0.0907, "recall@10": 0.1127, "ndcg@5": 0.0758, "ndcg@10": 0.0829},
    ("yelp", "frqud"): {"recall@5": 0.0266, "recall@10": 0.0432, "ndcg@5": 0.0173, "ndcg@10": 0.0227},
    ("yelp", "sdud"): {"recall@5": 0.0267, "recall@10": 0.0439, "ndcg@5": 0.0171, "ndcg@10": 0.0227},
    ("yelp", "both"): {"recall@5": 0.0273, "recall@10": 0.0437, "ndcg@5": 0.0175, "ndcg@10": 0.0227},
}

CONFIG_MATCHERS = {
    ("beauty", "frqud"): {
        "use_adaptive_selection": True,
        "use_learnable_sigma_gumbel": False,
        "hot_threshold_ratio": 1.5,
        "num_beams": 20,
        "gumbel_tau": 2.0,
    },
    ("beauty", "sdud"): {
        "use_adaptive_selection": False,
        "use_learnable_sigma_gumbel": True,
        "use_simple_uncertainty_loss": True,
        "sigma_lambda": 1.7,
        "initial_std": 2.0,
        "num_beams": 20,
        "gumbel_tau": 2.0,
    },
    ("beauty", "both"): {
        "use_adaptive_selection": True,
        "use_learnable_sigma_gumbel": True,
        "hot_threshold_ratio": 1.5,
        "sigma_reg_weight": 2.0,
        "initial_std": 1.0,
        "num_beams": 20,
        "gumbel_tau": 2.0,
    },
    ("instruments", "frqud"): {
        "use_adaptive_selection": True,
        "use_learnable_sigma_gumbel": False,
        "hot_threshold_ratio": 2.0,
        "num_beams": 20,
        "gumbel_tau": 2.0,
    },
    ("instruments", "sdud"): {
        "use_adaptive_selection": False,
        "use_learnable_sigma_gumbel": True,
        "use_simple_uncertainty_loss": True,
        "sigma_lambda": 1.8,
        "initial_std": 1.0,
        "num_beams": 20,
        "gumbel_tau": 2.0,
    },
    ("instruments", "both"): {
        "use_adaptive_selection": True,
        "use_learnable_sigma_gumbel": True,
        "use_simple_uncertainty_loss": True,
        "sigma_lambda": 1.8,
        "hot_threshold_ratio": 1.5,
        "initial_std": 1.5,
        "num_beams": 20,
        "gumbel_tau": 2.0,
    },
    ("yelp", "frqud"): {
        "use_adaptive_selection": True,
        "use_learnable_sigma_gumbel": False,
        "use_simple_uncertainty_loss": True,
        "sigma_lambda": 1.0,
        "initial_std": 2.0,
        "hot_threshold_ratio": 1.1,
        "num_beams": 80,
        "gumbel_tau": 1.5,
    },
    ("yelp", "sdud"): {
        "use_adaptive_selection": False,
        "use_learnable_sigma_gumbel": True,
        "use_simple_uncertainty_loss": True,
        "sigma_lambda": 1.7,
        "initial_std": 2.0,
        "hot_threshold_ratio": 1.1,
        "num_beams": 80,
        "gumbel_tau": 1.5,
    },
    ("yelp", "both"): {
        "use_adaptive_selection": True,
        "use_learnable_sigma_gumbel": True,
        "use_simple_uncertainty_loss": True,
        "sigma_lambda": 1.0,
        "initial_std": 2.0,
        "hot_threshold_ratio": 1.1,
        "num_beams": 80,
        "gumbel_tau": 1.5,
    },
}


def parse_pythonish_dict(line):
    match = re.search(r"\{.*\}", line)
    if not match:
        return None
    text = re.sub(r"device\(type='[^']+'\)", "'cuda'", match.group(0))
    text = re.sub(r"<accelerate\.[^>]+>", "'accelerator'", text)
    text = re.sub(r"np\.float64\(([^)]+)\)", r"\1", text)
    try:
        return ast.literal_eval(text)
    except Exception:
        return None


def close_enough(actual, target, rel_tol=0.01, abs_tol=0.0005):
    return abs(actual - target) <= max(abs_tol, abs(target) * rel_tol)


def log_variant(config):
    dataset = config.get("dataset")
    if dataset not in {"beauty", "instruments", "yelp"}:
        return None
    for key, matcher in CONFIG_MATCHERS.items():
        if key[0] != dataset:
            continue
        matched = True
        for name, expected in matcher.items():
            actual = config.get(name)
            if isinstance(expected, float):
                matched = actual is not None and abs(float(actual) - expected) < 1e-8
            else:
                matched = actual == expected
            if not matched:
                break
        if matched:
            return key
    return None


def main():
    latest = {}
    run_log_paths = sorted(
        set(
            glob.glob("logs/*/*.log")
            + glob.glob("reproduction_logs/*.log")
            + glob.glob("reproduction_logs/*.driver.log")
        )
    )
    if not run_log_paths and os.environ.get("VERIFY_WITH_BACKUP", "0") == "1":
        run_log_paths = sorted(glob.glob("backup/logs/*.log"))
        print("No fresh run logs found; using backup logs.")
    elif not run_log_paths:
        print("No fresh run logs found in logs/*/*.log or reproduction_logs/*.")
        print("Set VERIFY_WITH_BACKUP=1 to validate against backup logs.")
    log_paths = run_log_paths
    for path in log_paths:
        config = None
        result = None
        with open(path, "r", errors="ignore") as handle:
            for line in handle:
                if " Config: " in line:
                    config = parse_pythonish_dict(line)
                if "Test Results:" in line and "Val Results:" not in line:
                    parsed = parse_pythonish_dict(line)
                    if parsed:
                        result = parsed
        if not config or not result:
            continue
        variant = log_variant(config)
        if variant is None:
            continue
        mtime = os.path.getmtime(path)
        if variant not in latest or mtime > latest[variant][0]:
            latest[variant] = (mtime, path, result)

    failed = False
    for variant, targets in TARGETS.items():
        if variant not in latest:
            print(f"MISSING {variant[0]}:{variant[1]}")
            failed = True
            continue
        _, path, result = latest[variant]
        checks = []
        ok = True
        for metric, target in targets.items():
            actual = float(result[metric])
            metric_ok = close_enough(actual, target)
            ok = ok and metric_ok
            checks.append(f"{metric}={actual:.6f} target={target:.4f} {'OK' if metric_ok else 'FAIL'}")
        print(f"{'PASS' if ok else 'FAIL'} {variant[0]}:{variant[1]} {path}")
        print("  " + ", ".join(checks))
        failed = failed or not ok

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
