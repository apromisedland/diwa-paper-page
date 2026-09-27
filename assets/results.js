// Generated from the manuscript ledger by scripts/prepare-assets.mjs.
window.DIWA_DATA = {
  "platforms": [
    {
      "method": "Octo",
      "LIBERO": 73.2,
      "RoboTwin": 55.6,
      "RoboCasa": 47.2,
      "real_robot_pct": 50,
      "macro_pct": 56.5
    },
    {
      "method": "OpenVLA",
      "LIBERO": 76,
      "RoboTwin": 59.8,
      "RoboCasa": 50.199999999999996,
      "real_robot_pct": 56,
      "macro_pct": 60.5
    },
    {
      "method": "GR-1",
      "LIBERO": 77.4,
      "RoboTwin": 61.6,
      "RoboCasa": 53.4,
      "real_robot_pct": 58,
      "macro_pct": 62.6
    },
    {
      "method": "RoboDreamer",
      "LIBERO": 78.2,
      "RoboTwin": 64,
      "RoboCasa": 56.199999999999996,
      "real_robot_pct": 60,
      "macro_pct": 64.6
    },
    {
      "method": "WorldVLA",
      "LIBERO": 79,
      "RoboTwin": 65.8,
      "RoboCasa": 57.6,
      "real_robot_pct": 62,
      "macro_pct": 66.1
    },
    {
      "method": "DreamVLA",
      "LIBERO": 79.60000000000001,
      "RoboTwin": 67.6,
      "RoboCasa": 59.06666666666667,
      "real_robot_pct": 65.33333333333333,
      "macro_pct": 67.9
    },
    {
      "method": "DIWA",
      "LIBERO": 84.4,
      "RoboTwin": 73.2,
      "RoboCasa": 64.4,
      "real_robot_pct": 72,
      "macro_pct": 73.5
    }
  ],
  "ood": [
    {
      "condition": "Appearance",
      "DreamVLA": 65.8,
      "DIWA": 76.2,
      "gain_pp": 10.4
    },
    {
      "condition": "Background motion",
      "DreamVLA": 65,
      "DIWA": 74.8,
      "gain_pp": 9.8
    },
    {
      "condition": "Contact counterfactual",
      "DreamVLA": 64.8,
      "DIWA": 76.4,
      "gain_pp": 11.6
    },
    {
      "condition": "Equal-weight mean",
      "DreamVLA": 65.2,
      "DIWA": 75.8,
      "gain_pp": 10.6
    }
  ],
  "budgets": [
    {
      "budget_pct": 6.25,
      "selected_queries": 3,
      "standard_success_pct": 62.9,
      "latency_ms": 61
    },
    {
      "budget_pct": 12.5,
      "selected_queries": 6,
      "standard_success_pct": 69.8,
      "latency_ms": 71
    },
    {
      "budget_pct": 25,
      "selected_queries": 12,
      "standard_success_pct": 73.6,
      "latency_ms": 91
    },
    {
      "budget_pct": 50,
      "selected_queries": 24,
      "standard_success_pct": 74.1,
      "latency_ms": 132
    },
    {
      "budget_pct": 100,
      "selected_queries": 48,
      "standard_success_pct": 73.4,
      "latency_ms": 213
    }
  ],
  "adaptive": {
    "maximum_budget_pct": 25,
    "mean_selected_pct": 23.7,
    "approximate_mean_queries_from_rounded_ratio": 11.376,
    "standard_success_pct": 73.5,
    "latency_ms": 89
  },
  "ablations": [
    {
      "variant": "DIWA",
      "standard_pct": 73.5,
      "ood_pct": 75.8,
      "cf_accuracy_pct": 81.6,
      "latency_ms": 89
    },
    {
      "variant": "Without influence estimator",
      "standard_pct": 68.2,
      "ood_pct": 66.1,
      "cf_accuracy_pct": 71.9,
      "latency_ms": 142
    },
    {
      "variant": "Without cross-episode swaps",
      "standard_pct": 71.4,
      "ood_pct": 70.2,
      "cf_accuracy_pct": 72.8,
      "latency_ms": 88
    },
    {
      "variant": "Without regret geometry",
      "standard_pct": 69.3,
      "ood_pct": 68.2,
      "cf_accuracy_pct": 70.6,
      "latency_ms": 89
    },
    {
      "variant": "Dense imagination",
      "standard_pct": 73.4,
      "ood_pct": 75.5,
      "cf_accuracy_pct": 81.5,
      "latency_ms": 213
    },
    {
      "variant": "Uniform Top-$K$",
      "standard_pct": 70.6,
      "ood_pct": 68.9,
      "cf_accuracy_pct": 74.3,
      "latency_ms": 88
    }
  ],
  "latency": {
    "DIWA (sparse)": 89,
    "DIWA (dense)": 213,
    "WorldVLA": 241,
    "RoboDreamer": 318
  }
};
