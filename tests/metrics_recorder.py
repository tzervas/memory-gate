from datetime import datetime
import json
from pathlib import Path
import statistics
from typing import Any


class MetricsRecorder:
    """Enhanced metrics recorder for test runs with configurable presentation."""

    # Configuration for which metrics to display
    DEFAULT_DISPLAY_CONFIG = {
        "show_performance": True,
        "show_quality": True,
        "show_timing": True,
        "show_comparison": True,
        "show_trends": True,
        "max_history_runs": 10,
        "detailed_breakdown": False,
    }

    def __init__(
        self,
        metrics_file: str = "test_metrics.json",
        display_config: dict | None = None,
    ):
        self.metrics_file = Path(metrics_file)
        self.display_config = {**self.DEFAULT_DISPLAY_CONFIG, **(display_config or {})}
        self.metrics = {
            "performance": {},
            "quality": {},
            "timing": {},
            "errors": [],
            "metadata": {},
        }
        self.load_previous_metrics()

    def load_previous_metrics(self):
        """Load previous test run metrics from file."""
        if self.metrics_file.exists():
            try:
                with open(self.metrics_file) as file:
                    self.previous_runs = json.load(file)
                # Keep only the last N runs based on config
                max_runs = self.display_config["max_history_runs"]
                self.previous_runs = self.previous_runs[-max_runs:]
            except (OSError, json.JSONDecodeError) as e:
                print(f"Warning: Could not load previous metrics: {e}")
                self.previous_runs = []
        else:
            self.previous_runs = []

    def record_performance_metric(self, name: str, value: float, unit: str = "seconds"):
        """Record a performance-related metric."""
        self.metrics["performance"][name] = {"value": value, "unit": unit}

    def record_quality_metric(self, name: str, value: Any):
        """Record a quality-related metric."""
        self.metrics["quality"][name] = value

    def record_timing_metric(self, name: str, value: float):
        """Record a timing-related metric."""
        self.metrics["timing"][name] = value

    def record_error(self, test_name: str, error_type: str, message: str):
        """Record an error that occurred during testing."""
        self.metrics["errors"].append(
            {
                "test": test_name,
                "type": error_type,
                "message": message,
                "timestamp": datetime.now().isoformat(),
            }
        )

    def record_metadata(self, key: str, value: Any):
        """Record metadata about the test run."""
        self.metrics["metadata"][key] = value

    def calculate_statistics(
        self, metric_category: str, metric_name: str
    ) -> dict[str, float]:
        """Calculate statistics for a metric across previous runs."""
        values = []
        for run in self.previous_runs:
            if metric_category in run and metric_name in run[metric_category]:
                metric_data = run[metric_category][metric_name]
                value = (
                    metric_data["value"]
                    if isinstance(metric_data, dict)
                    else metric_data
                )
                if isinstance(value, (int, float)):
                    values.append(value)

        if not values:
            return {"count": 0}

        return {
            "count": len(values),
            "mean": statistics.mean(values),
            "median": statistics.median(values),
            "std_dev": statistics.stdev(values) if len(values) > 1 else 0.0,
            "min": min(values),
            "max": max(values),
        }

    def get_trend(self, metric_category: str, metric_name: str, last_n: int = 5) -> str:
        """Determine if a metric is trending up, down, or stable."""
        recent_runs = (
            self.previous_runs[-last_n:]
            if len(self.previous_runs) >= last_n
            else self.previous_runs
        )
        values = []

        for run in recent_runs:
            if metric_category in run and metric_name in run[metric_category]:
                metric_data = run[metric_category][metric_name]
                value = (
                    metric_data["value"]
                    if isinstance(metric_data, dict)
                    else metric_data
                )
                if isinstance(value, (int, float)):
                    values.append(value)

        if len(values) < 2:
            return "insufficient_data"

        # Simple linear trend calculation
        x = list(range(len(values)))
        y = values
        n = len(values)

        slope = (n * sum(x[i] * y[i] for i in range(n)) - sum(x) * sum(y)) / (
            n * sum(x[i] ** 2 for i in range(n)) - sum(x) ** 2
        )

        if abs(slope) < 0.01:  # Threshold for "stable"
            return "stable"
        return "increasing" if slope > 0 else "decreasing"

    def save_metrics(self):
        """Save current metrics to file."""
        current_run = {"timestamp": datetime.now().isoformat(), **self.metrics}
        self.previous_runs.append(current_run)

        # Keep only the last N runs
        max_runs = self.display_config["max_history_runs"]
        self.previous_runs = self.previous_runs[-max_runs:]

        try:
            with open(self.metrics_file, "w") as file:
                json.dump(self.previous_runs, file, indent=2)
        except OSError as e:
            print(f"Warning: Could not save metrics: {e}")

    def format_duration(self, seconds: float) -> str:
        """Format duration in a human-readable way."""
        if seconds < 1:
            return f"{seconds * 1000:.1f}ms"
        if seconds < 60:
            return f"{seconds:.2f}s"
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"

    def generate_report(self) -> str:
        """Generate a comprehensive report based on display configuration."""
        report_sections = []

        # Header
        report_sections.append("\n" + "=" * 80)
        report_sections.append("TEST METRICS REPORT")
        report_sections.append("=" * 80)

        # Current run summary
        if self.display_config["show_timing"]:
            report_sections.append("\n📊 TIMING METRICS:")
            for name, value in self.metrics["timing"].items():
                formatted_time = self.format_duration(value)
                report_sections.append(f"  • {name}: {formatted_time}")

        if self.display_config["show_performance"]:
            report_sections.append("\n⚡ PERFORMANCE METRICS:")
            for name, data in self.metrics["performance"].items():
                value = data["value"]
                unit = data["unit"]
                if unit == "seconds":
                    formatted_value = self.format_duration(value)
                else:
                    formatted_value = f"{value} {unit}"
                report_sections.append(f"  • {name}: {formatted_value}")

        if self.display_config["show_quality"]:
            report_sections.append("\n🎯 QUALITY METRICS:")
            for name, value in self.metrics["quality"].items():
                report_sections.append(f"  • {name}: {value}")

        # Errors
        if self.metrics["errors"]:
            report_sections.append("\n❌ ERRORS:")
            for error in self.metrics["errors"]:
                report_sections.append(
                    f"  • {error['test']}: {error['type']} - {error['message']}"
                )

        # Historical comparison
        if self.display_config["show_comparison"] and self.previous_runs:
            report_sections.append("\n📈 HISTORICAL COMPARISON:")

            # Compare timing metrics
            for name, value in self.metrics["timing"].items():
                stats = self.calculate_statistics("timing", name)
                if stats["count"] > 0:
                    if stats["mean"] != 0:  # Prevent division by zero
                        current_vs_avg = ((value - stats["mean"]) / stats["mean"]) * 100
                    else:
                        current_vs_avg = 0
                    trend_symbol = (
                        "📈"
                        if current_vs_avg > 5
                        else "📉" if current_vs_avg < -5 else "➡️"
                    )
                    report_sections.append(
                        f"  {trend_symbol} {name}: {self.format_duration(value)} (avg: {self.format_duration(stats['mean'])}, {current_vs_avg:+.1f}%)"
                    )

        # Trends
        if self.display_config["show_trends"] and len(self.previous_runs) >= 2:
            report_sections.append("\n📊 TRENDS (last 5 runs):")
            for name in self.metrics["timing"].keys():
                trend = self.get_trend("timing", name)
                trend_emoji = {
                    "increasing": "📈",
                    "decreasing": "📉",
                    "stable": "➡️",
                    "insufficient_data": "❓",
                }[trend]
                report_sections.append(f"  {trend_emoji} {name}: {trend}")

        # Metadata
        if self.metrics["metadata"]:
            report_sections.append("\n📋 METADATA:")
            for key, value in self.metrics["metadata"].items():
                report_sections.append(f"  • {key}: {value}")

        report_sections.append("\n" + "=" * 80)

        return "\n".join(report_sections)

    def toggle_display_config(self, **kwargs):
        """Update display configuration."""
        self.display_config.update(kwargs)

    def get_summary_metrics(self) -> dict[str, Any]:
        """Get a summary of key metrics for quick review."""
        summary = {}

        # Total timing
        if "Total Run Time" in self.metrics["timing"]:
            summary["total_time"] = self.format_duration(
                self.metrics["timing"]["Total Run Time"]
            )

        # Error count
        summary["error_count"] = len(self.metrics["errors"])

        # Quality metrics count
        summary["quality_metrics_count"] = len(self.metrics["quality"])

        return summary
