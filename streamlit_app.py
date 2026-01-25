"""
Power BI Consumption Cost Breakpoints: Snowflake vs Fabric Serving Layer

A steering model to help data platform teams understand when consolidating
semantic models on a Fabric serving layer becomes more cost-effective than
letting multiple Power BI models query Snowflake directly.

Key insight: As model count grows, Architecture A (Snowflake-fed sprawl) incurs
duplicated compute, spiky concurrency, and higher operational overhead. Architecture B
(Fabric serving layer) amortizes ETL cost and improves reuse.

Run with: streamlit run streamlit_app.py

Author: Data Platform Engineering
"""

import streamlit as st
import streamlit.components.v1 as components
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


# =============================================================================
# MERMAID DIAGRAM RENDERING
# =============================================================================

def render_mermaid(mermaid_code: str, height: int = 400) -> None:
    """Render a Mermaid diagram using HTML components."""
    html_code = f"""
    <html>
    <head>
        <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
        <script>mermaid.initialize({{startOnLoad:true, theme:'neutral'}});</script>
    </head>
    <body>
        <div class="mermaid">
{mermaid_code}
        </div>
    </body>
    </html>
    """
    components.html(html_code, height=height)


# =============================================================================
# CONSTANTS AND PRICING DEFAULTS
# =============================================================================

# Snowflake warehouse size to credits per hour mapping (standard compute pricing)
# These are approximate and vary by edition/region/contract
SNOWFLAKE_WAREHOUSE_CREDITS = {
    "XS": 1,
    "S": 2,
    "M": 4,
    "L": 8,
    "XL": 16,
    "2XL": 32,
    "3XL": 64,
    "4XL": 128,
}

# Fabric SKU to CU (Capacity Units) mapping
FABRIC_SKU_CUS = {
    "F2": 2,
    "F4": 4,
    "F8": 8,
    "F16": 16,
    "F32": 32,
    "F64": 64,
    "F128": 128,
    "F256": 256,
    "F512": 512,
}

# Default pricing assumptions (USD)
DEFAULT_SNOWFLAKE_PRICE_PER_CREDIT = 3.00  # Varies by contract, typically $2-$4
DEFAULT_FABRIC_CU_PRICE_PER_HOUR = 0.18    # Azure list price, varies by region
DEFAULT_HOURS_PER_MONTH = 730              # ~30.4 days * 24 hours
DEFAULT_POWER_BI_PRO_PRICE = 14.00         # Per user per month
DEFAULT_POWER_BI_PPU_PRICE = 24.00         # Per user per month (Premium Per User)
DEFAULT_EGRESS_PRICE_PER_TB = 90.00        # Snowflake egress, varies by cloud/region
DEFAULT_OPS_UNIT_COST = 150.00             # Proxy cost per model-change event
DEFAULT_AVG_REFRESH_RUNTIME_HOURS = 0.25   # 15 minutes average refresh time


@dataclass
class ScenarioParams:
    """Parameters defining a cost comparison scenario."""

    # Model and user counts
    num_models: int = 10
    creators_count: int = 5
    viewers_count: int = 50

    # Usage patterns
    refreshes_per_model_per_day: float = 4.0
    interactive_query_hours_per_day: float = 8.0
    avg_refresh_runtime_hours: float = DEFAULT_AVG_REFRESH_RUNTIME_HOURS

    # Snowflake configuration
    avg_warehouse_size: str = "M"
    avg_clusters: int = 2
    snowflake_price_per_credit: float = DEFAULT_SNOWFLAKE_PRICE_PER_CREDIT
    warehouse_idle_drift_factor: float = 1.3  # Accounts for warehouse staying warm

    # Fabric configuration
    fabric_sku: str = "F64"
    fabric_cu_price_per_hour: float = DEFAULT_FABRIC_CU_PRICE_PER_HOUR
    fabric_hours_per_month: int = DEFAULT_HOURS_PER_MONTH
    fabric_paused_percent: float = 0.0  # Percent of hours paused (0-100)

    # Licensing
    power_bi_pro_price: float = DEFAULT_POWER_BI_PRO_PRICE
    power_bi_ppu_price: float = DEFAULT_POWER_BI_PPU_PRICE
    creators_use_ppu: bool = False

    # Egress (optional)
    egress_applicable: bool = False
    egress_tb_per_month: float = 1.0
    egress_price_per_tb: float = DEFAULT_EGRESS_PRICE_PER_TB

    # Operational overhead factors
    reuse_factor: float = 0.3  # 0-1, higher = more shared/certified reuse
    change_rate_per_month: int = 5  # Number of logic changes per model per month
    ops_unit_cost: float = DEFAULT_OPS_UNIT_COST
    shared_pipeline_count: int = 5  # Number of shared pipelines in Arch B
    base_ops_cost: float = 500.0  # Base operational cost


@dataclass
class CostBreakdown:
    """Detailed cost breakdown for an architecture."""

    compute_cost: float = 0.0
    licensing_cost: float = 0.0
    egress_cost: float = 0.0
    ops_overhead_cost: float = 0.0
    fabric_serving_cost: float = 0.0  # Only for Architecture B

    @property
    def total_cost(self) -> float:
        return (
            self.compute_cost
            + self.licensing_cost
            + self.egress_cost
            + self.ops_overhead_cost
            + self.fabric_serving_cost
        )


def compute_snowflake_hours(params: ScenarioParams) -> float:
    """
    Calculate estimated Snowflake warehouse running hours per month.

    Components:
    - Refresh hours: time spent refreshing semantic models
    - Interactive hours: time users actively query
    - Idle drift: warehouse staying warm between queries

    Returns:
        Estimated hours per month the warehouse is running.
    """
    # Refresh hours: num_models * refreshes_per_day * days * avg_runtime
    refresh_hours = (
        params.num_models
        * params.refreshes_per_model_per_day
        * 30  # days per month
        * params.avg_refresh_runtime_hours
    )

    # Interactive hours: hours per day * days
    interactive_hours = params.interactive_query_hours_per_day * 30

    # Apply idle drift factor (warehouse stays warm, doesn't auto-suspend perfectly)
    raw_hours = (refresh_hours + interactive_hours) * params.warehouse_idle_drift_factor

    # Cap at maximum hours per month
    return min(params.fabric_hours_per_month, raw_hours)


def compute_architecture_a_costs(params: ScenarioParams) -> CostBreakdown:
    """
    Compute costs for Architecture A: Snowflake-fed sprawl.

    Each Power BI semantic model connects directly to Snowflake,
    causing duplicated compute and higher operational overhead.
    """
    breakdown = CostBreakdown()

    # --- Snowflake Compute Cost ---
    credits_per_hour = SNOWFLAKE_WAREHOUSE_CREDITS.get(params.avg_warehouse_size, 4)
    hours_running = compute_snowflake_hours(params)

    breakdown.compute_cost = (
        credits_per_hour
        * hours_running
        * params.avg_clusters
        * params.snowflake_price_per_credit
    )

    # --- Power BI Licensing ---
    # Creators: Pro or PPU
    creator_price = (
        params.power_bi_ppu_price if params.creators_use_ppu else params.power_bi_pro_price
    )
    creators_cost = params.creators_count * creator_price

    # Viewers: Always Pro in Architecture A (no Premium capacity)
    viewers_cost = params.viewers_count * params.power_bi_pro_price

    breakdown.licensing_cost = creators_cost + viewers_cost

    # --- Egress Cost (optional) ---
    if params.egress_applicable:
        breakdown.egress_cost = params.egress_tb_per_month * params.egress_price_per_tb

    # --- Operational Overhead (proxy) ---
    # More models with less reuse = more duplicated work and change management
    duplication_factor = 1 - params.reuse_factor
    breakdown.ops_overhead_cost = (
        params.base_ops_cost
        + params.num_models
        * params.change_rate_per_month
        * duplication_factor
        * params.ops_unit_cost
    )

    return breakdown


def compute_architecture_b_costs(params: ScenarioParams) -> CostBreakdown:
    """
    Compute costs for Architecture B: Fabric serving layer.

    Curated data is landed into Fabric (Lakehouse/Warehouse) once,
    then semantic models sit on top of Fabric. This improves reuse
    and lowers marginal cost per additional model.
    """
    breakdown = CostBreakdown()

    # --- Fabric Serving Cost ---
    fabric_cus = FABRIC_SKU_CUS.get(params.fabric_sku, 64)
    effective_hours = params.fabric_hours_per_month * (1 - params.fabric_paused_percent / 100)

    breakdown.fabric_serving_cost = (
        fabric_cus * params.fabric_cu_price_per_hour * effective_hours
    )

    # --- Power BI Licensing ---
    # Creators: Pro or PPU
    creator_price = (
        params.power_bi_ppu_price if params.creators_use_ppu else params.power_bi_pro_price
    )
    creators_cost = params.creators_count * creator_price

    # Viewers: Free if Fabric >= F64, otherwise Pro
    # F64+ allows unlimited free viewers (embedded/app scenarios)
    if fabric_cus >= 64:
        viewers_cost = 0.0
    else:
        viewers_cost = params.viewers_count * params.power_bi_pro_price

    breakdown.licensing_cost = creators_cost + viewers_cost

    # --- Compute Cost (minimal Snowflake for source extraction only) ---
    # In Architecture B, Snowflake usage is much lower - just for landing data
    # Assume ~10% of the compute compared to Architecture A
    credits_per_hour = SNOWFLAKE_WAREHOUSE_CREDITS.get(params.avg_warehouse_size, 4)
    # Fixed ETL window, not scaling with num_models
    etl_hours_per_month = min(100, 2 * 30)  # ~2 hours/day for ETL

    breakdown.compute_cost = (
        credits_per_hour
        * etl_hours_per_month
        * 1  # Single cluster for ETL
        * params.snowflake_price_per_credit
    )

    # --- Egress Cost (still applicable, but potentially lower with batching) ---
    if params.egress_applicable:
        # Assume batched egress is ~70% of sprawl egress
        breakdown.egress_cost = params.egress_tb_per_month * params.egress_price_per_tb * 0.7

    # --- Operational Overhead (lower due to centralization) ---
    # Shared pipelines mean changes are managed centrally
    breakdown.ops_overhead_cost = (
        params.base_ops_cost
        + params.shared_pipeline_count
        * params.change_rate_per_month
        * params.ops_unit_cost
        * 0.5  # Lower unit cost due to standardization
    )

    return breakdown


def compute_costs(params: ScenarioParams) -> Tuple[CostBreakdown, CostBreakdown]:
    """
    Compute costs for both architectures.

    Returns:
        Tuple of (Architecture A costs, Architecture B costs)
    """
    cost_a = compute_architecture_a_costs(params)
    cost_b = compute_architecture_b_costs(params)
    return cost_a, cost_b


def sweep_models(
    params: ScenarioParams,
    max_models: int = 100,
    step: int = 1
) -> pd.DataFrame:
    """
    Sweep across different num_models values to find breakpoint.

    Returns:
        DataFrame with columns: num_models, cost_a, cost_b
    """
    results = []

    for n in range(1, max_models + 1, step):
        sweep_params = ScenarioParams(
            num_models=n,
            creators_count=params.creators_count,
            viewers_count=params.viewers_count,
            refreshes_per_model_per_day=params.refreshes_per_model_per_day,
            interactive_query_hours_per_day=params.interactive_query_hours_per_day,
            avg_refresh_runtime_hours=params.avg_refresh_runtime_hours,
            avg_warehouse_size=params.avg_warehouse_size,
            avg_clusters=params.avg_clusters,
            snowflake_price_per_credit=params.snowflake_price_per_credit,
            warehouse_idle_drift_factor=params.warehouse_idle_drift_factor,
            fabric_sku=params.fabric_sku,
            fabric_cu_price_per_hour=params.fabric_cu_price_per_hour,
            fabric_hours_per_month=params.fabric_hours_per_month,
            fabric_paused_percent=params.fabric_paused_percent,
            power_bi_pro_price=params.power_bi_pro_price,
            power_bi_ppu_price=params.power_bi_ppu_price,
            creators_use_ppu=params.creators_use_ppu,
            egress_applicable=params.egress_applicable,
            egress_tb_per_month=params.egress_tb_per_month,
            egress_price_per_tb=params.egress_price_per_tb,
            reuse_factor=params.reuse_factor,
            change_rate_per_month=params.change_rate_per_month,
            ops_unit_cost=params.ops_unit_cost,
            shared_pipeline_count=params.shared_pipeline_count,
            base_ops_cost=params.base_ops_cost,
        )

        cost_a, cost_b = compute_costs(sweep_params)

        results.append({
            "num_models": n,
            "cost_a": cost_a.total_cost,
            "cost_b": cost_b.total_cost,
            "compute_a": cost_a.compute_cost,
            "compute_b": cost_b.compute_cost,
            "licensing_a": cost_a.licensing_cost,
            "licensing_b": cost_b.licensing_cost,
            "ops_a": cost_a.ops_overhead_cost,
            "ops_b": cost_b.ops_overhead_cost,
        })

    return pd.DataFrame(results)


def find_breakpoint(sweep_df: pd.DataFrame) -> Optional[int]:
    """
    Find the model count where Architecture B becomes cheaper.

    Returns:
        Number of models at breakpoint, or None if no crossover.
    """
    # Find where cost_b becomes less than cost_a
    crossover = sweep_df[sweep_df["cost_b"] < sweep_df["cost_a"]]

    if len(crossover) == 0:
        return None

    return int(crossover.iloc[0]["num_models"])


def plot_breakpoint(sweep_df: pd.DataFrame, current_models: int) -> plt.Figure:
    """
    Create the breakpoint chart showing cost curves for both architectures.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(
        sweep_df["num_models"],
        sweep_df["cost_a"],
        label="Architecture A: Snowflake-fed sprawl",
        linewidth=2,
        marker="o",
        markevery=10,
    )
    ax.plot(
        sweep_df["num_models"],
        sweep_df["cost_b"],
        label="Architecture B: Fabric serving layer",
        linewidth=2,
        marker="s",
        markevery=10,
    )

    # Find and annotate breakpoint
    breakpoint = find_breakpoint(sweep_df)
    if breakpoint is not None:
        bp_cost = sweep_df[sweep_df["num_models"] == breakpoint]["cost_a"].values[0]
        ax.axvline(x=breakpoint, color="gray", linestyle="--", alpha=0.7)
        ax.annotate(
            f"Breakeven: ~{breakpoint} models",
            xy=(breakpoint, bp_cost),
            xytext=(breakpoint + 5, bp_cost * 1.1),
            fontsize=10,
            arrowprops=dict(arrowstyle="->", color="gray"),
        )

    # Mark current position
    ax.axvline(x=current_models, color="red", linestyle=":", alpha=0.7, linewidth=2)
    ax.annotate(
        f"Current: {current_models} models",
        xy=(current_models, ax.get_ylim()[1] * 0.95),
        fontsize=9,
        color="red",
        ha="center",
    )

    ax.set_xlabel("Number of Semantic Models", fontsize=11)
    ax.set_ylabel("Monthly Cost (USD)", fontsize=11)
    ax.set_title("Cost Breakpoint Analysis: Snowflake vs Fabric", fontsize=13)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    plt.tight_layout()
    return fig


def plot_cost_breakdown(cost_a: CostBreakdown, cost_b: CostBreakdown) -> plt.Figure:
    """
    Create a stacked bar chart comparing cost components.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    categories = ["Compute", "Licensing", "Egress", "Ops Overhead", "Fabric Serving"]

    # Architecture A values
    values_a = [
        cost_a.compute_cost,
        cost_a.licensing_cost,
        cost_a.egress_cost,
        cost_a.ops_overhead_cost,
        0,  # No Fabric serving cost
    ]

    # Architecture B values
    values_b = [
        cost_b.compute_cost,
        cost_b.licensing_cost,
        cost_b.egress_cost,
        cost_b.ops_overhead_cost,
        cost_b.fabric_serving_cost,
    ]

    width = 0.6

    # Create stacked bars
    bottom_a = 0.0
    bottom_b = 0.0

    colors = plt.cm.tab10.colors

    for i, (cat, val_a, val_b) in enumerate(zip(categories, values_a, values_b)):
        if val_a > 0 or val_b > 0:  # Only show non-zero categories
            ax.bar(0, val_a, width, bottom=bottom_a, label=cat, color=colors[i])
            ax.bar(1, val_b, width, bottom=bottom_b, color=colors[i])
            bottom_a += val_a
            bottom_b += val_b

    # Add total labels on top
    ax.text(0, cost_a.total_cost + 100, f"${cost_a.total_cost:,.0f}", ha="center", fontsize=11, fontweight="bold")
    ax.text(1, cost_b.total_cost + 100, f"${cost_b.total_cost:,.0f}", ha="center", fontsize=11, fontweight="bold")

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["A: Snowflake-fed\nSprawl", "B: Fabric\nServing Layer"], fontsize=11)
    ax.set_ylabel("Monthly Cost (USD)", fontsize=11)
    ax.set_title("Cost Breakdown by Component", fontsize=13)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    plt.tight_layout()
    return fig


def plot_quiver(params: ScenarioParams, sweep_df: pd.DataFrame) -> plt.Figure:
    """
    Create a simplified 2D phase portrait showing cost dynamics.

    X-axis: Model sprawl (normalized num_models)
    Y-axis: Unit cost per model

    Arrows show direction of cost pressure as models increase.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create a working copy to avoid modifying the original
    df = sweep_df.copy()

    # Create grid
    max_models = df["num_models"].max()

    # Calculate unit costs (avoid division by zero)
    df["unit_cost_a"] = df["cost_a"] / df["num_models"].clip(lower=1)
    df["unit_cost_b"] = df["cost_b"] / df["num_models"].clip(lower=1)

    # Normalize model count to 0-1
    df["normalized_models"] = df["num_models"] / max_models

    # Sample points for quiver plot
    sample_indices = np.linspace(0, len(df) - 2, 10, dtype=int)

    # Architecture A vectors (tend to drift up-right with low reuse)
    for idx in sample_indices:
        x = df.iloc[idx]["normalized_models"]
        y = df.iloc[idx]["unit_cost_a"]

        # Vector direction: rightward (more models) and upward (higher unit cost)
        dx = 0.05
        dy = (df.iloc[idx + 1]["unit_cost_a"] - y) * 2  # Amplify for visibility

        ax.quiver(x, y, dx, dy, angles="xy", scale_units="xy", scale=1,
                  color="coral", alpha=0.7, width=0.008)

    # Architecture B vectors (tend toward stable region)
    for idx in sample_indices:
        x = df.iloc[idx]["normalized_models"]
        y = df.iloc[idx]["unit_cost_b"]

        dx = 0.05
        dy = (df.iloc[idx + 1]["unit_cost_b"] - y) * 2

        ax.quiver(x, y, dx, dy, angles="xy", scale_units="xy", scale=1,
                  color="steelblue", alpha=0.7, width=0.008)

    # Plot the actual curves
    ax.plot(
        df["normalized_models"],
        df["unit_cost_a"],
        label="A: Snowflake-fed (unit cost)",
        linewidth=2,
        color="coral",
    )
    ax.plot(
        df["normalized_models"],
        df["unit_cost_b"],
        label="B: Fabric serving (unit cost)",
        linewidth=2,
        color="steelblue",
    )

    # Mark current position
    current_norm = params.num_models / max_models
    ax.axvline(x=current_norm, color="gray", linestyle=":", alpha=0.7)

    ax.set_xlabel("Model Sprawl (normalized)", fontsize=11)
    ax.set_ylabel("Unit Cost per Model (USD/model/month)", fontsize=11)
    ax.set_title("Cost Dynamics: Unit Cost vs Scale", fontsize=13)
    ax.legend(loc="upper right", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Format y-axis as currency
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"${x:,.0f}"))

    plt.tight_layout()
    return fig


def get_scenario_preset(preset_name: str) -> Dict[str, Any]:
    """Return parameter values for predefined scenarios."""
    presets = {
        "Today": {
            "num_models": 8,
            "creators_count": 3,
            "viewers_count": 25,
            "refreshes_per_model_per_day": 2.0,
            "interactive_query_hours_per_day": 4.0,
            "avg_warehouse_size": "S",
            "avg_clusters": 1,
            "fabric_sku": "F32",
            "reuse_factor": 0.2,
            "change_rate_per_month": 3,
        },
        "6-month growth": {
            "num_models": 25,
            "creators_count": 8,
            "viewers_count": 100,
            "refreshes_per_model_per_day": 4.0,
            "interactive_query_hours_per_day": 8.0,
            "avg_warehouse_size": "M",
            "avg_clusters": 2,
            "fabric_sku": "F64",
            "reuse_factor": 0.3,
            "change_rate_per_month": 5,
        },
        "Enterprise scale": {
            "num_models": 75,
            "creators_count": 20,
            "viewers_count": 500,
            "refreshes_per_model_per_day": 6.0,
            "interactive_query_hours_per_day": 12.0,
            "avg_warehouse_size": "L",
            "avg_clusters": 4,
            "fabric_sku": "F128",
            "reuse_factor": 0.4,
            "change_rate_per_month": 8,
        },
    }
    return presets.get(preset_name, presets["Today"])


def main():
    """Main Streamlit application."""

    st.set_page_config(
        page_title="Power BI Cost Breakpoints: Snowflake vs Fabric",
        page_icon="📊",
        layout="wide",
    )

    st.title("Power BI Consumption Cost Breakpoints: Snowflake vs Fabric Serving Layer")
    st.markdown("""
    This tool helps you understand when consolidating semantic models on a
    **Fabric serving layer** becomes more cost-effective than letting multiple
    Power BI models query **Snowflake directly**.
    """)

    # Initialize session state for preset handling
    if "preset_applied" not in st.session_state:
        st.session_state.preset_applied = None

    # ==========================================================================
    # MAIN TABS
    # ==========================================================================

    tab_cost, tab_governance = st.tabs(["Cost Analysis", "Architecture & Governance Case"])

    # ==========================================================================
    # SIDEBAR: Configuration
    # ==========================================================================

    with st.sidebar:
        st.header("Configuration")

        # Scenario Presets
        st.subheader("Scenario Presets")
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("Today", use_container_width=True):
                st.session_state.preset_applied = "Today"
                st.rerun()
        with col2:
            if st.button("6-mo Growth", use_container_width=True):
                st.session_state.preset_applied = "6-month growth"
                st.rerun()
        with col3:
            if st.button("Enterprise", use_container_width=True):
                st.session_state.preset_applied = "Enterprise scale"
                st.rerun()

        # Load preset values if one was selected
        preset_values = {}
        if st.session_state.preset_applied:
            preset_values = get_scenario_preset(st.session_state.preset_applied)
            st.info(f"Preset: {st.session_state.preset_applied}")

        st.divider()

        # --- Model & User Counts ---
        st.subheader("Models & Users")

        num_models = st.slider(
            "Number of Semantic Models",
            min_value=1,
            max_value=150,
            value=preset_values.get("num_models", 10),
            help="Total Power BI semantic models connecting to data source",
        )

        creators_count = st.slider(
            "Report Creators",
            min_value=1,
            max_value=100,
            value=preset_values.get("creators_count", 5),
            help="Users who build reports and semantic models",
        )

        viewers_count = st.slider(
            "Report Viewers",
            min_value=0,
            max_value=2000,
            value=preset_values.get("viewers_count", 50),
            help="Users who consume reports",
        )

        st.divider()

        # --- Usage Patterns ---
        st.subheader("Usage Patterns")

        refreshes_per_model_per_day = st.slider(
            "Refreshes per Model per Day",
            min_value=0.5,
            max_value=24.0,
            value=preset_values.get("refreshes_per_model_per_day", 4.0),
            step=0.5,
            help="Average data refresh frequency per semantic model",
        )

        interactive_query_hours_per_day = st.slider(
            "Interactive Query Hours/Day",
            min_value=0.0,
            max_value=24.0,
            value=preset_values.get("interactive_query_hours_per_day", 8.0),
            step=0.5,
            help="Hours per day when users actively query dashboards",
        )

        st.divider()

        # --- Snowflake Configuration ---
        st.subheader("Snowflake Config")

        avg_warehouse_size = st.selectbox(
            "Average Warehouse Size",
            options=list(SNOWFLAKE_WAREHOUSE_CREDITS.keys()),
            index=list(SNOWFLAKE_WAREHOUSE_CREDITS.keys()).index(
                preset_values.get("avg_warehouse_size", "M")
            ),
            help="Typical warehouse size serving BI workloads",
        )

        avg_clusters = st.slider(
            "Multi-cluster Warehouses",
            min_value=1,
            max_value=10,
            value=preset_values.get("avg_clusters", 2),
            help="Number of clusters for concurrency scaling",
        )

        snowflake_price_per_credit = st.slider(
            "Snowflake $/Credit",
            min_value=1.50,
            max_value=5.00,
            value=3.00,
            step=0.10,
            help="Your contracted Snowflake credit price (typically $2-$4)",
        )

        warehouse_idle_drift_factor = st.slider(
            "Warehouse Idle Drift Factor",
            min_value=1.0,
            max_value=2.0,
            value=1.3,
            step=0.1,
            help="Multiplier for warehouse staying warm between queries (1.0 = perfect auto-suspend)",
        )

        st.divider()

        # --- Fabric Configuration ---
        st.subheader("Fabric Config")

        fabric_sku = st.selectbox(
            "Fabric SKU",
            options=list(FABRIC_SKU_CUS.keys()),
            index=list(FABRIC_SKU_CUS.keys()).index(
                preset_values.get("fabric_sku", "F64")
            ),
            help="Fabric capacity SKU (F64+ enables free viewers)",
        )

        fabric_cu_price_per_hour = st.slider(
            "Fabric CU $/Hour",
            min_value=0.10,
            max_value=0.50,
            value=0.18,
            step=0.01,
            help="Azure Fabric pricing per CU-hour",
        )

        fabric_paused_percent = st.slider(
            "Fabric Paused Off-Hours %",
            min_value=0,
            max_value=70,
            value=0,
            help="Percent of month Fabric is paused (nights/weekends)",
        )

        st.divider()

        # --- Power BI Licensing ---
        st.subheader("Power BI Licensing")

        power_bi_pro_price = st.number_input(
            "Pro License $/User/Month",
            min_value=0.0,
            max_value=50.0,
            value=14.0,
            step=1.0,
        )

        power_bi_ppu_price = st.number_input(
            "PPU License $/User/Month",
            min_value=0.0,
            max_value=50.0,
            value=24.0,
            step=1.0,
        )

        creators_use_ppu = st.checkbox(
            "Creators use PPU (vs Pro)",
            value=False,
            help="Premium Per User for creators enables more features",
        )

        st.divider()

        # --- Egress (Optional) ---
        st.subheader("Egress (Optional)")

        egress_applicable = st.checkbox(
            "Include Egress Costs",
            value=False,
            help="Model cross-cloud/region egress charges",
        )

        if egress_applicable:
            egress_tb_per_month = st.slider(
                "Egress TB/Month",
                min_value=0.1,
                max_value=50.0,
                value=1.0,
                step=0.1,
            )
            egress_price_per_tb = st.slider(
                "Egress $/TB",
                min_value=0.0,
                max_value=200.0,
                value=90.0,
                step=5.0,
            )
        else:
            egress_tb_per_month = 0.0
            egress_price_per_tb = 90.0

        st.divider()

        # --- Operational Overhead ---
        st.subheader("Operational Overhead")

        reuse_factor = st.slider(
            "Reuse Factor",
            min_value=0.0,
            max_value=1.0,
            value=preset_values.get("reuse_factor", 0.3),
            step=0.05,
            help="0 = no reuse (duplicated logic), 1 = fully certified/shared",
        )

        change_rate_per_month = st.slider(
            "Logic Changes/Month/Model",
            min_value=0,
            max_value=20,
            value=preset_values.get("change_rate_per_month", 5),
            help="How often model logic changes (impacts ops cost)",
        )

        ops_unit_cost = st.number_input(
            "Ops Cost per Change Event ($)",
            min_value=0.0,
            max_value=500.0,
            value=150.0,
            step=10.0,
            help="Proxy cost for each model-change event (dev time, testing, etc.)",
        )

    # ==========================================================================
    # BUILD SCENARIO AND COMPUTE COSTS
    # ==========================================================================

    params = ScenarioParams(
        num_models=num_models,
        creators_count=creators_count,
        viewers_count=viewers_count,
        refreshes_per_model_per_day=refreshes_per_model_per_day,
        interactive_query_hours_per_day=interactive_query_hours_per_day,
        avg_warehouse_size=avg_warehouse_size,
        avg_clusters=avg_clusters,
        snowflake_price_per_credit=snowflake_price_per_credit,
        warehouse_idle_drift_factor=warehouse_idle_drift_factor,
        fabric_sku=fabric_sku,
        fabric_cu_price_per_hour=fabric_cu_price_per_hour,
        fabric_paused_percent=fabric_paused_percent,
        power_bi_pro_price=power_bi_pro_price,
        power_bi_ppu_price=power_bi_ppu_price,
        creators_use_ppu=creators_use_ppu,
        egress_applicable=egress_applicable,
        egress_tb_per_month=egress_tb_per_month,
        egress_price_per_tb=egress_price_per_tb,
        reuse_factor=reuse_factor,
        change_rate_per_month=change_rate_per_month,
        ops_unit_cost=ops_unit_cost,
    )

    cost_a, cost_b = compute_costs(params)
    sweep_df = sweep_models(params, max_models=100)
    breakpoint = find_breakpoint(sweep_df)

    # ==========================================================================
    # TAB 1: COST ANALYSIS
    # ==========================================================================

    # Compute winner for both tabs
    cost_diff = cost_a.total_cost - cost_b.total_cost
    winner = "B" if cost_diff > 0 else "A"

    # ==========================================================================
    # TAB 1: COST ANALYSIS
    # ==========================================================================

    with tab_cost:

        # ======================================================================
        # SECTION A: Executive Summary
        # ======================================================================

        st.header("A. Executive Summary")

        # Determine color scheme
        if winner == "B":
            recommendation = "Architecture B (Fabric) is recommended"
        else:
            recommendation = "Architecture A (Snowflake) is currently cheaper"

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                label="Architecture A: Snowflake-fed",
                value=f"${cost_a.total_cost:,.0f}/mo",
                delta=f"-${abs(cost_diff):,.0f}" if winner == "A" else f"+${abs(cost_diff):,.0f}",
                delta_color="normal" if winner == "A" else "inverse",
            )

        with col2:
            st.metric(
                label="Architecture B: Fabric Serving",
                value=f"${cost_b.total_cost:,.0f}/mo",
                delta=f"-${abs(cost_diff):,.0f}" if winner == "B" else f"+${abs(cost_diff):,.0f}",
                delta_color="normal" if winner == "B" else "inverse",
            )

        with col3:
            if breakpoint:
                st.metric(
                    label="Breakeven Point",
                    value=f"{breakpoint} models",
                    delta=f"Currently at {num_models}",
                )
            else:
                st.metric(
                    label="Breakeven Point",
                    value="N/A",
                    delta="No crossover in range",
                )

        # Big summary card
        if winner == "B":
            st.success(
                f"**{recommendation}**: Architecture B is **${abs(cost_diff):,.0f}/month cheaper** "
                f"and offers lower duplication/ops risk at {num_models} models."
            )
        else:
            st.info(
                f"**{recommendation}**: Architecture A is **${abs(cost_diff):,.0f}/month cheaper** "
                f"at {num_models} models. Consider re-evaluating as you scale."
            )

        # ======================================================================
        # SECTION B: Cost Comparison Breakdown
        # ======================================================================

        st.header("B. Cost Comparison Breakdown")

        col1, col2 = st.columns([2, 1])

        with col1:
            fig_breakdown = plot_cost_breakdown(cost_a, cost_b)
            st.pyplot(fig_breakdown)
            plt.close(fig_breakdown)

        with col2:
            st.subheader("Cost Details")

            breakdown_data = {
                "Component": ["Compute", "Licensing", "Egress", "Ops Overhead", "Fabric Serving", "TOTAL"],
                "Arch A ($)": [
                    f"{cost_a.compute_cost:,.0f}",
                    f"{cost_a.licensing_cost:,.0f}",
                    f"{cost_a.egress_cost:,.0f}",
                    f"{cost_a.ops_overhead_cost:,.0f}",
                    "---",
                    f"{cost_a.total_cost:,.0f}",
                ],
                "Arch B ($)": [
                    f"{cost_b.compute_cost:,.0f}",
                    f"{cost_b.licensing_cost:,.0f}",
                    f"{cost_b.egress_cost:,.0f}",
                    f"{cost_b.ops_overhead_cost:,.0f}",
                    f"{cost_b.fabric_serving_cost:,.0f}",
                    f"{cost_b.total_cost:,.0f}",
                ],
            }

            st.table(pd.DataFrame(breakdown_data))

        # ======================================================================
        # SECTION C: Breakpoint Chart
        # ======================================================================

        st.header("C. Breakpoint Analysis")

        fig_breakpoint = plot_breakpoint(sweep_df, num_models)
        st.pyplot(fig_breakpoint)
        plt.close(fig_breakpoint)

        if breakpoint:
            st.markdown(
                f"**Interpretation**: At approximately **{breakpoint} semantic models**, "
                f"Architecture B becomes more cost-effective. You currently have **{num_models} models**."
            )
            if num_models >= breakpoint:
                st.success("You are past the breakpoint -- consolidating on Fabric is recommended.")
            else:
                st.warning(
                    f"You are {breakpoint - num_models} models away from breakpoint. "
                    f"Plan for Fabric migration as you grow."
                )
        else:
            if cost_b.total_cost < cost_a.total_cost:
                st.success("Architecture B is cheaper across the entire range analyzed (1-100 models).")
            else:
                st.info(
                    "No crossover detected in the 1-100 model range. "
                    "Architecture A remains cheaper, but this may change with different parameters."
                )

        # ======================================================================
        # SECTION D: Phase Portrait (Optional)
        # ======================================================================

        with st.expander("D. Cost Dynamics: Unit Cost Phase Portrait (Advanced)"):
            st.markdown("""
            This visualization shows how **unit cost per model** evolves as model count increases.

            - **Coral arrows/line**: Architecture A -- unit cost tends to drift upward with sprawl
            - **Blue arrows/line**: Architecture B -- unit cost decreases as fixed Fabric cost is amortized
            """)

            fig_quiver = plot_quiver(params, sweep_df)
            st.pyplot(fig_quiver)
            plt.close(fig_quiver)

        # ======================================================================
        # SECTION E: Recommendations
        # ======================================================================

        st.header("E. Recommendations")

        if winner == "B":
            st.markdown("### What to Do Next (Architecture B Wins)")
            st.markdown("""
            1. **Land curated serving tables into Fabric** -- Create a Lakehouse or Warehouse
               with pre-aggregated, certified data marts
            2. **Enforce certified datasets/semantic reuse** -- Publish endorsed semantic models
               that multiple reports can share
            3. **Limit number of independent semantic models** -- Consolidate redundant models
               into shared assets
            4. **Publish governance guardrails** -- Document which datasets are authoritative
               and establish review processes
            5. **Consider Fabric pausing** -- Pause capacity during off-hours to further reduce costs
            """)
        else:
            breakpoint_display = breakpoint if breakpoint else "N/A"
            st.markdown("### When Architecture A is Fine")
            st.markdown(f"""
            At your current scale, the Snowflake-fed approach may be appropriate because:

            - **Small number of models**: Limited duplication overhead
            - **Low refresh frequency**: Snowflake compute costs remain manageable
            - **Low concurrency**: Single-cluster warehouse handles the load
            - **Established Snowflake investment**: Existing pipelines and expertise

            **However**, monitor these warning signs for when to re-evaluate:
            - Model count exceeding ~{breakpoint_display} models
            - Increasing warehouse size or cluster count
            - Rising operational complexity with duplicated logic
            - Viewer count growth (Pro licensing costs scale linearly)
            """)

        # ======================================================================
        # SECTION F: Assumptions
        # ======================================================================

        with st.expander("F. Assumptions & Formulas"):
            st.markdown("""
            ### Pricing Assumptions

            **Snowflake Credits per Hour by Warehouse Size:**
            | Size | Credits/Hour |
            |------|--------------|
            | XS | 1 |
            | S | 2 |
            | M | 4 |
            | L | 8 |
            | XL | 16 |
            | 2XL | 32 |
            | 3XL | 64 |
            | 4XL | 128 |

            > **Note**: Snowflake $/credit varies by contract, edition, and cloud provider.
            > Typical range is $2-$4 per credit. Adjust the slider to match your contract.

            ---

            ### Cost Formulas

            **Architecture A: Snowflake Compute**
            ```
            refresh_hours = num_models x refreshes_per_day x 30 x avg_refresh_runtime
            interactive_hours = interactive_query_hours_per_day x 30
            hours_running = min(730, (refresh_hours + interactive_hours) x idle_drift_factor)
            compute_cost_A = credits_per_hour x hours_running x clusters x $/credit
            ```

            **Architecture B: Fabric Serving**
            ```
            effective_hours = 730 x (1 - paused_percent/100)
            fabric_cost_B = fabric_CUs x $/CU-hour x effective_hours
            snowflake_etl_cost = credits_per_hour x 60 x 1 x $/credit  (fixed ETL window)
            ```

            **Power BI Licensing**
            ```
            creators_cost = creators x (Pro or PPU price)
            viewers_cost_A = viewers x Pro price  (always)
            viewers_cost_B = 0 if Fabric >= F64, else viewers x Pro price
            ```

            **Operational Overhead (Proxy)**
            ```
            ops_cost_A = base_ops + num_models x change_rate x (1 - reuse_factor) x ops_unit_cost
            ops_cost_B = base_ops + shared_pipelines x change_rate x ops_unit_cost x 0.5
            ```

            ---

            ### Key Assumptions

            - **Ops overhead is a proxy**: Actual costs depend on team size, tooling, and process maturity
            - **Fabric F64+ enables free viewers**: This follows Power BI embedded/capacity licensing rules
            - **Snowflake auto-suspend is imperfect**: The "idle drift factor" accounts for warehouses staying warm
            - **Architecture B reduces Snowflake usage to ETL only**: ~2 hours/day fixed window vs. demand-driven
            - **Egress applies to cross-cloud scenarios**: May not apply if Snowflake and Fabric are co-located

            ---

            ### Directional Stability

            The breakpoint is **directionally stable** because:
            1. Architecture A compute scales with `num_models x refreshes x concurrency`
            2. Architecture B compute is largely fixed (one-time landing)
            3. Ops overhead in A scales with model sprawl; in B it's centralized
            4. Licensing in B can be zero for viewers with F64+

            As model count increases, Architecture A's costs grow faster than B's.
            """)

        st.divider()
        st.caption(
            "This is a **steering model** with adjustable assumptions. "
            "Actual costs depend on your specific contracts, usage patterns, and organizational factors. "
            "Use this tool to inform directional decisions, not as a precise forecast."
        )

    # ==========================================================================
    # TAB 2: ARCHITECTURE & GOVERNANCE CASE
    # ==========================================================================

    with tab_governance:

        st.header("Architecture Comparison: Visualizing the Sprawl")

        st.markdown("""
        Use the slider below to see how increasing semantic models affects each architecture.
        Watch how **Architecture A** creates a tangle of connections to Snowflake, while
        **Architecture B** maintains a clean, governed flow through Fabric.
        """)

        # Slider for number of models to visualize
        viz_models = st.slider(
            "Number of Semantic Models",
            min_value=2,
            max_value=30,
            value=6,
            step=1,
            key="viz_slider"
        )

        # Create side-by-side architecture diagrams
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))

        # Colors
        snowflake_color = "#29B5E8"
        fabric_color = "#0078D4"
        model_color_a = "#FF6B6B"
        model_color_b = "#107C10"
        report_color = "#FFB347"
        arrow_color_a = "#CC5555"
        arrow_color_b = "#0066AA"

        # =====================================================================
        # Architecture A: Snowflake-fed Sprawl
        # =====================================================================

        ax1.set_xlim(0, 10)
        ax1.set_ylim(0, 10)
        ax1.set_aspect('equal')
        ax1.axis('off')
        ax1.set_title(f"Architecture A: Snowflake-fed Sprawl\n({viz_models} models)",
                      fontsize=14, fontweight='bold', color=model_color_a)

        # Draw Snowflake (source) - single cylinder at top
        snowflake_x, snowflake_y = 5, 9
        snowflake_circle = plt.Circle((snowflake_x, snowflake_y), 0.8, color=snowflake_color, ec='black', linewidth=2)
        ax1.add_patch(snowflake_circle)
        ax1.text(snowflake_x, snowflake_y, "Snowflake", ha='center', va='center', fontsize=9, fontweight='bold', color='white')

        # Draw semantic models in middle tier - spread across
        model_y = 5.5
        model_positions_a = []
        for i in range(viz_models):
            # Spread models across the width
            x = 1 + (8 * i / max(viz_models - 1, 1)) if viz_models > 1 else 5
            model_positions_a.append((x, model_y))

            # Draw model box
            rect = plt.Rectangle((x - 0.4, model_y - 0.3), 0.8, 0.6,
                                   color=model_color_a, ec='black', linewidth=1, alpha=0.8)
            ax1.add_patch(rect)

            # Draw arrow from Snowflake to model (the sprawl!)
            ax1.annotate('', xy=(x, model_y + 0.3), xytext=(snowflake_x, snowflake_y - 0.8),
                        arrowprops=dict(arrowstyle='->', color=arrow_color_a, lw=1.5, alpha=0.7))

        # Draw reports at bottom
        report_y = 2
        for i in range(viz_models):
            x = model_positions_a[i][0]

            # Draw report
            rect = plt.Rectangle((x - 0.35, report_y - 0.25), 0.7, 0.5,
                                   color=report_color, ec='black', linewidth=1, alpha=0.8)
            ax1.add_patch(rect)

            # Arrow from model to report
            ax1.annotate('', xy=(x, report_y + 0.25), xytext=(x, model_y - 0.3),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=1, alpha=0.5))

        # Labels
        ax1.text(5, 0.5, f"{viz_models} independent queries to Snowflake",
                ha='center', fontsize=11, style='italic', color=model_color_a)

        # =====================================================================
        # Architecture B: Fabric Serving Layer
        # =====================================================================

        ax2.set_xlim(0, 10)
        ax2.set_ylim(0, 10)
        ax2.set_aspect('equal')
        ax2.axis('off')
        ax2.set_title(f"Architecture B: Fabric Serving Layer\n({viz_models} models served)",
                      fontsize=14, fontweight='bold', color=model_color_b)

        # Draw Snowflake (source) - at top left
        sf_x, sf_y = 2, 9
        snowflake_circle2 = plt.Circle((sf_x, sf_y), 0.7, color=snowflake_color, ec='black', linewidth=2)
        ax2.add_patch(snowflake_circle2)
        ax2.text(sf_x, sf_y, "Snowflake", ha='center', va='center', fontsize=8, fontweight='bold', color='white')

        # Draw Fabric Lakehouse - center top
        fabric_x, fabric_y = 6, 9
        fabric_rect = plt.Rectangle((fabric_x - 1, fabric_y - 0.5), 2, 1,
                                      color=fabric_color, ec='black', linewidth=2)
        ax2.add_patch(fabric_rect)
        ax2.text(fabric_x, fabric_y, "Fabric\nLakehouse", ha='center', va='center', fontsize=8, fontweight='bold', color='white')

        # Single ETL arrow from Snowflake to Fabric
        ax2.annotate('', xy=(fabric_x - 1, fabric_y), xytext=(sf_x + 0.7, sf_y),
                    arrowprops=dict(arrowstyle='->', color=arrow_color_b, lw=3))
        ax2.text(4, 9.3, "ETL Once", ha='center', fontsize=9, color=arrow_color_b, fontweight='bold')

        # Draw certified semantic models (just 2-3 regardless of report count)
        num_certified = min(3, max(2, viz_models // 5 + 1))
        certified_y = 6
        certified_positions = []
        for i in range(num_certified):
            x = 3 + (4 * i / max(num_certified - 1, 1)) if num_certified > 1 else 5
            certified_positions.append((x, certified_y))

            # Draw certified model (larger, with checkmark effect)
            rect = plt.Rectangle((x - 0.6, certified_y - 0.4), 1.2, 0.8,
                                   color=model_color_b, ec='black', linewidth=2)
            ax2.add_patch(rect)
            ax2.text(x, certified_y, f"Certified\nModel {i+1}", ha='center', va='center',
                    fontsize=7, fontweight='bold', color='white')

            # Arrow from Fabric to certified model
            ax2.annotate('', xy=(x, certified_y + 0.4), xytext=(fabric_x, fabric_y - 0.5),
                        arrowprops=dict(arrowstyle='->', color=arrow_color_b, lw=2))

        # Draw reports at bottom - distributed across certified models
        report_y = 2.5
        for i in range(viz_models):
            # Assign to a certified model
            cert_idx = i % num_certified
            cert_x = certified_positions[cert_idx][0]

            # Spread reports under their certified model
            reports_per_cert = viz_models // num_certified + (1 if i < viz_models % num_certified else 0)
            local_idx = i // num_certified
            spread = min(2.5, 0.6 * reports_per_cert)
            x = cert_x + (local_idx - reports_per_cert/2) * 0.5

            # Keep within bounds
            x = max(1, min(9, x))

            # Draw report
            rect = plt.Rectangle((x - 0.3, report_y - 0.2), 0.6, 0.4,
                                   color=report_color, ec='black', linewidth=1, alpha=0.8)
            ax2.add_patch(rect)

            # Arrow from certified model to report
            ax2.annotate('', xy=(x, report_y + 0.2), xytext=(cert_x, certified_y - 0.4),
                        arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.5))

        # Labels
        ax2.text(5, 0.5, f"1 ETL process, {num_certified} certified models serve {viz_models} reports",
                ha='center', fontsize=11, style='italic', color=model_color_b)

        plt.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        # Key insight
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            st.error(f"""
            **Architecture A at {viz_models} models:**
            - {viz_models} independent connections to Snowflake
            - {viz_models} separate refresh schedules
            - {viz_models} potential metric definitions
            - Compute scales linearly with models
            """)
        with col2:
            st.success(f"""
            **Architecture B at {viz_models} models:**
            - 1 ETL connection to Snowflake
            - {num_certified} certified models serve all {viz_models} reports
            - Consistent metrics across all reports
            - Compute stays flat as reports grow
            """)


if __name__ == "__main__":
    main()
