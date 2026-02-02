"""
Power BI Consumption Cost Breakpoints: Snowflake vs Fabric Serving Layer

A steering model to help data platform teams understand when consolidating
semantic models on a Fabric serving layer becomes more cost-effective than
letting multiple Power BI models query Snowflake directly.

Key insight: As model count grows, Architecture A (Snowflake-fed sprawl) incurs
duplicated compute, spiky concurrency, and higher operational overhead. Architecture B
(Fabric serving layer) amortizes ETL cost and improves reuse.

Compatible with Snowflake Streamlit-in-Snowflake (SiS) environments.

Run with: streamlit run streamlit_app.py

Author: Data Platform Engineering
"""

import streamlit as st
import altair as alt
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from dataclasses import dataclass


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


# =============================================================================
# COST COMPUTATION FUNCTIONS
# =============================================================================

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


# =============================================================================
# ALTAIR CHART FUNCTIONS
# =============================================================================

def chart_breakpoint(sweep_df: pd.DataFrame, current_models: int, breakpoint_val: Optional[int]) -> alt.LayerChart:
    """Create breakpoint line chart using Altair."""

    # Prepare long-form data for Altair
    chart_data = sweep_df[["num_models", "cost_a", "cost_b"]].melt(
        id_vars="num_models",
        var_name="Architecture",
        value_name="Monthly Cost (USD)",
    )
    chart_data["Architecture"] = chart_data["Architecture"].map({
        "cost_a": "A: Snowflake-fed sprawl",
        "cost_b": "B: Fabric serving layer",
    })

    # Main cost curves
    lines = (
        alt.Chart(chart_data)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("num_models:Q", title="Number of Semantic Models"),
            y=alt.Y(
                "Monthly Cost (USD):Q",
                title="Monthly Cost (USD)",
                axis=alt.Axis(format="$,.0f"),
            ),
            color=alt.Color(
                "Architecture:N",
                scale=alt.Scale(
                    domain=["A: Snowflake-fed sprawl", "B: Fabric serving layer"],
                    range=["#E74C3C", "#2E86C1"],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Architecture:N"),
                alt.Tooltip("num_models:Q", title="Models"),
                alt.Tooltip("Monthly Cost (USD):Q", format="$,.0f"),
            ],
        )
    )

    layers = [lines]

    # Current position vertical rule
    current_df = pd.DataFrame({"x": [current_models], "label": [f"Current: {current_models}"]})
    current_rule = (
        alt.Chart(current_df)
        .mark_rule(color="red", strokeDash=[4, 4], strokeWidth=2)
        .encode(x="x:Q")
    )
    current_text = (
        alt.Chart(current_df)
        .mark_text(align="center", baseline="bottom", dy=-5, fontSize=11, fontWeight="bold", color="red")
        .encode(x="x:Q", y=alt.value(15), text="label:N")
    )
    layers.extend([current_rule, current_text])

    # Breakpoint vertical rule
    if breakpoint_val is not None:
        bp_df = pd.DataFrame({"x": [breakpoint_val], "label": [f"Breakeven: {breakpoint_val}"]})
        bp_rule = (
            alt.Chart(bp_df)
            .mark_rule(color="gray", strokeDash=[6, 3])
            .encode(x="x:Q")
        )
        bp_text = (
            alt.Chart(bp_df)
            .mark_text(align="left", baseline="bottom", dx=5, dy=-5, fontSize=11, color="gray")
            .encode(x="x:Q", y=alt.value(30), text="label:N")
        )
        layers.extend([bp_rule, bp_text])

    chart = (
        alt.layer(*layers)
        .properties(width="container", height=400, title="Cost Breakpoint Analysis: Snowflake vs Fabric")
    )

    return chart


def chart_cost_breakdown(cost_a: CostBreakdown, cost_b: CostBreakdown) -> alt.Chart:
    """Create stacked bar chart comparing cost components using Altair."""

    data = pd.DataFrame({
        "Component": ["Compute", "Licensing", "Egress", "Ops Overhead", "Fabric Serving"] * 2,
        "Architecture": ["A: Snowflake-fed"] * 5 + ["B: Fabric serving"] * 5,
        "Cost": [
            cost_a.compute_cost, cost_a.licensing_cost, cost_a.egress_cost,
            cost_a.ops_overhead_cost, 0,
            cost_b.compute_cost, cost_b.licensing_cost, cost_b.egress_cost,
            cost_b.ops_overhead_cost, cost_b.fabric_serving_cost,
        ],
    })

    # Filter out zero-cost components for cleaner display
    non_zero = data.groupby("Component")["Cost"].sum()
    active_components = non_zero[non_zero > 0].index.tolist()
    data = data[data["Component"].isin(active_components)]

    chart = (
        alt.Chart(data)
        .mark_bar()
        .encode(
            x=alt.X("Architecture:N", title="", axis=alt.Axis(labelFontSize=12)),
            y=alt.Y("Cost:Q", title="Monthly Cost (USD)", axis=alt.Axis(format="$,.0f"), stack=True),
            color=alt.Color(
                "Component:N",
                scale=alt.Scale(scheme="tableau10"),
                legend=alt.Legend(title="Cost Component", orient="right"),
            ),
            order=alt.Order("Component:N"),
            tooltip=[
                alt.Tooltip("Architecture:N"),
                alt.Tooltip("Component:N"),
                alt.Tooltip("Cost:Q", format="$,.0f"),
            ],
        )
        .properties(width="container", height=400, title="Cost Breakdown by Component")
    )

    return chart


def chart_unit_cost(sweep_df: pd.DataFrame, current_models: int) -> alt.LayerChart:
    """Create unit cost dynamics chart using Altair."""

    df = sweep_df.copy()
    df["unit_a"] = df["cost_a"] / df["num_models"].clip(lower=1)
    df["unit_b"] = df["cost_b"] / df["num_models"].clip(lower=1)

    chart_data = df[["num_models", "unit_a", "unit_b"]].melt(
        id_vars="num_models",
        var_name="Architecture",
        value_name="Unit Cost ($/model/month)",
    )
    chart_data["Architecture"] = chart_data["Architecture"].map({
        "unit_a": "A: Snowflake-fed (unit cost)",
        "unit_b": "B: Fabric serving (unit cost)",
    })

    lines = (
        alt.Chart(chart_data)
        .mark_line(strokeWidth=2.5)
        .encode(
            x=alt.X("num_models:Q", title="Number of Semantic Models"),
            y=alt.Y(
                "Unit Cost ($/model/month):Q",
                title="Unit Cost per Model ($/model/month)",
                axis=alt.Axis(format="$,.0f"),
            ),
            color=alt.Color(
                "Architecture:N",
                scale=alt.Scale(
                    domain=["A: Snowflake-fed (unit cost)", "B: Fabric serving (unit cost)"],
                    range=["#E74C3C", "#2E86C1"],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Architecture:N"),
                alt.Tooltip("num_models:Q", title="Models"),
                alt.Tooltip("Unit Cost ($/model/month):Q", format="$,.0f"),
            ],
        )
    )

    # Current position
    current_df = pd.DataFrame({"x": [current_models]})
    current_rule = (
        alt.Chart(current_df)
        .mark_rule(color="gray", strokeDash=[4, 4])
        .encode(x="x:Q")
    )

    chart = (
        alt.layer(lines, current_rule)
        .properties(width="container", height=350, title="Cost Dynamics: Unit Cost vs Scale")
    )

    return chart


# =============================================================================
# ARCHITECTURE VISUALIZATION (HTML-based, SiS-compatible)
# =============================================================================

def render_architecture_html(n_models: int) -> Tuple[str, str]:
    """Generate HTML for side-by-side architecture comparison diagrams."""

    # --- Architecture A: Snowflake-fed sprawl ---
    model_boxes_a = "".join(
        f'<div style="background:#FF6B6B;color:#fff;padding:3px 7px;margin:2px;'
        f'border-radius:4px;display:inline-block;font-size:10px;font-weight:bold;">'
        f'M{i+1}</div>'
        for i in range(n_models)
    )

    report_boxes_a = "".join(
        f'<div style="background:#FFB347;color:#333;padding:2px 6px;margin:1px;'
        f'border-radius:3px;display:inline-block;font-size:9px;">'
        f'R{i+1}</div>'
        for i in range(n_models)
    )

    arrow_count = min(n_models, 20)
    arrows_a = '<span style="color:#CC5555;font-size:14px;letter-spacing:2px;">' + " &darr; " * arrow_count + "</span>"

    html_a = f"""
    <div style="text-align:center;padding:20px;border:2px solid #FF6B6B;border-radius:12px;background:rgba(255,107,107,0.04);min-height:320px;">
        <div style="font-size:14px;font-weight:bold;color:#CC5555;margin-bottom:12px;">
            Architecture A: Snowflake-fed Sprawl
        </div>
        <div style="background:#29B5E8;color:#fff;padding:12px 28px;border-radius:30px;display:inline-block;font-weight:bold;font-size:15px;margin-bottom:6px;">
            Snowflake
        </div>
        <div style="margin:8px 0;">{arrows_a}</div>
        <div style="margin:10px 4px;">{model_boxes_a}</div>
        <div style="color:#999;margin:6px 0;font-size:16px;">&darr; &darr; &darr;</div>
        <div style="margin:10px 4px;">{report_boxes_a}</div>
        <div style="color:#CC5555;font-size:12px;margin-top:14px;font-style:italic;">
            {n_models} independent queries to Snowflake
        </div>
    </div>
    """

    # --- Architecture B: Fabric serving layer ---
    n_certified = min(3, max(2, n_models // 5 + 1))

    cert_boxes = "".join(
        f'<div style="background:#107C10;color:#fff;padding:6px 14px;margin:3px;'
        f'border-radius:6px;display:inline-block;font-size:11px;font-weight:bold;">'
        f'Certified {i+1}</div>'
        for i in range(n_certified)
    )

    report_boxes_b = "".join(
        f'<div style="background:#FFB347;color:#333;padding:2px 6px;margin:1px;'
        f'border-radius:3px;display:inline-block;font-size:9px;">'
        f'R{i+1}</div>'
        for i in range(n_models)
    )

    html_b = f"""
    <div style="text-align:center;padding:20px;border:2px solid #107C10;border-radius:12px;background:rgba(16,124,16,0.04);min-height:320px;">
        <div style="font-size:14px;font-weight:bold;color:#107C10;margin-bottom:12px;">
            Architecture B: Fabric Serving Layer
        </div>
        <div style="display:inline-flex;align-items:center;gap:10px;margin-bottom:6px;">
            <div style="background:#29B5E8;color:#fff;padding:8px 16px;border-radius:20px;font-weight:bold;font-size:12px;">
                Snowflake
            </div>
            <span style="font-size:22px;color:#0066AA;font-weight:bold;">&rarr;</span>
            <div style="background:#0078D4;color:#fff;padding:10px 20px;border-radius:8px;font-weight:bold;font-size:13px;">
                Fabric Lakehouse
            </div>
        </div>
        <div style="color:#0066AA;font-size:10px;margin:4px 0 6px 0;font-weight:bold;">Single ETL</div>
        <div style="color:#107C10;font-size:16px;margin:4px 0;">&darr;</div>
        <div style="margin:10px 4px;">{cert_boxes}</div>
        <div style="color:#999;margin:6px 0;font-size:16px;">&darr; &darr; &darr;</div>
        <div style="margin:10px 4px;">{report_boxes_b}</div>
        <div style="color:#107C10;font-size:12px;margin-top:14px;font-style:italic;">
            1 ETL, {n_certified} certified models serve {n_models} reports
        </div>
    </div>
    """

    return html_a, html_b


# =============================================================================
# SCENARIO PRESETS
# =============================================================================

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


# =============================================================================
# MAIN APPLICATION
# =============================================================================

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

    tab_cost, tab_arch, tab_strategy = st.tabs([
        "Cost Analysis",
        "Architecture Visualization",
        "Governance Strategy"
    ])

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

    # Compute winner
    cost_diff = cost_a.total_cost - cost_b.total_cost
    winner = "B" if cost_diff > 0 else "A"

    # ==========================================================================
    # TAB 1: COST ANALYSIS
    # ==========================================================================

    with tab_cost:

        # ==================================================================
        # SECTION A: Executive Summary
        # ==================================================================

        st.header("A. Executive Summary")

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

        # ==================================================================
        # SECTION B: Cost Comparison Breakdown
        # ==================================================================

        st.header("B. Cost Comparison Breakdown")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.altair_chart(chart_cost_breakdown(cost_a, cost_b), use_container_width=True)

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

        # ==================================================================
        # SECTION C: Breakpoint Chart
        # ==================================================================

        st.header("C. Breakpoint Analysis")

        st.altair_chart(
            chart_breakpoint(sweep_df, num_models, breakpoint),
            use_container_width=True,
        )

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

        # ==================================================================
        # SECTION D: Unit Cost Dynamics (Advanced)
        # ==================================================================

        with st.expander("D. Cost Dynamics: Unit Cost per Model (Advanced)"):
            st.markdown("""
            This visualization shows how **unit cost per model** evolves as model count increases.

            - **Red line**: Architecture A -- unit cost tends to stay high or rise with sprawl
            - **Blue line**: Architecture B -- unit cost decreases as fixed Fabric cost is amortized
            """)

            st.altair_chart(
                chart_unit_cost(sweep_df, num_models),
                use_container_width=True,
            )

        # ==================================================================
        # SECTION E: Recommendations
        # ==================================================================

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

        # ==================================================================
        # SECTION F: Assumptions
        # ==================================================================

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
    # TAB 2: ARCHITECTURE VISUALIZATION
    # ==========================================================================

    with tab_arch:

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

        # Render side-by-side architecture diagrams using HTML
        html_a, html_b = render_architecture_html(viz_models)

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown(html_a, unsafe_allow_html=True)

        with col_b:
            st.markdown(html_b, unsafe_allow_html=True)

        # Key insight
        st.divider()

        n_certified = min(3, max(2, viz_models // 5 + 1))

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
            - {n_certified} certified models serve all {viz_models} reports
            - Consistent metrics across all reports
            - Compute stays flat as reports grow
            """)

    # ==========================================================================
    # TAB 3: GOVERNANCE STRATEGY
    # ==========================================================================

    with tab_strategy:

        st.header("Power BI Governance Strategy")

        st.markdown("""
        This tab outlines our recommended governance strategy for Power BI semantic models,
        integrating with our existing data platform (Databricks + Snowflake) and leveraging
        Microsoft Fabric's deployment pipelines for controlled, auditable releases.
        """)

        st.divider()

        # ==================================================================
        # End-to-End Data Flow
        # ==================================================================

        st.subheader("End-to-End Data Platform Architecture")

        st.markdown("""
        Our data work happens on purpose-built platforms. Power BI is the **presentation layer**,
        not the transformation layer.
        """)

        # Architecture flow using styled HTML
        arch_html = """
        <div style="display:flex;align-items:stretch;gap:8px;flex-wrap:wrap;justify-content:center;margin:20px 0;">
            <div style="background:#FF3621;color:#fff;padding:16px 20px;border-radius:10px;text-align:center;flex:1;min-width:140px;">
                <div style="font-weight:bold;font-size:14px;">Databricks</div>
                <div style="font-size:11px;margin-top:4px;">dbt transforms</div>
                <div style="font-size:10px;font-style:italic;margin-top:2px;">Bronze &rarr; Silver &rarr; Gold</div>
            </div>
            <div style="display:flex;align-items:center;font-size:24px;color:#666;font-weight:bold;">&rarr;</div>
            <div style="background:#29B5E8;color:#fff;padding:16px 20px;border-radius:10px;text-align:center;flex:1;min-width:140px;">
                <div style="font-weight:bold;font-size:14px;">Snowflake</div>
                <div style="font-size:11px;margin-top:4px;">Gold Schema</div>
                <div style="font-size:10px;font-style:italic;margin-top:2px;">Data Exposure Layer</div>
            </div>
            <div style="display:flex;align-items:center;font-size:24px;color:#666;font-weight:bold;">&rarr;</div>
            <div style="background:#0078D4;color:#fff;padding:16px 20px;border-radius:10px;text-align:center;flex:1;min-width:140px;">
                <div style="font-weight:bold;font-size:14px;">Fabric</div>
                <div style="font-size:11px;margin-top:4px;">Lakehouse / Warehouse</div>
                <div style="font-size:10px;font-style:italic;margin-top:2px;">BI Serving Layer</div>
            </div>
            <div style="display:flex;align-items:center;font-size:24px;color:#666;font-weight:bold;">&rarr;</div>
            <div style="background:#F2C811;color:#333;padding:16px 20px;border-radius:10px;text-align:center;flex:1;min-width:140px;">
                <div style="font-weight:bold;font-size:14px;">Power BI</div>
                <div style="font-size:11px;margin-top:4px;">Reports & Dashboards</div>
                <div style="font-size:10px;font-style:italic;margin-top:2px;">Presentation Layer</div>
            </div>
        </div>
        """
        st.markdown(arch_html, unsafe_allow_html=True)

        # What happens where
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.caption("Complex transforms | Business logic | Data quality")
        with col2:
            st.caption("Governed exposure | Access control | Audit logging")
        with col3:
            st.caption("Certified models | DAX measures | BI optimization")
        with col4:
            st.caption("Visualization | Self-service | Distribution")

        st.info(
            "**Key Principle:** Data transformations stay in Databricks/dbt. "
            "Power BI semantic models handle presentation logic (DAX measures, relationships) -- not ETL."
        )

        st.divider()

        # ==================================================================
        # Deployment Pipeline
        # ==================================================================

        st.subheader("Fabric Deployment Pipeline: Dev -> Prod with PR Gates")

        st.markdown("""
        We use Fabric's native deployment pipelines with Git integration to ensure all changes
        are reviewed, tested, and traceable.
        """)

        # Deployment pipeline flow using styled HTML
        pipeline_html = """
        <div style="display:flex;align-items:stretch;gap:6px;flex-wrap:wrap;justify-content:center;margin:20px 0;">
            <div style="background:#9E9E9E;color:#fff;padding:14px 16px;border-radius:10px;text-align:center;flex:1;min-width:120px;">
                <div style="font-weight:bold;font-size:13px;">Private</div>
                <div style="font-size:10px;margin-top:4px;">No Capacity</div>
                <div style="font-size:9px;font-style:italic;margin-top:2px;">Analyst Sandbox</div>
            </div>
            <div style="display:flex;align-items:center;font-size:20px;color:#666;font-weight:bold;">&rarr;</div>
            <div style="background:#FF9800;color:#fff;padding:14px 16px;border-radius:10px;text-align:center;flex:1;min-width:120px;">
                <div style="font-weight:bold;font-size:13px;">DEV</div>
                <div style="font-size:10px;margin-top:4px;">Fabric Capacity</div>
                <div style="font-size:9px;font-style:italic;margin-top:2px;">Git Connected</div>
            </div>
            <div style="display:flex;align-items:center;font-size:20px;color:#666;font-weight:bold;">&rarr;</div>
            <div style="background:#9C27B0;color:#fff;padding:14px 16px;border-radius:10px;text-align:center;flex:1;min-width:120px;border:3px solid #7B1FA2;">
                <div style="font-weight:bold;font-size:13px;">PR Gate</div>
                <div style="font-size:10px;margin-top:4px;">Senior Approval</div>
                <div style="font-size:9px;font-style:italic;margin-top:2px;">Quality Checkpoint</div>
            </div>
            <div style="display:flex;align-items:center;font-size:20px;color:#666;font-weight:bold;">&rarr;</div>
            <div style="background:#4CAF50;color:#fff;padding:14px 16px;border-radius:10px;text-align:center;flex:1;min-width:120px;">
                <div style="font-weight:bold;font-size:13px;">PROD</div>
                <div style="font-size:10px;margin-top:4px;">Fabric Capacity</div>
                <div style="font-size:9px;font-style:italic;margin-top:2px;">Deployment Pipeline</div>
            </div>
        </div>
        """
        st.markdown(pipeline_html, unsafe_allow_html=True)

        # Workflow explanation
        st.markdown("""
        #### How It Works

        | Stage | Workspace | Who | What Happens |
        |-------|-----------|-----|--------------|
        | **Experiment** | Private (no capacity) | Any analyst | Build models, test DAX, create reports. Share requires Pro license. |
        | **Develop** | DEV (Fabric capacity) | Data team | Proven products promoted here. Git-tracked. Integrate into certified models. |
        | **Review** | PR Gate | Senior/Lead | Code review required. Approve or request changes. Quality checkpoint. |
        | **Release** | PROD (Fabric capacity) | Automated | Deployment pipeline pushes to production. Free viewers with F64+. |
        """)

        st.info("""
        **When to go back to Databricks/dbt:**
        - Complex business logic that should be materialized
        - Calculations needed for audit/compliance
        - Performance optimization (pre-aggregate in Gold)
        - Logic reused across multiple semantic models

        *DAX is for presentation logic. dbt is for data logic.*
        """)

        st.divider()

        # ==================================================================
        # Certified vs Self-Service Models
        # ==================================================================

        st.subheader("Certified Models vs. Self-Service: The 80/20 Rule")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Certified Models (80% of use cases)")
            st.success("""
            **Pre-built semantic models on Fabric Lakehouse:**

            - Standardized DAX measures for common KPIs
            - Documented metric definitions
            - Row-level security configured
            - Refresh schedules managed centrally
            - Endorsed and discoverable in the org

            **Who uses them:**
            - Business analysts creating reports
            - Executives viewing dashboards
            - Cross-functional teams needing consistent metrics

            **Governance:** Full CI/CD, PR-gated, senior approval
            """)

        with col2:
            st.markdown("#### Self-Service Models (20% of use cases)")
            st.warning("""
            **Analyst-created models in private workspaces:**

            - Ad-hoc analysis and exploration
            - Department-specific metrics
            - Prototyping new reports
            - One-off requests

            **Who creates them:**
            - Power users with modeling skills
            - Analysts with specific domain needs
            - Teams testing new data sources

            **Governance:** Requires Pro license to share.
            Successful products graduate to certified.
            """)

        st.markdown("""
        ---

        **The graduation path:**

        ```
        Private Workspace (experiment) -> Business Approval -> DEV (integrate) -> PR Review -> PROD (certified)
        ```

        This keeps governance tight on the models that matter (certified), while allowing
        innovation at the edges (self-service). Sprawl is contained because self-service
        models can't reach broad audiences without going through the governance gate.
        """)

        st.divider()

        # ==================================================================
        # DevSecOps with Purview and APIs
        # ==================================================================

        st.subheader("DevSecOps: Automated Controls with Purview & REST APIs")

        st.markdown("""
        Centralizing on Fabric unlocks powerful automated governance capabilities that are
        impossible with sprawled semantic models.
        """)

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Microsoft Purview Integration")
            st.markdown("""
            **Sensitivity Labels:**
            - Auto-classify semantic models based on data content
            - Labels inherit from source (Snowflake -> Fabric -> Power BI)
            - Downstream inheritance to reports and dashboards

            **Data Loss Prevention (DLP):**
            - Policies scan semantic models for sensitive data
            - Block or alert on policy violations
            - Audit trail of all access and changes

            **Unified Governance:**
            - Single pane of glass in Purview Hub
            - Lineage from Databricks through to Power BI
            - Compliance reporting for auditors

            [Learn more: Purview DLP for Fabric](https://learn.microsoft.com/en-us/purview/dlp-powerbi-get-started)
            """)

        with col2:
            st.markdown("#### Power BI REST & Scanner APIs")
            st.markdown("""
            **Metadata Scanning:**
            - Inventory all semantic models, tables, columns, measures
            - Extract DAX expressions for review
            - Identify unused or duplicate models

            **Automated Governance:**
            - Service principal authentication (no user dependency)
            - Scheduled scans for compliance checks
            - Integration with Purview, Collibra, Alation

            **Programmatic Control:**
            - Refresh management via API
            - Workspace provisioning automation
            - Custom alerting on governance violations

            [Learn more: Scanner APIs](https://learn.microsoft.com/en-us/power-bi/enterprise/service-admin-metadata-scanning)
            """)

        st.markdown("""
        ---

        #### Why This Matters for Sprawl Prevention

        | With Sprawl (Many Models) | With Centralization (Few Certified Models) |
        |---------------------------|---------------------------------------------|
        | Must scan/govern hundreds of models | Scan/govern a handful of certified models |
        | Sensitivity labels applied inconsistently | Labels inherit automatically through lineage |
        | DLP policies hard to enforce | DLP applies at the serving layer |
        | API automation complex and brittle | Clean automation against known models |
        | Audit trail fragmented | Complete lineage from source to report |
        """)

        st.divider()

        # ==================================================================
        # Tie back to cost
        # ==================================================================

        st.subheader("Connecting to the Cost Case")

        st.markdown("""
        This governance strategy directly supports the economic argument from the **Cost Analysis** tab:
        """)

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(label="Snowflake Egress", value="Minimized", delta="1 ETL vs N queries")
            st.caption("Single ETL from Snowflake to Fabric, not per-model queries")

        with col2:
            st.metric(label="Viewer Licensing", value="$0/user", delta="F64+ capacity")
            st.caption("Free viewers with Fabric capacity vs Pro licenses for all")

        with col3:
            st.metric(label="Ops Overhead", value="Reduced", delta="Centralized maintenance")
            st.caption("Fewer models = less maintenance, testing, and support")

        st.success("""
        **The governance strategy enables the cost savings:**

        1. **Single ETL path** -> Lower Snowflake compute costs
        2. **Certified models on Fabric** -> Free viewer access
        3. **PR-gated deployments** -> Quality control reduces rework
        4. **Automated compliance** -> Less manual governance effort
        5. **Contained sprawl** -> Predictable, manageable environment
        """)

        st.divider()

        # ==================================================================
        # Resources
        # ==================================================================

        with st.expander("Resources & References"):
            st.markdown("""
            #### Microsoft Documentation

            **Fabric Deployment & CI/CD:**
            - [Deployment Pipelines Overview](https://learn.microsoft.com/en-us/fabric/cicd/deployment-pipelines/intro-to-deployment-pipelines)
            - [Git Integration with Deployment Pipelines](https://learn.microsoft.com/en-us/fabric/real-time-intelligence/git-deployment-pipelines)
            - [CI/CD Workflow Options](https://learn.microsoft.com/en-us/fabric/cicd/manage-deployment)
            - [Best Practices for Lifecycle Management](https://learn.microsoft.com/en-us/fabric/cicd/best-practices-cicd)

            **Governance & Security:**
            - [Sensitivity Labels in Power BI](https://learn.microsoft.com/en-us/fabric/enterprise/powerbi/service-security-sensitivity-label-overview)
            - [DLP for Fabric and Power BI](https://learn.microsoft.com/en-us/purview/dlp-powerbi-get-started)
            - [Metadata Scanning Overview](https://learn.microsoft.com/en-us/fabric/governance/metadata-scanning-overview)
            - [Power BI REST APIs](https://learn.microsoft.com/en-us/rest/api/power-bi/)

            **Semantic Model Best Practices:**
            - [Semantic Models in Power BI Service](https://learn.microsoft.com/en-us/power-bi/connect-data/service-datasets-understand)

            #### Community Resources
            - [Fabric Lifecycle Management Blog](https://blog.fabric.microsoft.com/en-us/blog/microsoft-fabric-lifecycle-management-getting-started-with-git-integration-and-deployment-pipelines/)
            - [Scanner API Enhancements](https://powerbi.microsoft.com/en-my/blog/announcing-scanner-api-admin-rest-apis-enhancements-to-include-dataset-tables-columns-measures-dax-expressions-and-mashup-queries/)
            """)


if __name__ == "__main__":
    main()
