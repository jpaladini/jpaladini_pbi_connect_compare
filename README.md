# Power BI Cost Breakpoints: Snowflake vs Fabric Serving Layer

A strategic planning tool that demonstrates why **centralized semantic model governance** is the most efficient approach for serving enterprise reporting and analytics at scale.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://blank-app-template.streamlit.app/)

## The Core Insight

As organizations scale their Power BI deployments, a critical architectural decision emerges:

- **Architecture A (Snowflake-fed Sprawl)**: Multiple independent semantic models each query Snowflake directly, causing duplicated compute, spiky concurrency, and compounding operational overhead.

- **Architecture B (Fabric Serving Layer)**: Curated data lands into Microsoft Fabric once, then certified semantic models serve multiple reports, improving reuse and lowering marginal costs.

**The math is clear**: Beyond a modest number of semantic models (typically 10-20), the centralized governance approach becomes significantly more cost-effective.

## Why Semantic Model Governance Matters

Data governance often stops at the warehouse layer, leaving the "last mile" of analytics ungoverned. This is a critical gap because **semantic models are where business logic lives**:

- How metrics are calculated (Revenue, Margin, Churn)
- How dimensions relate to facts
- What filters and hierarchies users can access
- Row-level security and access controls

### The Three Pillars of the Governance Argument

1. **Economic Efficiency**: Every independent model incurs compute, maintenance, testing, and support costs. Centralized models amortize these costs across all consumers.

2. **Metric Consistency**: When the CFO's dashboard shows different revenue than the Sales VP's report, credibility evaporates. Certified semantic models enforce single definitions.

3. **Operational Resilience**: Ungoverned sprawl creates fragility. Centralized governance provides documented ownership, standardized patterns, and clear escalation paths.

## What This Tool Provides

### Cost Analysis Tab
- **Executive Summary**: Side-by-side cost comparison with clear recommendations
- **Cost Breakdown**: Stacked bar chart showing compute, licensing, egress, and operational overhead
- **Breakpoint Analysis**: Interactive chart showing where Architecture B becomes cheaper
- **Scenario Presets**: "Today", "6-month Growth", and "Enterprise Scale" configurations

### Architecture & Governance Case Tab
- **Visual Architecture Diagrams**: Mermaid-rendered comparison of both approaches
- **Business Case Framework**: Three pillars of the governance argument
- **Implementation Roadmap**: Phased approach from pilot to optimization
- **Objection Handling**: Responses to common concerns about centralization
- **Recommended Next Steps**: Actionable guidance based on your scenario

## Key Assumptions

| Component | Architecture A | Architecture B |
|-----------|---------------|----------------|
| Snowflake Compute | Scales with model count | Fixed ETL window |
| Fabric Capacity | Not used | Fixed CU cost |
| Power BI Licensing | Pro for all viewers | Free viewers with F64+ |
| Operational Overhead | Scales with sprawl | Centralized, lower per-model |

See the **Assumptions & Formulas** section in the app for detailed calculations.

## Getting Started

### Prerequisites

- Python 3.8+
- pip

### Installation

```bash
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run streamlit_app.py
```

## Usage

1. **Select a Scenario Preset** or configure parameters manually in the sidebar
2. **Review the Cost Analysis** tab to understand the economic tradeoffs
3. **Explore the Architecture & Governance Case** tab for the strategic argument
4. **Adjust parameters** to model your specific situation
5. **Use the insights** to inform your analytics platform strategy

## Making the Case to Leadership

This tool is designed to support conversations with Analytics leadership about extending centralized governance to semantic models. Key talking points:

1. **Cost reduction is quantifiable**: The breakpoint analysis provides concrete projections based on your parameters.

2. **Governance enables scale**: Without it, every new report creates new maintenance burden. With it, new reports leverage existing investments.

3. **Consistency builds trust**: Conflicting metrics in executive meetings erode confidence in data-driven decisions.

4. **The alternative is reactive cleanup**: Proactive governance is always cheaper than fixing sprawl after it happens.

## Contributing

This is a strategic planning tool with intentionally adjustable assumptions. If your organization's cost structures differ significantly, modify the constants in `streamlit_app.py` to reflect your reality.

## Disclaimer

This tool provides directional guidance based on typical pricing assumptions. Actual costs depend on your specific contracts, usage patterns, and organizational factors. Use it to inform decisions, not as a precise forecast.

---

*Built to demonstrate that centralized semantic model governance isn't just a technical best practice -- it's an economic imperative.*
