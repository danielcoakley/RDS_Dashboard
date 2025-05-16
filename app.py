import streamlit as st
import pandas as pd
from utils import preprocess_data, evaluate_meter_models, plot_monthly_comparison, add_percent_savings, style_summary_table

def format_delta(val):
    """Format the delta value with appropriate arrow and color."""
    if val > 0:
        return f"↑ {val:+.1f}%", "red"
    elif val < 0:
        return f"↓ {abs(val):.1f}%", "green"
    else:
        return f"{val:.1f}%", "gray"

st.set_page_config(page_title="Energy Baseline Dashboard", layout="wide")

st.title("📊 Energy Baseline Dashboard – RDS Site")

# Sidebar organization
with st.sidebar:
    st.header("⚙️ Settings")
    
    # File Upload Section
    st.subheader("📂 Required Files")
    files = {
        "Energy Data": st.file_uploader("Energy Data", type=["csv"], help="Upload energy consumption data"),
        "HDD Data": st.file_uploader("HDD Data", type=["csv"], help="Upload Heating Degree Days data"),
        "CDD Data": st.file_uploader("CDD Data", type=["csv"], help="Upload Cooling Degree Days data")
    }
    
    # Year Selection Section
    st.subheader("📅 Year Selection")
    col1, col2 = st.columns(2)
    with col1:
        baseline_year = st.selectbox("Baseline Year", [2023, 2024], key="baseline")
    with col2:
        comparison_year = st.selectbox("Comparison Year", [2024, 2025], key="comparison")

if all(files.values()):
    try:
        energy_df = pd.read_csv(files["Energy Data"], encoding="latin1")
        hdd_df = pd.read_csv(files["HDD Data"])
        cdd_df = pd.read_csv(files["CDD Data"])

        st.success("✅ Files loaded successfully")

        gas_df, elec_df = preprocess_data(energy_df, hdd_df, cdd_df)

        tab1, tab2, tab3 = st.tabs(["📋 General Summary", "⚡ Electricity Analysis", "🔥 Gas Analysis"])

        with tab1:

            # Get the model predictions for comparison
            gas_summary = evaluate_meter_models(gas_df, climate_col="HDD", train_year=baseline_year, test_year=comparison_year)
            elec_summary = evaluate_meter_models(elec_df, climate_col="CDD", train_year=baseline_year, test_year=comparison_year)

            # Calculate totals using predicted values
            total_gas_baseline = gas_summary['Baseline'].sum()
            total_gas_actual = gas_summary['Actual'].sum()
            total_gas_predicted = gas_summary['Predicted'].sum()
            gas_diff = total_gas_actual - total_gas_predicted
            gas_pct = 100 * gas_diff / total_gas_predicted if total_gas_predicted else 0

            total_elec_baseline = elec_summary['Baseline'].sum()
            total_elec_actual = elec_summary['Actual'].sum()
            total_elec_predicted = elec_summary['Predicted'].sum()
            elec_diff = total_elec_actual - total_elec_predicted
            elec_pct = 100 * elec_diff / total_elec_predicted if total_elec_predicted else 0

            st.subheader(f"📊 Total Consumption Summary ({baseline_year} vs {comparison_year})")
            col1, col2 = st.columns(2)

            with col1:
                gas_delta, gas_color = format_delta(-gas_pct)  # Negative because less consumption is better
                st.markdown(f"""
                <div style='text-align: center;'>
                    <h3>🔥 Gas Consumption Change</h3>
                    <p style='font-size: 24px;'>{total_gas_actual:,.0f} kWh</p>
                    <p style='font-size: 18px; color: gray;'>(vs {total_gas_predicted:,.0f} predicted)</p>
                    <p style='font-size: 20px; color: {gas_color};'>{gas_delta}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                elec_delta, elec_color = format_delta(-elec_pct)  # Negative because less consumption is better
                st.markdown(f"""
                <div style='text-align: center;'>
                    <h3>⚡ Electricity Consumption Change</h3>
                    <p style='font-size: 24px;'>{total_elec_actual:,.0f} kWh</p>
                    <p style='font-size: 18px; color: gray;'>(vs {total_elec_predicted:,.0f} predicted)</p>
                    <p style='font-size: 20px; color: {elec_color};'>{elec_delta}</p>
                </div>
                """, unsafe_allow_html=True)

            # Add a note about partial year comparison
            st.info("ℹ️ The comparison above uses climate-normalized predictions to account for partial year data and weather variations. Green indicates energy savings, red indicates increased consumption.")

            st.markdown("### 📘 Dashboard Guide")
            st.markdown("""
This dashboard provides a climate-normalized analysis of energy consumption at the RDS site.

**Tabs:**
- **General Summary** (this page): High-level overview of total gas and electricity usage, comparing actual consumption against climate-normalized predictions.
- **Electricity Analysis**: Regression models of electricity consumption vs cooling degree days (CDD) for each meter.
- **Gas Analysis**: Regression models of gas consumption vs heating degree days (HDD) for each meter.

**Key Terms in Summary Tables:**
- **Baseline**: Total consumption during the baseline year.
- **Actual**: Observed consumption during the reporting year.
- **Predicted**: Model-predicted consumption for the reporting year based on baseline conditions.
- **Estimated Savings**: Difference between predicted and actual consumption. Positive means energy savings.
- **% Savings**: Estimated savings as a percentage of actual consumption.
- **R-squared**: Model fit quality (closer to 1 is better).
- **Sensitivity**: kWh increase per unit HDD or CDD.
- **Operational Effect**: Additional baseline consumption when the building is operational.
- **Baseline/Actual Days**: Number of operational days used in the model comparison.

Use this summary to quickly identify overall performance and then drill into each tab for detailed per-meter breakdowns.
            """)

        with tab2:
            st.header("⚡ Electricity Model Evaluation")
            elec_summary = add_percent_savings(elec_summary)
            st.dataframe(style_summary_table(elec_summary), use_container_width=True)

            st.subheader("Monthly Actual vs Predicted – Electricity")
            plot_monthly_comparison(elec_df, elec_summary, climate_col="CDD", train_year=baseline_year, test_year=comparison_year, streamlit_mode=True)

        with tab3:
            st.header("🔥 Gas Model Evaluation")
            gas_summary = add_percent_savings(gas_summary)
            st.dataframe(style_summary_table(gas_summary), use_container_width=True)

            st.subheader("Monthly Actual vs Predicted – Gas")
            plot_monthly_comparison(gas_df, gas_summary, climate_col="HDD", train_year=baseline_year, test_year=comparison_year, streamlit_mode=True)

    except Exception as e:
        st.error(f"⚠️ Error during processing: {e}")

else:
    st.info("👈 Please upload all required files to begin analysis.")
