import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import preprocess_data, evaluate_meter_models, plot_monthly_comparison, add_percent_savings, style_summary_table
import time

print("Starting Energy Baseline Dashboard...")

def format_delta(val):
    """Format the delta value with appropriate arrow and color."""
    if val > 0:
        return f"↑ {val:.1f}%", "red"  # Higher consumption (worse)
    elif val < 0:
        return f"↓ {abs(val):.1f}%", "green"  # Lower consumption (better)
    else:
        return f"{val:.1f}%", "gray"

# Initialize session state for file upload status
if 'files_uploaded' not in st.session_state:
    st.session_state.files_uploaded = False

st.set_page_config(page_title="Energy Baseline Dashboard", layout="wide")

with st.spinner('Loading the Energy Baseline Dashboard...'):
    time.sleep(2)  # Simulate loading time, adjust as needed

st.title("📊 Energy Baseline Dashboard – RDS Site")

# Sidebar organization
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Year Selection Section (at top)
    st.subheader("📅 Year Selection")
    col1, col2 = st.columns(2)
    with col1:
        baseline_year = st.selectbox("Baseline Year", [2023, 2024], key="baseline")
    with col2:
        comparison_year = st.selectbox("Comparison Year", [2024, 2025], key="comparison")
    
    # File Upload Section
    st.subheader("📂 Required Files")
    files = {
        "Energy Data": st.file_uploader("Energy Data", type=["csv"], help="Upload energy consumption data"),
        "HDD Data": st.file_uploader("HDD Data", type=["csv"], help="Upload Heating Degree Days data"),
        "CDD Data": st.file_uploader("CDD Data", type=["csv"], help="Upload Cooling Degree Days data"),
        "SEU Mapping": st.file_uploader("SEU Mapping", type=["csv"], help="Upload SEU mapping data for ISO 50001 compliance")
    }

if all(files.values()):
    try:
        # Load and validate main data files
        energy_df = pd.read_csv(files["Energy Data"], encoding="latin1")
        hdd_df = pd.read_csv(files["HDD Data"])
        cdd_df = pd.read_csv(files["CDD Data"])
        
        # Load and validate SEU mapping file
        try:
            seu_mapping = pd.read_csv(files["SEU Mapping"])
            required_columns = ['Meter', 'SEU_Category']
            missing_columns = [col for col in required_columns if col not in seu_mapping.columns]
            if missing_columns:
                st.error(f"⚠️ SEU mapping file is missing required columns: {', '.join(missing_columns)}")
                st.stop()
            seu_mapping.to_csv('seu_mapping.csv', index=False)
        except Exception as e:
            st.error(f"⚠️ Error loading SEU mapping file: {str(e)}")
            st.stop()

        st.success("✅ Files loaded successfully")

        gas_df, elec_df = preprocess_data(energy_df, hdd_df, cdd_df)

        tab1, tab2, tab3, tab4 = st.tabs(["📋 General Summary", "⚡ Electricity Analysis", "🔥 Gas Analysis", "🏗️ SEU Analysis"])

        with tab1:
            # Get the model predictions for comparison
            gas_summary = evaluate_meter_models(gas_df, train_year=baseline_year, test_year=comparison_year)
            elec_summary = evaluate_meter_models(elec_df, train_year=baseline_year, test_year=comparison_year)

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
                gas_delta, gas_color = format_delta(gas_pct)
                st.markdown(f"""
                <div style='text-align: center;'>
                    <h3>🔥 Gas Consumption Change</h3>
                    <p style='font-size: 24px;'>{total_gas_actual:,.0f} kWh</p>
                    <p style='font-size: 18px; color: gray;'>(vs {total_gas_predicted:,.0f} predicted)</p>
                    <p style='font-size: 20px; color: {gas_color};'>{gas_delta}</p>
                </div>
                """, unsafe_allow_html=True)

            with col2:
                elec_delta, elec_color = format_delta(elec_pct)
                st.markdown(f"""
                <div style='text-align: center;'>
                    <h3>⚡ Electricity Consumption Change</h3>
                    <p style='font-size: 24px;'>{total_elec_actual:,.0f} kWh</p>
                    <p style='font-size: 18px; color: gray;'>(vs {total_elec_predicted:,.0f} predicted)</p>
                    <p style='font-size: 20px; color: {elec_color};'>{elec_delta}</p>
                </div>
                """, unsafe_allow_html=True)

            # Add a note about partial year comparison
            st.info("ℹ️ The comparison above uses climate-normalized predictions to account for partial year data and weather variations.")

            with st.expander("📘 Dashboard Guide", expanded=False):
                st.markdown("""
This dashboard provides a climate-normalized analysis of energy consumption at the RDS site.

**Tabs:**
- **General Summary** (this page): High-level overview of total gas and electricity usage, comparing actual consumption against climate-normalized predictions.
- **Electricity Analysis**: Regression models of electricity consumption vs cooling degree days (CDD) for each meter.
- **Gas Analysis**: Regression models of gas consumption vs heating degree days (HDD) for each meter.
- **SEU Analysis**: Analysis of gas consumption by Significant Energy Use (SEU) categories for ISO 50001 compliance.

**Key Terms in Summary Tables:**
- **Baseline**: Total consumption during the baseline year.
- **Actual**: Observed consumption during the reporting year.
- **Predicted**: Model-predicted consumption for the reporting year based on baseline conditions.
- **Estimated Savings**: Difference between predicted and actual consumption. Positive means energy savings.
- **% Savings**: Estimated savings as a percentage of actual consumption.
- **Baseline/Actual Days**: Number of operational days used in the model comparison.

Use this summary to quickly identify overall performance and then drill into each tab for detailed per-meter breakdowns.
                """)

            with st.expander("🧮 How Model Evaluation Works", expanded=False):
                st.markdown("""
The model evaluation in this dashboard uses a linear regression approach for each meter, tailored to the type of energy use (SEU category):

- **Model Type:** Ordinary Least Squares (OLS) linear regression is fitted for each meter using historical (baseline year) data.
- **Features Used:**
    - For most meters, the model uses a climate variable (HDD for gas, CDD for electricity) and whether the building was operational on each day.
    - For some SEU categories, only operational days or fixed values are used, depending on the nature of the load.
    - If climate data is not available, only the operational status is used.
- **Training:**
    - The model is trained on data from the selected baseline year.
    - The features are used to predict daily consumption.
- **Prediction:**
    - The trained model predicts what consumption would have been in the comparison year, given the actual weather and operational days in that year.
- **Outputs:**
    - **Predicted**: The sum of model-predicted daily consumptions for the comparison year.
    - **Actual**: The sum of observed daily consumptions for the comparison year.
    - **Estimated Savings**: The difference between predicted and actual consumption.

**Dependent Variables (Normalization Factors) by SEU Category:**

| SEU Category                      | Normalization / Model Features         |
|-----------------------------------|----------------------------------------|
| Boiler Systems (Gas)              | HDD, IsOperational                     |
| Air Handling Units (Gas)          | HDD, IsOperational                     |
| Catering Equipment                | IsOperational                          |
| Lighting Systems                  | IsOperational                          |
| Air Conditioning & Refrigeration  | CDD, IsOperational                     |
| Electric Space Heaters            | HDD, IsOperational                     |
| ICT & Server Room Cooling         | Fixed (no regression, baseline only)   |
| EV Charging Infrastructure        | IsOperational                          |
| Onsite Solar PV                   | Fixed (no regression, baseline only)   |
| Other/Unknown                     | IsOperational                          |

- **HDD**: Heating Degree Days (for gas-driven heating loads)
- **CDD**: Cooling Degree Days (for cooling/electric loads)
- **IsOperational**: Whether the building was operational on a given day (binary)
- **Fixed**: No regression; baseline consumption is used as prediction

This approach allows for fair, climate-normalized comparisons between years, accounting for weather, operational differences, and the specific nature of each energy use.
                """)

        with tab2:
            st.header("⚡ Electricity Model Evaluation")
            elec_summary = add_percent_savings(elec_summary)
            
            # Sort by actual consumption
            elec_summary = elec_summary.sort_values('Actual', ascending=False)
            # Add page numbers based on position in sorted list
            elec_summary['Page'] = (np.arange(len(elec_summary)) // 8) + 1
            
            # Display the table with page numbers
            st.dataframe(style_summary_table(elec_summary), use_container_width=True)

            st.subheader("Monthly Actual vs Predicted – Electricity")
            total_pages = (len(elec_summary) + 7) // 8
            
            # Create columns for page navigation
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                page = st.selectbox("Select Page", range(1, total_pages + 1), key="elec_page")
            
            start_idx = (page - 1) * 8
            end_idx = min(start_idx + 8, len(elec_summary))
            paginated_elec_summary = elec_summary.iloc[start_idx:end_idx]
            
            # Show which meters are being displayed
            st.write(f"Showing meters {start_idx + 1} to {end_idx} of {len(elec_summary)}")
            plot_monthly_comparison(elec_df, paginated_elec_summary, climate_col="CDD", train_year=baseline_year, test_year=comparison_year, streamlit_mode=True)

        with tab3:
            st.header("🔥 Gas Model Evaluation")
            gas_summary = add_percent_savings(gas_summary)
            
            # Sort by actual consumption
            gas_summary = gas_summary.sort_values('Actual', ascending=False)
            # Add page numbers based on position in sorted list
            gas_summary['Page'] = (np.arange(len(gas_summary)) // 8) + 1
            
            # Display the table with page numbers
            st.dataframe(style_summary_table(gas_summary), use_container_width=True)

            st.subheader("Monthly Actual vs Predicted – Gas")
            total_pages = (len(gas_summary) + 7) // 8
            
            # Create columns for page navigation
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                page = st.selectbox("Select Page", range(1, total_pages + 1), key="gas_page")
            
            start_idx = (page - 1) * 8
            end_idx = min(start_idx + 8, len(gas_summary))
            paginated_gas_summary = gas_summary.iloc[start_idx:end_idx]
            
            # Show which meters are being displayed
            st.write(f"Showing meters {start_idx + 1} to {end_idx} of {len(gas_summary)}")
            plot_monthly_comparison(gas_df, paginated_gas_summary, climate_col="HDD", train_year=baseline_year, test_year=comparison_year, streamlit_mode=True)

        with tab4:
            st.header("🏗️ SEU Group Analysis")
            # Load SEU mapping for aggregation
            try:
                seu_mapping = pd.read_csv('seu_mapping.csv')
                seu_mapping['Meter'] = seu_mapping['Meter'].str.strip()
            except Exception as e:
                st.error(f"⚠️ Could not load SEU mapping: {str(e)}")
                st.stop()

            # Helper to aggregate summary by SEU
            def aggregate_by_seu(summary_df, seu_mapping):
                # Drop SEU_Category from summary_df if present to avoid merge conflicts
                if 'SEU_Category' in summary_df.columns:
                    summary_df = summary_df.drop(columns=['SEU_Category'])
                merged = summary_df.merge(seu_mapping, on='Meter', how='left')
                merged['SEU_Category'] = merged['SEU_Category'].fillna('Unknown')
                # Group and sum numeric columns
                group_cols = ['Baseline', 'Predicted', 'Actual', 'Estimated Savings', 'Baseline Days', 'Actual Days']
                grouped = merged.groupby('SEU_Category')[group_cols].sum().reset_index()
                # Recalculate % Savings
                grouped['% Savings'] = np.where(
                    grouped['Actual'] != 0,
                    np.round(100 * grouped['Estimated Savings'] / grouped['Actual'], 1),
                    np.nan
                )
                return grouped

            # Tabs for Gas and Electricity SEU analysis
            seu_tab1, seu_tab2 = st.tabs(["🔥 Gas SEU Analysis", "⚡ Electricity SEU Analysis"])

            with seu_tab1:
                st.subheader("Gas SEU Summary")
                try:
                    gas_seu_summary = aggregate_by_seu(gas_summary, seu_mapping)
                    # Remove Baseline Days and Actual Days columns
                    gas_seu_summary = gas_seu_summary.drop(columns=['Baseline Days', 'Actual Days'], errors='ignore')
                    st.dataframe(style_summary_table(gas_seu_summary), use_container_width=True)
                except Exception as e:
                    st.error(f"Error in Gas SEU aggregation: {e}")

                # --- Monthly Plots for Gas SEUs ---
                st.subheader("Monthly Consumption by SEU Category - Gas")
                try:
                    gas_df_seu = gas_df.copy()
                    gas_df_seu['SEU_Category'] = gas_df_seu['SEU_Category'].fillna('Unknown')
                    gas_df_seu = gas_df_seu[gas_df_seu['Year'].isin([baseline_year, comparison_year])]
                    # Order categories by actual consumption (comparison year)
                    actual_by_seu = gas_df_seu[gas_df_seu['Year'] == comparison_year].groupby('SEU_Category')['Consumption'].sum()
                    categories = actual_by_seu.sort_values(ascending=False).index.tolist()
                    n_categories = len(categories)
                    n_cols = 2
                    n_rows = (n_categories + 1) // 2
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
                    axes = axes.flatten()
                    bar_width = 0.25
                    x = np.arange(1, 13)
                    for i, category in enumerate(categories):
                        ax = axes[i]
                        baseline = gas_df_seu[(gas_df_seu['SEU_Category'] == category) & (gas_df_seu['Year'] == baseline_year)]
                        baseline_monthly = baseline.groupby(baseline['Date'].dt.month)['Consumption'].sum().reindex(x, fill_value=0)
                        ax.bar(x - bar_width/2, baseline_monthly.values, width=bar_width, alpha=0.5, label='Baseline')
                        actual = gas_df_seu[(gas_df_seu['SEU_Category'] == category) & (gas_df_seu['Year'] == comparison_year)]
                        actual_monthly = actual.groupby(actual['Date'].dt.month)['Consumption'].sum().reindex(x, fill_value=0)
                        ax.bar(x + bar_width/2, actual_monthly.values, width=bar_width, alpha=0.7, label='Actual')
                        pred_monthly = np.zeros(12)
                        for meter in seu_mapping[seu_mapping['SEU_Category'] == category]['Meter']:
                            meter_data = gas_df[(gas_df['Meter'] == meter) & (gas_df['Year'].isin([baseline_year, comparison_year]))].copy()
                            if meter_data.empty:
                                continue
                            train = meter_data[meter_data['Year'] == baseline_year].dropna(subset=['HDD', 'Consumption'])
                            test = meter_data[meter_data['Year'] == comparison_year].dropna(subset=['HDD', 'Consumption'])
                            if len(train) < 30 or len(test) < 1:
                                continue
                            from sklearn.linear_model import LinearRegression
                            X_train = train[['HDD', 'IsOperational']]
                            y_train = train['Consumption']
                            model = LinearRegression().fit(X_train, y_train)
                            X_test = test[['HDD', 'IsOperational']]
                            test = test.copy()
                            test['Predicted'] = model.predict(X_test)
                            monthly_pred = test.groupby(test['Date'].dt.month)['Predicted'].sum().reindex(x, fill_value=0)
                            pred_monthly += monthly_pred.values
                        ax.plot(x, pred_monthly, linestyle='--', marker='x', label='Predicted')
                        ax.set_title(category)
                        ax.set_xticks(x)
                        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        if i % n_cols == 0:
                            ax.set_ylabel('Consumption (kWh)')
                    for j in range(len(categories), len(axes)):
                        axes[j].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error in Gas SEU plotting: {e}")

            with seu_tab2:
                st.subheader("Electricity SEU Summary")
                try:
                    elec_seu_summary = aggregate_by_seu(elec_summary, seu_mapping)
                    # Remove Baseline Days and Actual Days columns
                    elec_seu_summary = elec_seu_summary.drop(columns=['Baseline Days', 'Actual Days'], errors='ignore')
                    st.dataframe(style_summary_table(elec_seu_summary), use_container_width=True)
                except Exception as e:
                    st.error(f"Error in Electricity SEU aggregation: {e}")

                # --- Monthly Plots for Electricity SEUs ---
                st.subheader("Monthly Consumption by SEU Category - Electricity")
                try:
                    elec_df_seu = elec_df.copy()
                    elec_df_seu['SEU_Category'] = elec_df_seu['SEU_Category'].fillna('Unknown')
                    elec_df_seu = elec_df_seu[elec_df_seu['Year'].isin([baseline_year, comparison_year])]
                    # Order categories by actual consumption (comparison year)
                    actual_by_seu = elec_df_seu[elec_df_seu['Year'] == comparison_year].groupby('SEU_Category')['Consumption'].sum()
                    categories = actual_by_seu.sort_values(ascending=False).index.tolist()
                    n_categories = len(categories)
                    n_cols = 2
                    n_rows = (n_categories + 1) // 2
                    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
                    axes = axes.flatten()
                    bar_width = 0.25
                    x = np.arange(1, 13)
                    for i, category in enumerate(categories):
                        ax = axes[i]
                        baseline = elec_df_seu[(elec_df_seu['SEU_Category'] == category) & (elec_df_seu['Year'] == baseline_year)]
                        baseline_monthly = baseline.groupby(baseline['Date'].dt.month)['Consumption'].sum().reindex(x, fill_value=0)
                        ax.bar(x - bar_width/2, baseline_monthly.values, width=bar_width, alpha=0.5, label='Baseline')
                        actual = elec_df_seu[(elec_df_seu['SEU_Category'] == category) & (elec_df_seu['Year'] == comparison_year)]
                        actual_monthly = actual.groupby(actual['Date'].dt.month)['Consumption'].sum().reindex(x, fill_value=0)
                        ax.bar(x + bar_width/2, actual_monthly.values, width=bar_width, alpha=0.7, label='Actual')
                        pred_monthly = np.zeros(12)
                        for meter in seu_mapping[seu_mapping['SEU_Category'] == category]['Meter']:
                            meter_data = elec_df[(elec_df['Meter'] == meter) & (elec_df['Year'].isin([baseline_year, comparison_year]))].copy()
                            if meter_data.empty:
                                continue
                            train = meter_data[meter_data['Year'] == baseline_year].dropna(subset=['CDD', 'Consumption'])
                            test = meter_data[meter_data['Year'] == comparison_year].dropna(subset=['CDD', 'Consumption'])
                            if len(train) < 30 or len(test) < 1:
                                continue
                            from sklearn.linear_model import LinearRegression
                            X_train = train[['CDD', 'IsOperational']]
                            y_train = train['Consumption']
                            model = LinearRegression().fit(X_train, y_train)
                            X_test = test[['CDD', 'IsOperational']]
                            test = test.copy()
                            test['Predicted'] = model.predict(X_test)
                            monthly_pred = test.groupby(test['Date'].dt.month)['Predicted'].sum().reindex(x, fill_value=0)
                            pred_monthly += monthly_pred.values
                        ax.plot(x, pred_monthly, linestyle='--', marker='x', label='Predicted')
                        ax.set_title(category)
                        ax.set_xticks(x)
                        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], rotation=45)
                        ax.grid(True, alpha=0.3)
                        ax.legend()
                        if i % n_cols == 0:
                            ax.set_ylabel('Consumption (kWh)')
                    for j in range(len(categories), len(axes)):
                        axes[j].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Error in Electricity SEU plotting: {e}")

    except Exception as e:
        st.error(f"⚠️ Error during processing: {e}")

else:
    st.info("👈 Please upload all required files to begin analysis.")
