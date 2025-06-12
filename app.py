import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from utils import preprocess_data, evaluate_meter_models, plot_monthly_comparison, add_percent_savings, style_summary_table

def format_delta(val):
    """Format the delta value with appropriate arrow and color."""
    if val > 0:
        return f"↑ {val:+.1f}%", "red"
    elif val < 0:
        return f"↓ {abs(val):.1f}%", "green"
    else:
        return f"{val:.1f}%", "gray"

# Initialize session state for file upload status
if 'files_uploaded' not in st.session_state:
    st.session_state.files_uploaded = False

st.set_page_config(page_title="Energy Baseline Dashboard", layout="wide")

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
            st.info("ℹ️ The comparison above uses climate-normalized predictions to account for partial year data and weather variations.")

            st.markdown("### 📘 Dashboard Guide")
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

        with tab4:
            st.header("🏗️ SEU Group Analysis")
            if 'SEU_Category' in gas_df.columns and 'SEU_Category' in elec_df.columns:
                try:
                    # Create tabs for Gas and Electricity SEU analysis
                    seu_tab1, seu_tab2 = st.tabs(["🔥 Gas SEU Analysis", "⚡ Electricity SEU Analysis"])
                    
                    # Function to create SEU analysis plots and tables
                    def create_seu_analysis(data_df, title):
                        seu_grouped = data_df.copy()
                        seu_grouped = seu_grouped[seu_grouped['Year'].isin([baseline_year, comparison_year])]
                        
                        # Check if we have data for both years
                        years_present = seu_grouped['Year'].unique()
                        if len(years_present) < 2:
                            st.warning(f"⚠️ Data is only available for {', '.join(map(str, years_present))}. Please ensure data exists for both baseline and comparison years.")
                            return
                        
                        # Calculate predictions for each SEU category
                        seu_predictions = []
                        monthly_predictions = []
                        
                        for category in seu_grouped['SEU_Category'].unique():
                            category_data = seu_grouped[seu_grouped['SEU_Category'] == category]
                            train_data = category_data[category_data['Year'] == baseline_year]
                            test_data = category_data[category_data['Year'] == comparison_year]
                            
                            if len(train_data) > 0 and len(test_data) > 0:
                                # Calculate monthly averages for baseline
                                monthly_avg = train_data.groupby(train_data['Date'].dt.month)['Consumption'].mean()
                                
                                # Calculate prediction based on operational days
                                test_data['Month'] = test_data['Date'].dt.month
                                test_data['Predicted'] = test_data['Month'].map(monthly_avg)
                                test_data['Predicted'] = test_data['Predicted'] * (test_data['IsOperational'].sum() / len(test_data))
                                
                                # Store predictions for summary
                                seu_predictions.append(test_data[['SEU_Category', 'Consumption', 'Predicted']])
                                
                                # Store monthly predictions for chart
                                monthly_pred = test_data.groupby('Month').agg({
                                    'Predicted': 'sum'
                                }).reset_index()
                                monthly_pred['SEU_Category'] = category
                                monthly_predictions.append(monthly_pred)
                        
                        if seu_predictions:
                            predictions_df = pd.concat(seu_predictions)
                            monthly_pred_df = pd.concat(monthly_predictions)
                            
                            # Create summary table
                            group_summary = predictions_df.groupby('SEU_Category').agg({
                                'Consumption': 'sum',
                                'Predicted': 'sum'
                            }).round(0)
                            
                            # Add baseline values for reference
                            baseline_values = seu_grouped[seu_grouped['Year'] == baseline_year].groupby('SEU_Category')['Consumption'].sum()
                            group_summary['Baseline'] = baseline_values
                            
                            # Calculate percentage change against prediction
                            group_summary['% Change'] = 100 * (group_summary['Consumption'] - group_summary['Predicted']) / group_summary['Predicted']
                            
                            # Format the dataframe
                            formatted_summary = group_summary.style.format({
                                'Baseline': '{:,.0f}',
                                'Consumption': '{:,.0f}',
                                'Predicted': '{:,.0f}',
                                '% Change': '{:+.1f}%'
                            }).background_gradient(
                                subset=['% Change'],
                                cmap='RdYlGn_r',
                                vmin=-50,
                                vmax=50
                            )
                            
                            st.dataframe(formatted_summary, use_container_width=True)
                            
                            # Create monthly comparison chart
                            st.subheader(f"Monthly Consumption by SEU Category - {title}")
                            
                            # Prepare monthly data
                            monthly_data = seu_grouped.copy()
                            monthly_data['Month'] = monthly_data['Date'].dt.month
                            monthly_summary = monthly_data.groupby(['SEU_Category', 'Year', 'Month'])['Consumption'].sum().reset_index()
                            
                            # Get unique categories
                            categories = monthly_summary['SEU_Category'].unique()
                            n_categories = len(categories)
                            
                            # Calculate number of rows and columns for subplots
                            n_cols = 2
                            n_rows = (n_categories + 1) // 2  # Round up division
                            
                            # Create the figure with subplots
                            fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
                            axes = axes.flatten()
                            
                            # Set up bar positions
                            bar_width = 0.25
                            x = np.arange(12)  # 12 months
                            
                            # Plot each category in its own subplot
                            for i, category in enumerate(categories):
                                ax = axes[i]
                                
                                # Plot baseline bars
                                baseline_data = monthly_summary[
                                    (monthly_summary['SEU_Category'] == category) & 
                                    (monthly_summary['Year'] == baseline_year)
                                ]
                                baseline_values = np.zeros(12)
                                for _, row in baseline_data.iterrows():
                                    baseline_values[int(row['Month'])-1] = row['Consumption']
                                ax.bar(x - bar_width/2, baseline_values, 
                                      width=bar_width, alpha=0.5,
                                      label='Baseline')
                                
                                # Plot actual bars
                                actual_data = monthly_summary[
                                    (monthly_summary['SEU_Category'] == category) & 
                                    (monthly_summary['Year'] == comparison_year)
                                ]
                                actual_values = np.zeros(12)
                                for _, row in actual_data.iterrows():
                                    actual_values[int(row['Month'])-1] = row['Consumption']
                                ax.bar(x + bar_width/2, actual_values, 
                                      width=bar_width, alpha=0.7,
                                      label='Actual')
                                
                                # Plot predictions as dashed lines
                                pred_data = monthly_pred_df[monthly_pred_df['SEU_Category'] == category]
                                pred_values = np.zeros(12)
                                for _, row in pred_data.iterrows():
                                    pred_values[int(row['Month'])-1] = row['Predicted']
                                ax.plot(x, pred_values, 
                                       linestyle='--', marker='x',
                                       label='Predicted')
                                
                                # Customize each subplot
                                ax.set_title(category)
                                ax.set_xticks(x)
                                ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                                                  'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'],
                                                 rotation=45)
                                ax.grid(True, alpha=0.3)
                                ax.legend()
                                
                                # Only add y-label to the leftmost plots
                                if i % n_cols == 0:
                                    ax.set_ylabel('Consumption (kWh)')
                            
                            # Remove any empty subplots
                            for j in range(len(categories), len(axes)):
                                axes[j].set_visible(False)
                            
                            plt.tight_layout()
                            st.pyplot(fig)
                        else:
                            st.warning("⚠️ Unable to calculate predictions. Please ensure sufficient data exists for both years.")
                    
                    # Create Gas SEU Analysis
                    with seu_tab1:
                        create_seu_analysis(gas_df, "Gas")
                    
                    # Create Electricity SEU Analysis
                    with seu_tab2:
                        create_seu_analysis(elec_df, "Electricity")
                        
                except Exception as e:
                    st.error(f"⚠️ Error in SEU analysis: {str(e)}")
            else:
                st.warning("⚠️ SEU metadata not available in dataset. Please ensure the SEU mapping file is properly formatted.")

    except Exception as e:
        st.error(f"⚠️ Error during processing: {e}")

else:
    st.info("👈 Please upload all required files to begin analysis.")
