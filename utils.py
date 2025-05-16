
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import streamlit as st

def preprocess_data(energy_df, hdd_df, cdd_df):
    # Reshape general energy data
    daily_data_rows = energy_df[energy_df['Period'] == 'Day'].copy()
    daily_data_rows['Meter'] = energy_df['Metered Sector'].shift(1)
    date_columns = daily_data_rows.columns[4:-1]
    energy_long = daily_data_rows.melt(
        id_vars='Meter',
        value_vars=date_columns,
        var_name='Date',
        value_name='Consumption'
    )
    energy_long['Date'] = pd.to_datetime(energy_long['Date'], format='%d/%m/%Y', errors='coerce')
    energy_long['Consumption'] = pd.to_numeric(energy_long['Consumption'], errors='coerce')
    energy_long.dropna(subset=['Date', 'Consumption'], inplace=True)

    # HDD and CDD cleaning
    hdd_clean = hdd_df.iloc[4:].copy()
    hdd_clean.columns = ['Date', 'HDD']
    hdd_clean['Date'] = pd.to_datetime(hdd_clean['Date'], errors='coerce')
    hdd_clean['HDD'] = pd.to_numeric(hdd_clean['HDD'], errors='coerce')
    hdd_clean.dropna(inplace=True)

    cdd_clean = cdd_df.iloc[4:].copy()
    cdd_clean.columns = ['Date', 'CDD']
    cdd_clean['Date'] = pd.to_datetime(cdd_clean['Date'], errors='coerce')
    cdd_clean['CDD'] = pd.to_numeric(cdd_clean['CDD'], errors='coerce')
    cdd_clean.dropna(inplace=True)

    # ---------------- GAS DATA ---------------- #
    gas_rows = energy_df[energy_df['Utility'] == 'Gas'].index
    gas_blocks = []
    for idx in gas_rows:
        meter_name = energy_df.loc[idx, 'Metered Sector']
        data_row = energy_df.loc[idx + 1:idx + 1].copy()
        data_row['Meter'] = meter_name
        gas_blocks.append(data_row)

    gas_df = pd.concat(gas_blocks, ignore_index=True)
    gas_df = gas_df.melt(id_vars='Meter',
                         value_vars=gas_df.columns.difference(['Meter','Metered Sector','Utility','Units','Period']),
                         var_name='Date', value_name='Consumption')
    gas_df['Date'] = pd.to_datetime(gas_df['Date'], dayfirst=True, errors='coerce')
    gas_df['Consumption'] = pd.to_numeric(gas_df['Consumption'], errors='coerce')
    gas_df.dropna(subset=['Date', 'Consumption'], inplace=True)
    gas_df.drop_duplicates(subset=['Meter', 'Date'], inplace=True)
    gas_df = gas_df.groupby(['Meter', 'Date'])['Consumption'].sum().reset_index()
    gas_df = gas_df.merge(hdd_clean, on='Date', how='left')

    # ---------------- ELECTRICITY DATA ---------------- #
    elec_rows = energy_df[energy_df['Utility'] == 'Electrical'].index
    elec_blocks = []
    for idx in elec_rows:
        meter_name = energy_df.loc[idx, 'Metered Sector']
        data_row = energy_df.loc[idx + 1:idx + 1].copy()
        data_row['Meter'] = meter_name
        elec_blocks.append(data_row)

    elec_df = pd.concat(elec_blocks, ignore_index=True)
    elec_df = elec_df.melt(id_vars='Meter',
                         value_vars=elec_df.columns.difference(['Meter','Metered Sector','Utility','Units','Period']),
                         var_name='Date', value_name='Consumption')
    elec_df['Date'] = pd.to_datetime(elec_df['Date'], dayfirst=True, errors='coerce')
    elec_df['Consumption'] = pd.to_numeric(elec_df['Consumption'], errors='coerce')
    elec_df.dropna(subset=['Date', 'Consumption'], inplace=True)
    elec_df.drop_duplicates(subset=['Meter', 'Date'], inplace=True)
    elec_df = elec_df.groupby(['Meter', 'Date'])['Consumption'].sum().reset_index()
    elec_df = elec_df.merge(cdd_clean, on='Date', how='left')

    for df in [gas_df, elec_df]:
        df['Year'] = df['Date'].dt.year
        df['IsOperational'] = df['Consumption'] > 0

    return gas_df, elec_df

def evaluate_meter_models(data, climate_col='HDD', train_year=2023, test_year=2025):
    results = []
    data['IsOperational'] = data['Consumption'] > 0

    for meter in data['Meter'].unique():
        meter_data = data[data['Meter'] == meter].copy()
        meter_data = meter_data.dropna(subset=[climate_col, 'Consumption'])

        train = meter_data[meter_data['Year'] == train_year]
        test = meter_data[meter_data['Year'] == test_year]

        if len(train) < 30 or len(test) < 30:
            continue

        train = train.dropna(subset=[climate_col, 'Consumption'])
        test = test.dropna(subset=[climate_col, 'Consumption'])

        X_train = train[[climate_col, 'IsOperational']]
        y_train = train['Consumption']
        model = LinearRegression().fit(X_train, y_train)

        X_test = test[[climate_col, 'IsOperational']]
        y_test = test['Consumption']
        y_pred = model.predict(X_test)

        r2 = model.score(X_train, y_train)

        results.append({
            'Meter': meter,
            'R-squared': round(r2, 3),
            'Baseline': train['Consumption'].sum(),
            'Predicted': round(y_pred.sum(), 1),
            'Actual': test['Consumption'].sum(),
            'Estimated Savings': round(y_pred.sum() - test['Consumption'].sum(), 0),
           # f'{climate_col} Sensitivity': round(model.coef_[0], 2),
           # 'Operational Effect': round(model.coef_[1], 2),
            'Baseline Days': int(train['IsOperational'].sum()),
            'Actual Days': int(test['IsOperational'].sum())
        })

    return pd.DataFrame(results)

def plot_monthly_comparison(data, summary_df, climate_col='HDD', train_year=2023, test_year=2025, streamlit_mode=False):
    full_months = pd.Index(range(1, 13), name='Month')
    summary_df = summary_df.sort_values(by='Actual', ascending=False)
    meters = summary_df['Meter'].tolist()[:8]

    fig, axes = plt.subplots(4, 2, figsize=(14, 16))
    axes = axes.flatten()

    for i, meter in enumerate(meters):
        ax = axes[i]
        df_train = data[(data['Meter'] == meter) & (data['Year'] == train_year)].copy()
        df_test = data[(data['Meter'] == meter) & (data['Year'] == test_year)].copy()

        df_train = df_train.dropna(subset=[climate_col, 'Consumption'])
        df_test = df_test.dropna(subset=[climate_col, 'Consumption'])

        if df_train.empty or df_test.empty:
            ax.set_visible(False)
            continue

        model = LinearRegression().fit(df_train[[climate_col, 'IsOperational']], df_train['Consumption'])
        df_test['Predicted'] = model.predict(df_test[[climate_col, 'IsOperational']])

        monthly_train = df_train.groupby(df_train['Date'].dt.month)['Consumption'].sum().reindex(full_months, fill_value=0)
        monthly_test = df_test.groupby(df_test['Date'].dt.month)['Consumption'].sum().reindex(full_months, fill_value=0)
        monthly_pred = df_test.groupby(df_test['Date'].dt.month)['Predicted'].sum().reindex(full_months, fill_value=0)

        x = full_months
        ax.bar(x - 0.2, monthly_train.values, width=0.2, label=f'{train_year} Actual')
        ax.bar(x, monthly_test.values, width=0.2, label=f'{test_year} Actual')
        ax.plot(x, monthly_pred.values, linestyle='--', marker='o', label='Predicted')

        ax.set_title(f'{meter}', fontsize=10)
        ax.set_xlabel('Month')
        ax.set_ylabel('kWh')
        ax.set_xticks(range(1, 13))
        ax.grid(True)

    for j in range(len(meters), len(axes)):
        axes[j].set_visible(False)

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.suptitle("Monthly Consumption: Actual vs Predicted", fontsize=14)

    if streamlit_mode:
        st.pyplot(fig)
    else:
        plt.show()


# --- Summary Table Enhancements ---

def add_percent_savings(summary_df):
    if 'Actual' in summary_df.columns and 'Estimated Savings' in summary_df.columns:
        summary_df['% Savings'] = round(100 * summary_df['Estimated Savings'] / summary_df['Actual'], 1)
    return summary_df

def style_summary_table(df, title=""):
    styled = (
        df.sort_values(by='Actual', ascending=False)
          .style
          .background_gradient(subset=['% Savings'], cmap='RdYlGn', vmin=-50, vmax=50)
          .format({
              'R-squared': '{:,.2f}',
              'Baseline': '{:,.0f}',
              'Predicted': '{:,.0f}',
              'Actual': '{:,.0f}',
              'Estimated Savings': '{:,.0f}',
              '% Savings': '{:+.1f}%',
              'MAE': '{:.1f}' if 'MAE' in df.columns else '{:.1f}',
              'MSE': '{:.1f}' if 'MSE' in df.columns else '{:.1f}',
              'Baseload (kWh)': '{:.1f}' if 'Baseload (kWh)' in df.columns else '{:.1f}',
              'HDD Sensitivity': '{:.2f}' if 'HDD Sensitivity' in df.columns else '{:.2f}',
              'CDD Sensitivity': '{:.2f}' if 'CDD Sensitivity' in df.columns else '{:.2f}',
              'Operational Effect': '{:,.0f}' if 'Operational Effect' in df.columns else '{:,.0f}',

          }, na_rep='–')
          .set_caption(title)
    )
    return styled
