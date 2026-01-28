import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import streamlit as st
import re

def extract_meter_id(meter_name):
    """Extract meter ID (EM** or GM**) from meter name"""
    match = re.search(r'(E|G)M\d+', meter_name)
    return match.group(0) if match else meter_name

def deduplicate_meters(df):
    """Remove duplicate meters based on meter ID, keeping the meter with the most data"""
    df['Meter_ID'] = df['Meter'].apply(extract_meter_id)
    
    # Group by Meter_ID and keep the meter with the most data points
    meter_counts = df.groupby('Meter_ID').size()
    meter_to_keep = meter_counts.idxmax()
    
    # For each Meter_ID, keep only the meter name that has the most data
    deduplicated_data = []
    for meter_id in df['Meter_ID'].unique():
        meter_group = df[df['Meter_ID'] == meter_id]
        if len(meter_group) > 1:
            # Keep the meter with the most data points
            meter_counts_in_group = meter_group.groupby('Meter').size()
            best_meter = meter_counts_in_group.idxmax()
            deduplicated_data.append(meter_group[meter_group['Meter'] == best_meter])
        else:
            deduplicated_data.append(meter_group)
    
    result = pd.concat(deduplicated_data, ignore_index=True)
    result = result.drop('Meter_ID', axis=1)
    return result

def preprocess_data(energy_df, hdd_df, cdd_df):
    # Work on a copy and normalise column names (handle BOM, quoted headers, spaces, etc.)
    energy_df = energy_df.copy()
    normalised_cols = []
    for c in energy_df.columns:
        name = str(c)
        # Strip common BOM artefacts from the start of the header
        for bom in ("\ufeff", "ï»¿"):
            if name.startswith(bom):
                name = name[len(bom):]
        # Remove outer quotes and surrounding whitespace
        name = name.strip().strip("'").strip('"')
        normalised_cols.append(name)
    energy_df.columns = normalised_cols

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
    seen_meter_ids = set()
    
    for idx in gas_rows:
        meter_name = energy_df.loc[idx, 'Metered Sector']
        meter_id = extract_meter_id(meter_name)
        
        # Skip if we've already seen this meter ID
        if meter_id in seen_meter_ids:
            continue
            
        seen_meter_ids.add(meter_id)
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
    seen_meter_ids = set()
    
    for idx in elec_rows:
        meter_name = energy_df.loc[idx, 'Metered Sector']
        meter_id = extract_meter_id(meter_name)
        
        # Skip if we've already seen this meter ID
        if meter_id in seen_meter_ids:
            continue
            
        seen_meter_ids.add(meter_id)
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

    # Merge SEU metadata into both dataframes
    try:
        seu_mapping = pd.read_csv('seu_mapping.csv')
        # Clean up meter names in SEU mapping to match energy data
        seu_mapping['Meter'] = seu_mapping['Meter'].str.strip()
        
        # Get list of meters to ignore
        ignored_meters = seu_mapping[seu_mapping['SEU_Category'] == 'Ignore']['Meter'].tolist()
        
        # Filter out ignored meters from gas and electricity data
        gas_df = gas_df[~gas_df['Meter'].isin(ignored_meters)]
        elec_df = elec_df[~elec_df['Meter'].isin(ignored_meters)]
        
        # Filter out meters marked as "Ignore" from SEU mapping
        seu_mapping = seu_mapping[seu_mapping['SEU_Category'] != 'Ignore']
        
        # Merge with gas data
        gas_df = gas_df.merge(seu_mapping, on='Meter', how='left')
        
        # Merge with electricity data
        elec_df = elec_df.merge(seu_mapping, on='Meter', how='left')
        
        # Always ensure SEU_Category exists and is filled
        for df in [gas_df, elec_df]:
            if 'SEU_Category' not in df.columns:
                df['SEU_Category'] = 'Unknown'
            df['SEU_Category'] = df['SEU_Category'].fillna('Unknown')
    except Exception as e:
        print(f"Warning: Could not load SEU mapping: {str(e)}")

    for df in [gas_df, elec_df]:
        df['Year'] = df['Date'].dt.year
        df['IsOperational'] = df['Consumption'] > 0

    return gas_df, elec_df

def evaluate_meter_models(data, train_year=2023, test_year=2025):
    results = []
    data['IsOperational'] = data['Consumption'] > 0
    if 'SEU_Category' not in data.columns:
        data['SEU_Category'] = 'Unknown'
    data['SEU_Category'] = data['SEU_Category'].fillna('Unknown')
    seu_norms = {
        'Boiler Systems (Gas)': 'hdd',
        'Air Handling Units (Gas)': 'hdd',
        'Catering Equipment': 'opday',
        'Lighting Systems': 'opday',
        'Air Conditioning & Refrigeration': 'cdd',
        'Electric Space Heaters': 'hdd',
        'ICT & Server Room Cooling': 'fixed',
        'EV Charging Infrastructure': 'opday',
        'Onsite Solar PV': 'pv',
    }
    for meter in data['Meter'].unique():
        meter_data = data[data['Meter'] == meter].copy()
        if meter_data.empty or 'SEU_Category' not in meter_data.columns:
            seu = 'Unknown'
        else:
            seu = meter_data['SEU_Category'].iloc[0] if pd.notna(meter_data['SEU_Category'].iloc[0]) else 'Unknown'
        norm = seu_norms.get(seu, 'opday')
        train = meter_data[meter_data['Year'] == train_year]
        test = meter_data[meter_data['Year'] == test_year]
        if len(train) < 10 or len(test) < 10:
            continue
        if not test.empty:
            max_test_date = test['Date'].max()
            # Create a cut-off date in the baseline year with the same month and day
            baseline_cutoff = pd.Timestamp(year=train_year, month=max_test_date.month, day=max_test_date.day)
            train = train[train['Date'] <= baseline_cutoff]
        if norm == 'hdd':
            train = train.dropna(subset=['HDD', 'Consumption'])
            test = test.dropna(subset=['HDD', 'Consumption'])
            if len(train) < 10 or len(test) < 1:
                continue
            X_train = train[['HDD', 'IsOperational']]
            y_train = train['Consumption']
            X_test = test[['HDD', 'IsOperational']]
            y_test = test['Consumption']
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = model.score(X_train, y_train)
        elif norm == 'cdd':
            train = train.dropna(subset=['CDD', 'Consumption'])
            test = test.dropna(subset=['CDD', 'Consumption'])
            if len(train) < 10 or len(test) < 1:
                continue
            X_train = train[['CDD', 'IsOperational']]
            y_train = train['Consumption']
            X_test = test[['CDD', 'IsOperational']]
            y_test = test['Consumption']
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = model.score(X_train, y_train)
        elif norm == 'opday':
            train = train.dropna(subset=['Consumption'])
            test = test.dropna(subset=['Consumption'])
            if len(train) < 10 or len(test) < 1:
                continue
            X_train = train[['IsOperational']]
            y_train = train['Consumption']
            X_test = test[['IsOperational']]
            y_test = test['Consumption']
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = model.score(X_train, y_train)
        elif norm == 'fixed':
            r2 = float('nan')
            y_pred = np.zeros(len(test))
            y_test = test['Consumption']
        elif norm == 'pv':
            r2 = float('nan')
            y_pred = np.zeros(len(test))
            y_test = test['Consumption']
        else:
            train = train.dropna(subset=['Consumption'])
            test = test.dropna(subset=['Consumption'])
            if len(train) < 10 or len(test) < 1:
                continue
            X_train = train[['IsOperational']]
            y_train = train['Consumption']
            X_test = test[['IsOperational']]
            y_test = test['Consumption']
            from sklearn.linear_model import LinearRegression
            model = LinearRegression().fit(X_train, y_train)
            y_pred = model.predict(X_test)
            r2 = model.score(X_train, y_train)
        results.append({
            'Meter': meter,
            'SEU_Category': seu,
            'Baseline': train['Consumption'].sum(),
            'Predicted': round(np.sum(y_pred), 1) if norm not in ['fixed', 'pv'] else test['Consumption'].sum(),
            'Actual': test['Consumption'].sum(),
            'Estimated Savings': round((np.sum(y_pred) if norm not in ['fixed', 'pv'] else test['Consumption'].sum()) - test['Consumption'].sum(), 0),
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
