# -*- coding: utf-8 -*-
"""10.WRI COFI_Cleaning_IJ.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1utFm44PVralYNz9CHFDoRsn43wEICXr1

Load Libraries
"""

import pandas as pd
import numpy as np
import re
import datetime
# IMPORT LOCAL FUNCTIONS
from COFI_cleaning_functions import extract_plant, extract_capacity, extract_fuel, extract_fuel_proxy, region_clean

"""### Set Variables
These variables can be adjusted upfront or via frontend platform without visiting the algorithm code
"""

# Set source data location
#path = r'G:\My Drive\ZFG Insights\Project Folder\51. WRI\1. COFI NLP Project\4. Cleaned Data Source'

# Load Master Mapping file
mapping = './Mapping Files/COFI Mapping_columns.xlsx'
mapping_source_sheet = 'Source'

# Load the name of source dataset for this code
source_name = 'IJ_Global'

# Load fuel mapping file
fuel_map = './Mapping Files/STI-reference_primary-fuel-mapping.xlsx'

# Cleand file output directory
outname = './Cleaned Data/'

# Load bank mapping file
bank_map = './Mapping Files/chinese_banks.xlsx'



"""### Load source data"""

# Set master path
# import os
# os.chdir(path)

# Load Master mapping file
map_df = pd.read_excel(mapping, sheet_name = mapping_source_sheet)

map_temp = map_df.loc[map_df['Source_Name'] == source_name]
map_temp = map_temp.reset_index(drop = True)

# Load fuel mapping file
fuel_map = pd.read_excel(fuel_map)

# Load bank mapping
bank_map = pd.read_excel(bank_map)

# Load source data
# handle in case there are multiple files per source database

# This code is only customized to Refinitiv Loan


if len(map_temp) > 1:
    appended_data = []
    for i in map_temp.index:
        # load Source Data
        filename = map_temp.loc[i,'File_Name']
        starting_row = map_temp.loc[i,'Starting_row']
        sheet = map_temp.loc[i,'Sheet_name']

        if '.xls' in filename:
            if pd.notna(sheet):
                df = pd.read_excel(filename, sheet_name=sheet, skiprows=starting_row)
            else:
                df = pd.read_excel(filename, skiprows=starting_row)
        elif filename.endswith('.csv'):
            df = pd.read_csv(filename)

        appended_data.append(df)

    df = pd.concat(appended_data)
    df = df.reset_index(drop = True)

else:
    # load Source Data
    filename = map_temp.loc[0,'File_Name']
    starting_row = map_temp.loc[0,'Starting_row']
    sheet = map_temp.loc[0,'Sheet_name']

    if '.xls' in filename:
        if pd.notna(sheet):
            df = pd.read_excel(filename, sheet_name=sheet, skiprows=starting_row)
        else:
            df = pd.read_excel(filename, skiprows=starting_row)
    elif filename.endswith('.csv'):
        df = pd.read_csv(filename)



"""### Clean Source Data
This chunk of code is customized per source database.

Variables need cleaning are listed in the Mapping file

#### Filter
1. filter by year, after 2000
"""

# Convert 'Latest Transaction Event Date' column to datetime format
df['Latest Transaction Event Date'] = pd.to_datetime(df['Latest Transaction Event Date'])

# Split the date into Year, Month, and Day columns
df['Year'] = df['Latest Transaction Event Date'].dt.year
df['Month'] = df['Latest Transaction Event Date'].dt.month
df['Day'] = df['Latest Transaction Event Date'].dt.day

df.shape

# Filter by year
df = df[df['Year'] >= 2000]
df = df.reset_index(drop = True)
df.shape

"""2. Filter by Chinese investor"""

# filtered in investor country "mainland"
df = df[df['Company Country HQ'].str.contains('Mainland', na=False)]
df.shape

"""3. Filter out investment destination China"""

# filter out investment destination China
df = df[~df['Transaction Country/Region'].str.contains('Mainland', na=False)]
df.shape

"""4. Filter out transaction role divestor"""

# filter out transaction role divestor
df = df[~df['Transaction Role'].str.contains('Divestor', na=False)]
df.shape

"""5. Filter out company name ADB and UBS securities"""

# filtered out company name ADB and UBS securities
df = df[~df['Company Name'].str.contains('UBS', na=False)]
df = df[~df['Company Name'].str.contains('ADB', na=False)]

df.shape

df = df.reset_index(drop = True)

"""#### Map columns and Convert Units
1. create new, modified columns based on mapping to final variables, as indicated in the mapping file
2. convert units of currency, unit, and capacity unit during step 1
"""

# Transform variables for future matching
for i in df.index:

    # Create a new column mapping to "power_plant_name"
    #df.loc[i,'power_plant_name'] = region_clean(df.loc[i,'Transaction Name'])
    df.loc[i,'power_plant_name'] = extract_plant(df.loc[i,'Transaction Name'])

    # Create a new column mapping to "primary_fuel" from Deal Synopsis column
    df.loc[i,'primary_fuel'] = extract_fuel_proxy(df.loc[i,'Transaction Sub-sector'])

    # Check if the investment is equity or debt
    if df.loc[i, 'Transaction Equity (USD m)'] is not None or df.loc[i, 'Transaction Equity (USD m)'] != 0:
        # Create a new column mapping to "equity_investor_name"
        df.loc[i,'equity_investor_name'] = df.loc[i,'Company Name']
        df.loc[i,'equity_investor_amount'] = df.loc[i,'Transaction Equity (USD m)']
        df.loc[i,'equity_investment_year'] = df.loc[i,'Year']
        df.loc[i,'parent_company_of_investor'] = df.loc[i,'Parent Company']
    if df.loc[i, 'Transaction Debt (USD m)'] is not None or df.loc[i, 'Transaction Debt (USD m)'] != 0:
        df.loc[i,'debt_investment_year'] = df.loc[i,'Year']
        df.loc[i,'debt_investment_amount'] = df.loc[i,'Transaction Debt (USD m)']

# Rename variables into final database names if the variable doesn't need transformation

# load variable names, which are the variables labeled in mapping file but without a Note.
df_var = pd.read_excel(mapping, sheet_name = source_name)
df_var = df_var[df_var['Note'].isna() & df_var[source_name].notna()]
df_var = df_var.reset_index(drop = True)
variables = df_var[source_name].tolist()

# rename each variable
for j in df_var.index:
    old_name = df_var.loc[j,source_name]
    new_name = df_var.loc[j,'Column_name']
    df[new_name] = df[old_name]

"""### Additional Fix - Tranche type fix by Ludovica

What we know:
* the data is at a <i>tranche</i> level for a company. Multiple tranches make up a transaction.
    * The money that is exchange and which we should consider is the "LT Accredited Value ($m)" value.
* A transaction is identified by its "j_id".
* A company can participate in a transaction either as a debt or equity, this is determined by the "Tranche Instrument Primary Type" value.

So, to bring the data at a <i>transaction</i> level we need to first aggregated at the j_id level, then at the company level, and then at the type level. Then to get the investment amount for each company in this transaction and for this type of transaction we just add the "LT Accredited Value" up.
    * so if a company does both equity and debt investments in a transaction, then in the end we should have two rows, one with its equity info (we add all the tranches that this company does under equity) and one with its debt info (we add all the tranches that this company does under equity)
    
To make the IJ_Global dataset for the merging step we will also to re-create the equity and debt columns:
* Equity: equity_investor_name, equity_investor_amount, equity_investment_year
* Debt: debt_investment_year, debt_investment_amount + one new column for the debt_investor_name
"""

ijg = df.copy()

aggregation_rules = {
    # what we are interested in: adding all the accredited values up
    "LT Accredited Value ($m)" : "sum",
    # these followings do need any particular aggregation so we just take the max
    'Year': "max",
    'power_plant_name': "max",
    'primary_fuel': "max",
    'parent_company_of_investor': "max",
    'country': "max",
    'province': "max",
    'investment_type': "max",
    'region': "max"
}

"""Here we fix the types so to facilitate the aggregation."""

# for all except accredited value and the year they go as strings
for col in aggregation_rules:
    if ijg[col].dtype != "float64" and  ijg[col].dtype != "int64":
        ijg[col] = ijg[col].astype("string")

# fill the nan values for the string with an empty string, to faciliate the grouping
for col in aggregation_rules:
    if ijg[col].dtype == "string":
        ijg[col] = ijg[col].fillna("")
ijg["Tranche Instrument Primary Type"] = ijg["Tranche Instrument Primary Type"].fillna("")

"""Here we aggregate"""

ijg_agg = ijg.groupby(by=["j_id", 'Company Name', "Tranche Instrument Primary Type"]).agg(aggregation_rules).reset_index()

"""We now re-create the equity and debt columns for the merging based on the tranche type column:
* Equity: equity_investor_name, equity_investor_amount, equity_investment_year
* Debt: debt_investment_year, debt_investment_amount + one new column for the debt_investor_name
"""

# create the new columns
for i, row in ijg_agg.iterrows():
    # the debt and equity rows need to be mapped differently
    if row["Tranche Instrument Primary Type"] == "Debt":
        # then add the debt information
        ijg_agg.at[i, "debt_investment_year"] = row['Year']
        ijg_agg.at[i, "debt_investment_amount"] = row['LT Accredited Value ($m)'] # this is the sum
        ijg_agg.at[i, "debt_investor_name"] = row['Company Name']
    elif row["Tranche Instrument Primary Type"] == "Equity":
        # add equity info
        ijg_agg.at[i, "equity_investment_year"] = row['Year']
        ijg_agg.at[i, "equity_investor_amount"] = row['LT Accredited Value ($m)'] # this is the sum
        ijg_agg.at[i, "equity_investor_name"] = row['Company Name']

"""Remove rows where investment_type == 'refinancing' and no Chinese banck is involved as debt investor for the same j_id"""

#keywords for chnese banks
bank_name = bank_map['Old'].unique().tolist() + bank_map['New'].unique().tolist()

rows_to_remove = []
for i in ijg_agg['j_id'][ijg_agg['investment_type'] == 'Refinancing'].unique().tolist():
    temp = ijg_agg[ijg_agg['j_id'] == i]
    temp['debt_investor_name'] = temp['debt_investor_name'].str.lower()
    temp = temp[temp['debt_investor_name'].isin(bank_name)]
    if temp.shape[0] >0:
        continue
    else:
        rows_to_remove.append(i)

ijg_agg = ijg_agg[~(ijg_agg['j_id'].isin(rows_to_remove)) | ~(ijg_agg['investment_type'] == 'Refinancing')]

"""### Save cleaned data"""

ijg_agg.to_excel(outname + source_name + '.xlsx', index = False)

ijg_agg

