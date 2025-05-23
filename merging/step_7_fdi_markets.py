# -*- coding: utf-8 -*-
"""Step 7 - fDI Markets.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/164EJYDPuoVWsAfkbBkUdzax5LdRxqMt-
"""

import pandas as pd
import numpy as np

# import custom-made functions
from functions.cleaning import get_variables, create_placeholder, create_unique_id, get_variables_advanced, preprocess_text
from functions.cleaning import  check_if_new_country_names, update_country_names, convert_primary_fuel
from functions.joining import join_pp_until_country, join_equity_transaction
from functions.final_tables_creation import get_tables_to_create_empty
# TODO: the preprocess_text should go in the cleaning functions file

# other useful libraries
import itertools
import statistics

# for the matching of power plant names
from thefuzz import process
from thefuzz import fuzz

from functions.joining import join_pp_full_equity_full
from functions.cleaning import my_read_excel, my_reset_index
from functions.cleaning import get_thresholds_methods_dicts, convert_equity_investment_type
from functions.matching import find_matches_equity_and_debt_complete
from functions.matching import find_matches_fdi
from functions.cleaning import fill_column_in_rows
from functions.cleaning import print_final_statistics
from functions.cleaning import fix_investor_names, clean_primary_fuel_column
from functions.cleaning import count_sources_in_id

# needed folder paths
input_folder = "Cleaned Data/" # or we can put this info in the "files name" sheet
intermediate_folder = "Intermediate Results/" # to save some intermediate results
current_final_tables_folder = "Current Final Tables/"

# load the excel containing all the mapping information
map_table_name = "Variables.xlsx"
map_table = pd.read_excel(map_table_name, sheet_name=None) # read all the sheets that we will be using togehter
print(len(map_table))

# load the excel containing all the mapping information
dictionary_fuel_name = "Dictionaries.xlsx"
dictionary_fuel = pd.read_excel(dictionary_fuel_name, sheet_name=None) # read all the sheets that we will be using togehter
print(len(dictionary_fuel))


# the sheet where there is the information that the client can change
files_names_sheet = "file names"
groupby_info_sheet = "formatting"
vertical_slicing_info_sheet = "variables to use"
aggregation_info_sheet = "aggregation"
db_structure_sheet = "db structure"
un_country_info_sheet = "UN Country Info"
country_names_dict_sheet = "country names dicionary"
bri_countries_sheet = "BRI countries"
filtering_sheet = "filtering"
merging_sheet = "merging"
matching_threshold_sheet = "matching thersholds"
matching_method_sheet = "matching methods"
matching_paremeter_sheet = "matching parameters"
# we will need this later
thresholds_df = map_table[matching_threshold_sheet]
methods_df = map_table[matching_method_sheet]
parameters_df = map_table[matching_paremeter_sheet]


# the tables we are creating here
tables_to_create = ["Power Plant", "City", "Country", "Equity", "Transaction", "Investor"]
investments_tables = ["Equity", "Transaction", "Investor"]

# already exising final data<<
# debt_file = "Debt.xlsx"
# power_plant_file = "Power Plant.xlsx"
# city_file = "City.xlsx"
# country_file = "Country.xlsx"
# bridge_file = "CITYKEY_BRIDGE_PP_C.xlsx"
# equity_file = "Equity.xlsx"
debt_file = "Debt.csv"
power_plant_file = "Power Plant.csv"
city_file = "City.csv"
country_file = "Country.csv"
bridge_file = "CITYKEY_BRIDGE_PP_C.csv"
equity_file = "Equity.csv"

# names of the databases used here
fdi_name = "FDI_Markets"

# names of the final tables to be used here
equity_name = "Equity"

"""## 1. Read data

#### 1.1 Load data
"""

# this table contains the name of the data sources
file_names = map_table[files_names_sheet]
file_names

fdi = my_read_excel(input_folder , file_names.loc[file_names['Database'] == fdi_name]["File name"].values[0])

fdi_full = fdi.copy(deep=True)

fdi.head()

"""#### 1.2 Load data for exising Power Plants and Debt data for matching purposes"""

# remember that we want to keep the keys so that we can use them if there is a match with exisitng data
columns_to_keep_pp = ["PP_key",  "primary_fuel", "city", "province", "country"]
columns_to_keep_equity = ["equity_id", "PP_key", "equity_investment_type",  "equity_investment_year", "investor_name", "amount"]

pp_full, eq_pp = join_pp_full_equity_full(columns_to_keep_pp, columns_to_keep_equity)

pp_full.head()

count_sources_in_id(pp_full, "PP_key")

count_sources_in_id(eq_pp, "equity_id")

"""#### 1.3 Load Equity table"""

equity = my_read_excel(current_final_tables_folder, equity_file)

equity.tail()

"""## 2. Clean the data

#### 2.1 Vertical slicing
"""

vertical_slicing = map_table[vertical_slicing_info_sheet]
vertical_slicing.head()

fdi_variables = get_variables(vertical_slicing, fdi_name, tables_to_create)
fdi_variables

fdi = fdi[fdi_variables]
fdi.head(2)

# check
print(len(fdi_variables) == len(fdi.columns))

"""#### 2.2 Clean if needed"""

fdi.isna().sum()

"""##### Country"""

# needed to run the functions to do the converting
country_names_dict_df = map_table[country_names_dict_sheet]
un_data = map_table[un_country_info_sheet]

fdi['country'] = fdi['country'].apply(lambda x: x.lower())

# check if there are new names that are needs to be added to the conversion list
# TODO: in the final product this can also be a feature when they load the db
check_if_new_country_names(list(fdi['country'].unique()), un_data, country_names_dict_df)

# convert the names
fdi = update_country_names(country_names_dict_df, fdi)

"""##### Primary_fuel

If the primary_fuel is nan or "other" then we keep/put it to nan because, respectively, it is not known and it may still match something in WEPP that we don't know.
* so when we try to find matches we do an equal join based on the primary_fuel. If the primary_fuel is nan, then we consider all the potential plants (in the same country etc.)
"""

fdi['primary_fuel'].unique()

# make conversion
fdi = clean_primary_fuel_column(fdi)

fdi['primary_fuel'].unique()

"""##### investment amount"""

# # rename now "Capital investment" in fdi since this is the same as the "equity_investor_name" column in r_ma
# # TODO: maybe this won't be needed in the future
# if "Capital investment" in fdi.columns:
#     fdi = fdi.rename(columns={"Capital investment": "total_investment_amount"})

# # TODO: WRI says to ignore the amount for FDI, this should be done in the cleaning step
# fdi['total_investment_amount'] = np.nan

"""##### Equity_investment_type

Add the equity_investment_type: since fdi is only about greenfield investments.
"""

fdi['equity_investment_type'] = "greenfield"

"""##### investor_name"""

fdi['equity_investor_name'] = fdi['equity_investor_name'].fillna("")

fdi = fix_investor_names(fdi, "equity_investor_name", False, None, drop_duplicates=False)

"""#### 2.3 Pre-process REFINTIV_MA and exisitng Power Plant and Equity data for matching"""

pp_full.columns

# check
for col in ['city', "province"]:
    print(f"Rows with unnamed {col}: {pp_full.loc[pp_full[col].str.contains('unnamed')].shape[0]}")

for col in pp_full.columns:
    if col in ["power_plant_name"]:
        pp_full[col] = pp_full[col].fillna("")
        pp_full[col] = pp_full[col].apply(preprocess_text)
        pp_full[col] = pp_full[col].astype("string")
    elif col in ["country", "primary_fuel"]:
        pp_full[col] = pp_full[col].fillna("")
        pp_full[col] = pp_full[col].astype("string")
    elif col in ['installed_capacity']:
        pp_full[col] = pp_full[col].astype("float64")
    elif col in ["city", "province"]:
        pp_full[col] = pp_full[col].fillna("")
        pp_full[col] = pp_full[col].apply(preprocess_text)
        pp_full[col] = pp_full[col].apply(lambda x: "" if "unnamed" in x else x)
        pp_full[col] = pp_full[col].astype("string")

# check
for col in ['city', "province"]:
    print(f"Rows with unnamed {col}: {pp_full.loc[pp_full[col].str.contains('unnamed')].shape[0]}")

eq_pp.columns

# TODO: I shouldn't do this here: when cleaning each dataset then I should also clean the equity_investment_year (so far the only dataset that needs cleaning is BU_CGP)
# I put the code to this conversion in the notebook for BU_CGP
print(f"Before: {eq_pp['equity_investment_type'].unique()}")
eq_pp = convert_equity_investment_type(dictionary_fuel["BU_CGP_equity_investment_type"], eq_pp)
print(f"After: {eq_pp['equity_investment_type'].unique()}")

# check
for col in ['city', "province"]:
    print(f"Rows with unnamed {col}: {eq_pp.loc[eq_pp[col].str.contains('unnamed')].shape[0]}")

for col in eq_pp.columns:
    if col in ["investor_name", "power_plant_name"]:
        eq_pp[col] = eq_pp[col].fillna("")
        eq_pp[col] = eq_pp[col].apply(lambda x: preprocess_text(x, True))
        eq_pp[col] = eq_pp[col].astype("string")
    elif col in ["country", "primary_fuel"]:
        eq_pp[col] = eq_pp[col].fillna("")
        eq_pp[col] = eq_pp[col].astype("string")
    elif (col == 'equity_investment_year') or (col == "amount"):
        eq_pp[col] = eq_pp[col].astype("float64")
    elif col in ["city", "province"]:
        eq_pp[col] = eq_pp[col].fillna("")
        eq_pp[col] = eq_pp[col].apply(lambda x: preprocess_text(x, True))
        eq_pp[col] = eq_pp[col].apply(lambda x: "" if "unnamed" in x else x)
        eq_pp[col] = eq_pp[col].astype("string")

# check
for col in ['city', "province"]:
    print(f"Rows with unnamed {col}: {eq_pp.loc[eq_pp[col].str.contains('unnamed')].shape[0]}")

fdi.columns

for col in fdi.columns:
    if col in ["equity_investment_year", "equity_investor_amount"]:
        fdi[col] = fdi[col].astype("float64")
    elif col in ["city", "power_plant_name", "equity_investor_name", "province", 'parent_company_of_investor']:
        fdi[col] = fdi[col].fillna("")
        fdi[col] = fdi[col].apply(lambda x: preprocess_text(x, True))
        fdi[col] = fdi[col].astype("string")
    elif col in ["country", "primary_fuel"]:
        fdi[col] = fdi[col].fillna("")
        fdi[col] = fdi[col].astype("string")
    elif col in ['equity_investment_type']:
        fdi[col] = fdi[col].astype("string")
        fdi[col] = fdi[col].apply(lambda x: preprocess_text(x, False))

fdi.dtypes

"""## 3. Match with both Equity and Power Plant data to find Power Plant matches

#### 3.1 Load thresholds and methods to use
"""

thrs_dict, mtds_dict, prmts_dict = get_thresholds_methods_dicts(thresholds_df, methods_df, parameters_df, fdi_name, "Special (FDI_M)")

thrs_dict

mtds_dict

prmts_dict

"""#### 3.2 Find the matches with the info in equity"""

fdi = my_reset_index(fdi)

"""Note: we can't really use use_equity_investment_tyep = True because in Equity there are rows that do not have an exact equity_investment_type (they are from IJ_Global)."""

fdi.columns

# find the matches
# no. of matches in default settings 13
fdi_matched, matches = find_matches_fdi(fdi, eq_pp, equity_or_debt="Equity",
                                    thresholds_dict=thrs_dict,
                                    methods_dict=mtds_dict, parameters_dict=prmts_dict)
matches

# check
index = 116
fdi.head(index + 1).tail(1)

eq_pp.loc[eq_pp['PP_key'] == matches[index][1]]

"""I checked all the rows:
* It should be fine as a match
* rows where there is mismatch in the equity investment type (e.g., greenfield vs m&A): 116, 131 + maybe another
"""

# TODO: should we disregard the matches that have a mismatch in teh equity_investment_type?

# for all the rows that didn't have a match, create a PP_key
fdi_matched = my_reset_index(fdi_matched)
fdi_matched = fill_column_in_rows(fdi_matched, "PP_key", "PLANTKEY_FDIMARKETS_")
fdi_matched.head()

# check
matched_rows = fdi_matched.loc[~fdi_matched['PP_key'].str.contains("FDIMARKETS")] # get the non FDI PP_keys
print(f"Same number of rows with non FDI PP_keys as there are matches: {matched_rows.shape[0] == len(matches)}")

# check that the matching went fine
# do some checks
matched = fdi_matched
newly_created_name = "FDIMARKETS"

# check that all the entries now have a proper PP_key

# no. of matched power_plants
key_wepp = matched.loc[matched['PP_key'].str.contains("WEPPGPPD")].shape[0]
key_bucgp = matched.loc[matched['PP_key'].str.contains("BUCGP")].shape[0]
key_bucgef = matched.loc[matched['PP_key'].str.contains("BUCGEF")].shape[0]
key_sais = matched.loc[matched['PP_key'].str.contains("SAIS")].shape[0]
key_ijg = matched.loc[matched['PP_key'].str.contains("IJG")].shape[0]
key_rma = matched.loc[matched['PP_key'].str.contains("REFINITIVMA")].shape[0]
# key_fdi = matched.loc[matched['PP_key'].str.contains("FDIMARKETS")].shape[0]


# no. of newly created keys, these are new entries
key_new = matched.loc[matched['PP_key'].str.contains(newly_created_name)].shape[0]

# check that matched keys got the PP_key from Power Plant
print(f"All matched keys got the PP_key from Power Plant: {key_wepp + key_bucgp + key_bucgef + key_sais + key_ijg + key_rma == matched.loc[~(matched['PP_key'].str.contains(newly_created_name))].shape[0]}")

# check that each row has a PP_key
print(f"Each row has a PP_key: {key_wepp + key_bucgp + key_bucgef + key_sais + key_ijg+ key_rma +  key_new == matched.shape[0]}")

pd.DataFrame(data=[["WEPP + GPPD", key_wepp], ["BUCGP", key_bucgp], ["BUCGEF", key_bucgef], ["SAIS_CLA + IAD_GEGI", key_sais],["IJGLOBAL", key_ijg], ["REFINITIVMA", key_rma], ["FDIMARKETS", key_new]], columns=['Data source', "Count"])

"""## 5. Create tables

#### 5.0 Create the empty tables
"""

db_structure = map_table[db_structure_sheet]
tables_df = get_tables_to_create_empty(db_structure)
# used as support for the creation
tables_df_tmp = get_tables_to_create_empty(db_structure)

"""#### 5.1 Add equity_id"""

# create unique ids with custom function
fdi_matched = create_unique_id(fdi_matched, "equity_id", "EQUITYID_FDIMARKETS_")
fdi_matched.head()

# check that we have all the IDs
for id_col in ['equity_id', "PP_key"]:
    print(f"Missing {id_col}: {fdi_matched.loc[fdi_matched[id_col].isna()].shape[0] + fdi_matched.loc[fdi_matched[id_col] == ''].shape[0]}")

"""#### 5.2 Adding to the Equity table

Since the Equity table (same for Debt) is updated immediately on each data source notebooks, we just concat the equity information from FDI_Markets directly to the (already loaded) equity table.
"""

# get the new entries
entity = equity_name
for column in tables_df_tmp[entity].columns:
    if column in fdi_matched.columns:
        tables_df_tmp[entity][column] = fdi_matched[column]

# concat
tables_df['Equity'] = pd.concat([equity, tables_df_tmp['Equity']])
# check
tables_df['Equity'].shape[0]  == equity.shape[0]  +  tables_df_tmp['Equity'].shape[0]

"""#### 5.3 Creating Transaction and Investor"""

# we need to rename some columns to match the columns in Investor and Transactio
renamings = get_variables_advanced(vertical_slicing, fdi_name, ['Investor', "Transaction"])
# the following two are not in the vertical_slicing (which comes from Variables.xlsx) because we created these two columns not from data coming from the sources (they are based
# on the index of the dataframes....)
renamings["equity_id"] = "investment_id"
renamings['total_investment_amount'] = "amount"
del renamings['Joint venture companies']
renamings

# do renaming
# Note: we need to filter because some renaimings rules only apply to eq_agg_full and some only to dt_agg_full
matched_db_v2 = fdi_matched.rename(columns=dict(filter(lambda x: x[0] in fdi_matched.columns, renamings.items())))
matched_db_v2.head(2)

# needed to facilitate the merging
matched_db_v2['Joint venture companies'] = matched_db_v2['Joint venture companies'].fillna("")
matched_db_v2["Joint venture companies"] = matched_db_v2["Joint venture companies"].astype("string")
matched_db_v2["Joint venture companies"] = matched_db_v2["Joint venture companies"].apply(lambda x: preprocess_text(x, True))

# TODO: RECHECK THIS STEP
new_transaction_rows = []
new_investor_rows = []
for i, row in matched_db_v2.iterrows():
    # get the investors names
    investors_names = [row['investor_name']]
    if row["Joint venture companies"] != "":
        investors_names.extend([x.strip() for x in row["Joint venture companies" ].split(",")])
    investors_names = list(set(investors_names))

    # compute the average of the investment amount
    # The FDI amounts are not to be considered
    # averaged_amount = row["amount"] / len(investors_names)

    # create the entries for Transaction and Investor table
    for investor in investors_names:
        # new_transaction_rows.append([row['investment_id'], investor, averaged_amount, "Y", "N"])
        new_transaction_rows.append([row['investment_id'], investor, np.nan, "N", "N", np.nan])
        if investor == row['investor_name']:
            # we can get their parent company
            new_investor_rows.append([investor, row["parent_company_of_investor"]])
        else:
            new_investor_rows.append([investor, ""])

# put the data that we already have in a dataframe
tables_df['Transaction'] = pd.DataFrame(new_transaction_rows, columns=tables_df_tmp['Transaction'].columns)
tables_df['Investor'] = pd.DataFrame(new_investor_rows, columns=tables_df_tmp['Investor'].columns)

"""#### 5.4 Make Power Plant and City and Country for new Power Plants found here!"""

missing_power_plants = fdi_matched.loc[fdi_matched['PP_key'].str.contains("PLANTKEY_FDIMARKETS_")]
print("Missing power plant: " + str(missing_power_plants.shape[0]))
print(f"check: {fdi_matched.shape[0] - len(matches) == missing_power_plants.shape[0]}")

# get the data from equity
for entity in ['Power Plant', "Country"]:
    for column in tables_df[entity].columns:
        if column in missing_power_plants.columns:
            tables_df[entity][column] = missing_power_plants[column]

entity = "City"
for column in tables_df[entity].columns:
    if column in missing_power_plants.columns:
        tables_df[entity][column] = missing_power_plants[column]
    elif column == "city_key":
        tables_df[entity][column] = missing_power_plants["PP_key"]

"""#### 5.5 Visual check"""

tables_df['Power Plant']

missing_fuel = tables_df['Power Plant'].loc[tables_df['Power Plant']['primary_fuel'] == ""].shape[0]
print(f"Missing primary_fuel: {missing_fuel}")
print(f"% over all rows: {missing_fuel / tables_df['Power Plant'].shape[0] * 100 }")

print(f"There are NO power plants name: {tables_df['Power Plant']['power_plant_name'].isna().sum() == tables_df['Power Plant'].shape[0]}")

tables_df['City']

for col in ['city', "province", "country"]:
    missing = tables_df['City'].loc[tables_df['City'][col] == ""].shape[0]
    print(f"Missing {col}: {missing}")
    print(f"% over all rows: {missing / tables_df['City'].shape[0] * 100 }")
    print()

tables_df['Country']

tables_df['Equity']

# all the FDI rows do have a fdi_id
tables_df['Equity'].loc[~tables_df['Equity']['fdi_id'].isna()].shape[0] == tables_df['Equity'].loc[tables_df['Equity']['equity_id'].str.contains("FDIMARKETS")].shape[0]

tables_df['Transaction']

print(f"All rows do NOT have an amount: {tables_df['Transaction']['amount'].isna().sum() == tables_df['Transaction'].shape[0]}")
print(f"All rows DO have an investment_id: {tables_df['Transaction']['investment_id'].isna().sum() == 0}")

for col in ['investment_id', "investor_name", "investment_averaged", "investment_weighted"]:
    print(f"Mssing {col}: {tables_df['Transaction'].loc[tables_df['Transaction'][col].isna()].shape[0] + tables_df['Transaction'].loc[tables_df['Transaction'][col] == ''].shape[0]}")

tables_df["Investor"]

for col in tables_df["Investor"].columns:
    print(f"Missing {col}: {tables_df['Investor'].loc[tables_df['Investor'][col] == ''].shape[0]}")

print(f'% of valid parent_company_of_investor: {tables_df["Investor"].loc[tables_df["Investor"]["parent_company_of_investor"] != ""].shape[0] / tables_df["Investor"].shape[0] * 100}')

"""#### 5.6 Save"""

print_final_statistics(tables_df, tables_to_create)

# all entities but Debt get saved in the intermediate folder
for entity in ['Power Plant', 'City', 'Country', 'Transaction', 'Investor']:
    tables_df[entity].to_csv(intermediate_folder + entity + "_FDIMARKETS.csv", index=False)

# since Equity is ready and doesn't need any further post-processing, we can save it here
entity = "Equity"
current_final_tables_folder + entity + ".csv"

tables_df['Equity'].to_csv(current_final_tables_folder + entity + ".csv", index=False)

