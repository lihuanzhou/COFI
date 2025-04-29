# -*- coding: utf-8 -*-
"""
Step_AIDDATA.py

This script processes the AidData source for inclusion in the COFI database.
It assumes that AidData has been pre-processed to extract structured information from the “Title” column:
  • power_plant_name (project name, as extracted from the title)
  • installed_capacity (with units, e.g. “250 MW” or “1.2 GW”)
  • primary_fuel (extracted and later converted to COFI standards)
  • country, city, province, and (optionally) additional_info.
  
The script cleans these fields, converts capacity values to MW, updates country names,
and then matches each AidData power plant with the existing COFI Power Plant table based on:
  • exact matches on country_iso3c and primary_fuel,
  • a fuzzy filter on installed_capacity (within a specified threshold),
  • and a similarity check on the power plant name (using additional_info).
  
For rows with a good match, the AidData row inherits the matched PP_key.
For rows with no match, a new unique PP_key is generated with prefix "PLANTKEY_AIDDATA_".
The resulting dataset is saved in the Intermediate Results folder for further integration.
  
Author: [Your Name]
Date: [Date]
"""
#@TODO check that you save to correct distenations
#@TODO city and country tables
import pandas as pd
import numpy as np
import string

from thefuzz import fuzz


# Import custom COFI functions
from functions.cleaning import (
    my_read_excel,
    my_reset_index,
    get_variables,
    get_variables_advanced,
    clean_primary_fuel_column,
    update_country_names,
    create_placeholder,
    create_unique_id,
    get_thresholds_methods_dicts,
    convert_primary_fuel,
    print_final_statistics,
    convert_equity_investment_type,
    fill_column_in_rows,
    prepare_for_joining,
    count_sources_in_id
)
from functions.matching import (
    find_matches_power_plant_or_fill_PP_key,
    prepare_pp_full_for_matching_commissioning_year
)
from functions.groupby import (
    do_groupby,
    get_pairs_to_fix,
    seperate_rows,
    do_groupby_new,
    do_clustering
)
from functions.final_tables_creation import get_tables_to_create_empty
from functions.joining import is_close_overlap

# Variables to track table creation progress
tables_to_create = ["Power Plant", "City", "Country", "Debt", "Transaction", "Investor"]
investments_tables = ["Debt", "Transaction", "Investor"]

# ---------------------- Configuration ---------------------------

# needed folder paths
input_folder = "Cleaned Data/" # or we can put this info in the "files name" sheet
intermediate_folder = "Intermediate Results/" # to save some intermediate results
current_final_tables_folder = "Current Final Tables/" # where we store the final tables (e.g. Power Plant)

# load the excel containing all the mapping information
map_table_name = "Variables.xlsx"
map_table = pd.read_excel(map_table_name, sheet_name=None) # read all the sheets that we will be using togehter
print(len(map_table))

# load funding agencies dictionary
funding_agencies_dict_name = "Funding Agencies dictionary.xlsx"
funding_agencies_df = pd.read_excel(funding_agencies_dict_name)
print(len(funding_agencies_df))

# load the excel containing all the mapping information
dictionary_file_name = "Dictionaries.xlsx"
map_dictionaries = pd.read_excel(dictionary_file_name, sheet_name=None) # read all the sheets that we will be using togehter
print(len(map_dictionaries))


# the sheet where there is the information that the client can change
files_names_sheet = "file names"
groupby_info_sheet = "formatting"
vertical_slicing_info_sheet = "variables to use"
aggregation_info_sheet = "aggregation"
db_structure_sheet = "db structure"
un_country_info_sheet = "UN Country Info"
country_names_dict_sheet = "country names dicionary"
bri_countries_sheet = "BRI countries"
joining_info_sheet = "joining"
commissioning_year_thresholds_info_sheet = "commissioning_year_thresholds"

# the tables we are creating here
tables_to_create = ["Power Plant", "City", "Country", "Equity", "Debt", "Transaction", "Investor"]
investments_tables = ["Equity", "Debt", "Transaction", "Investor"]

# already exising final data
# power_plant_file = "Power Plant.xlsx"
power_plant_file = "Power Plant.csv"
country_file = "Country.csv"
city_file="City.csv"
city_key_bridge_file="CITYKEY_BRIDGE_PP_C.csv"
debt_file = "Debt.csv"
aid_data_name = 'AidData'

# ---------------------- Load Data---------------------------
file_names = map_table[files_names_sheet]

aid_data = my_read_excel(input_folder, file_names.loc[file_names['Database'] == aid_data_name]["File name"].values[0])
#aid_data.dropna(inplace=True)
# safe copy
aid_data.dropna(how="all", inplace=True)
aid_data_full = aid_data.copy(deep=True)

# Load existing power plant data for matching
pp = my_read_excel(current_final_tables_folder, power_plant_file)
pp_copy = pp.copy(deep=True)

country_df = my_read_excel(current_final_tables_folder, country_file)
city_df = my_read_excel(current_final_tables_folder, city_file)
city_key_bridge_df = my_read_excel(current_final_tables_folder, city_key_bridge_file)
debt_df = my_read_excel(current_final_tables_folder, debt_file)

# ---------------------- Clean Data ---------------------------

# Get variables to keep based on vertical slicing configuration
#@TODO check if we need vertical slicing (as in choosing the columns to work on)
# vertical_slicing = map_table[vertical_slicing_info_sheet]
# aid_data_variables = get_variables(vertical_slicing, aid_data_name, tables_to_create)
# aid_data = aid_data[aid_data_variables].copy(deep=True)
pp_columns_to_keep = pp_copy.columns
# pp_city_bridge = pp_copy.merge(
#     city_key_bridge_df,
#     on="PP_key",     # because CITYKEY_BRIDGE_PP_C has PP_key
#     how="left"       # use left join so we don't lose Power Plants with no city
# )

# pp_city = pp_city_bridge.merge(
#     city_df,
#     on="city_key",   # the column that links to City.city_key
#     how="left"
# )

# pp_city_country = pp_city.merge(
#     country_df,
#     on="country",    # or whatever column is used in City + Country
#     how="left"
# )

# pp_city_country = pp_city.merge(
#     country_df,
#     left_on="country",      # city_df’s column
#     right_on="country",# country_df’s column
#     how="left"
# )

# Step 1: Merge with city bridge
pp_city_bridge = pp_copy.merge(
    city_key_bridge_df,
    on="PP_key",
    how="left"
)

# Step 2: Merge with city info
pp_city = pp_city_bridge.merge(
    city_df,
    on="city_key",
    how="left",
    suffixes=('', '_city')
)

# Step 3: Merge with country info (only once)
pp_city_country = pp_city.merge(
    country_df,
    left_on="country",
    right_on="country",
    how="left",
    suffixes=('', '_country')
)

# Step 4: Merge with debt info
pp_city_country_debt = pp_city_country.merge(
    debt_df,
    on="PP_key",
    how="left",
    suffixes=('', '_debt')
)
print("NULL DEBT INVESTENT YEAR IN DEBT.csv_-___________" , debt_df["debt_investment_year"].isna().sum())
print("NULL in pp_city_country_debt, " , pp_city_country_debt["debt_investment_year"].isna().sum())
print(pp_city_country_debt.shape)
print(debt_df.shape)
print("number of pp_key in pp_city_country_debt", pp_city_country_debt["PP_key"].nunique())
print("number of pp_key in debt_df", debt_df["PP_key"].nunique())
print("number of pp_key inside pp_city_country_debt and in debt_df as well",pp_city_country_debt[pp_city_country_debt["PP_key"].isin(debt_df["PP_key"])].shape)
db_structure = map_table[db_structure_sheet]
tables_df = get_tables_to_create_empty(db_structure)

# create support databases
tables_df_tmp = get_tables_to_create_empty(db_structure)

#pp_city_country.set_index("PP_key", inplace=True)

# full_cofi= pd.read_excel("cofi_v2.2_final.xlsx")
# full_cofi["power_plant_name"] = full_cofi["power_plant_name"].fillna("")
# full_cofi["city"] = full_cofi["city"].fillna("")
# full_cofi["province"] = full_cofi["province"].fillna("")

pp_city_country['power_plant_name'] = pp_city_country['power_plant_name'].fillna("")
pp_city_country['city'] = pp_city_country['city'].fillna("")
pp_city_country['province'] = pp_city_country['province'].fillna("")


# Clean and standardize country names
country_names_dict_df = map_table[country_names_dict_sheet]
un_data = map_table[un_country_info_sheet]


# Clean primary fuel data
aid_data = clean_primary_fuel_column(aid_data)


# Clean installed capacity (convert to MW if needed)
def clean_capacity(capacity_str):
    if pd.isna(capacity_str):
        return np.nan
    capacity_str = str(capacity_str).upper()
    value = float(''.join(filter(lambda x: x.isdigit() or x == '.', capacity_str)))
    if 'GW' in capacity_str:
        value *= 1000
    return value

aid_data['installed_capacity'] = aid_data['installed_capacity'].apply(clean_capacity)

aid_data["Recipient ISO-3"] = aid_data["Recipient ISO-3"].fillna("")
aid_data["primary_fuel"] = aid_data["primary_fuel"].fillna("")
aid_data["additional_info"] = aid_data["additional_info"].fillna("")
aid_data['city'] = aid_data['city'].fillna("")
aid_data['province'] = aid_data['province'].fillna("")


# ---------------------- Generate Power Plant Table---------------------------
#@TODO get threshold from variables sheet
def preprocess_text(text, remove_punctuation=False):
    """Preprocess text by converting to lowercase and optionally removing punctuation."""
    if pd.isna(text):
        return ""
    new_string = str(text).lower()
    if remove_punctuation:
        translator = str.maketrans('', '', string.punctuation)
        new_string = new_string.translate(translator)
    return new_string

def compute_similarity(text_1, text_2, similarity_func=fuzz.partial_token_set_ratio):
    """Compute similarity between two text strings."""
    if pd.isna(text_1) or pd.isna(text_2):
        return 0
    return similarity_func(str(text_1), str(text_2))


def match_power_plants(aid_data_row, pp_df, threshold_installed_capacity=300):
    """Match a single AidData power plant with existing power plants."""
    potential_matches = pp_df.copy()
    
    # Filter by country
    if 'Recipient ISO-3' in aid_data_row and 'country_iso3c' in potential_matches.columns:
        potential_matches = potential_matches[
            potential_matches['country_iso3c'] == aid_data_row['Recipient ISO-3']
        ]
    
    # Filter by primary fuel
    if 'primary_fuel' in aid_data_row and 'primary_fuel' in potential_matches.columns:
        potential_matches = potential_matches[
            potential_matches['primary_fuel'] == aid_data_row['primary_fuel']
        ]
    
    # Filter by installed capacity
    if not pd.isna(aid_data_row.get('installed_capacity')):
        capacity = float(aid_data_row['installed_capacity'])
        potential_matches = potential_matches[
            (potential_matches['installed_capacity'] >= capacity - threshold_installed_capacity) &
            (potential_matches['installed_capacity'] <= capacity + threshold_installed_capacity)
        ]
    
    best_match = None
    best_score = 0
    
    for _, target_row in potential_matches.iterrows():
        scores = []
        
        # Name similarity
        if 'additional_info' in aid_data_row and 'power_plant_name' in target_row:
            name_score = compute_similarity(
                preprocess_text(aid_data_row['additional_info'], True),
                preprocess_text(target_row['power_plant_name'], True)
            )
            scores.append(name_score * 0.333333)  # Weight for name similarity
        
        # Location similarity
        if 'city' in aid_data_row and 'city' in target_row:
            city_score = compute_similarity(
                preprocess_text(aid_data_row['city'], True),
                preprocess_text(target_row['city'], True)
            )
            scores.append(city_score * 0.33333)  # Weight for city similarity
        
        if 'province' in aid_data_row and 'province' in target_row:
            province_score = compute_similarity(
                preprocess_text(aid_data_row['province'], True),
                preprocess_text(target_row['province'], True)
            )
            scores.append(province_score * 0.333333)  # Weight for province similarity
        
        if scores:
            total_score = sum(scores)
            if total_score > best_score:
                best_score = total_score
                best_match = target_row
    
    return best_match, best_score


# Lists to store matched and unmatched entries
matched_entries = []
unmatched_entries = []
unmatched_indecies = []
cnt=0


# NOTE power_plant_name => title commisioning_year => Commitment Year
# Column name mapping between AidData and Power Plant table
column_mapping = {
    'additional_info': 'power_plant_name',
    'Commitment Year': 'commissioning_year'
}

# First pass: Process each AidData power plant for matching
matched_entries = []
unmatched_entries = []
for idx, row in aid_data.iterrows():
    best_match, match_score = match_power_plants(row, pp_city_country )

    if best_match is not None:
        cnt += 1
    
    # Prepare power plant entry
    plant_entry = {}
    
    # Copy all columns that exist in Power Plant table structure with correct mapping
    for pp_column in tables_df['Power Plant'].columns:
        # Check if there's a mapping for this column
        aid_column = next((aid_col for aid_col, pp_col in column_mapping.items() 
                          if pp_col == pp_column), pp_column)
        
        if aid_column in row:
            plant_entry[pp_column] = row[aid_column]
    
    if best_match is not None and match_score >= 70:  # Threshold for good match
        # For matched entries, use the existing PP_key
        plant_entry["PP_key"] = best_match["PP_key"]
        plant_entry['match_quality'] = match_score
        
        # Update any missing or better information from the match
        for column in tables_df['Power Plant'].columns:
            if column in best_match and (column not in plant_entry or pd.isna(plant_entry[column])):
                plant_entry[column] = best_match[column]
        matched_entries.append(plant_entry)
    else:
        # For unmatched entries, we'll generate PP_key later
        plant_entry['match_quality'] = 0
        unmatched_entries.append(plant_entry)
        unmatched_indecies.append(idx)

print(f"Found {cnt} potential matches")

# Create DataFrames
matched_df = pd.DataFrame(matched_entries)


# Generate unique IDs for unmatched entries
unmatched_df = pd.DataFrame(unmatched_entries)
unmatched_df = create_unique_id(unmatched_df, 'PP_key', 'PLANTKEY_AIDDATA_')


# Verify PP_keys before concatenation
print(f"Matched PP_keys null: {matched_df['PP_key'].isnull().sum()}")
print(f"Unmatched PP_keys null: {unmatched_df['PP_key'].isnull().sum()}")

# Combine all entries into Power Plant table
tables_df['Power Plant'] = pd.concat([
    tables_df["Power Plant"],
    matched_df,
    unmatched_df
], ignore_index=True)



# Print statistics
print(f"Total power plants processed: {len(tables_df['Power Plant'])}")
print(f"Matched plants: {len(tables_df['Power Plant'][tables_df['Power Plant']['match_quality'] > 0])}")
print(f"New plants: {len(tables_df['Power Plant'][tables_df['Power Plant']['match_quality'] == 0])}")

tables_df["Power Plant"].dropna(how="all", inplace=True)

tables_df["Power Plant"].to_csv("Intermediate Results/Power Plant_AIDDATA.csv", encoding='utf-8',index=False)


# ---------------------- Generate Investment Tables---------------------------
# Clean funding agencies
aid_data = aid_data.copy(deep=True)
# get unmatched subset
aid_data["Funding Agencies"] = aid_data["Funding Agencies"].apply(lambda x: x.lower())

# map funding agencies 
company_dict = {}
for i, row in funding_agencies_df.iterrows():
    company_dict[row['Old']] = row['New']
# Process funding agencies to create investor information
def process_funding_agencies(row):
    agencies = []
    if pd.notna(row['Funding Agencies']):
        agencies.extend(str(row['Funding Agencies']).split('|'))
    if pd.notna(row['Co-financing Agencies']):
        agencies.extend(str(row['Co-financing Agencies']).split('|'))
    return list(set(filter(None, [a.strip() for a in agencies])))

# Create investor table
aid_data['investor_list'] = aid_data.apply(process_funding_agencies, axis=1)



unique_investors = set()
for investors in aid_data['investor_list']:
    unique_investors.update(investors)

for investor in unique_investors:
    
    # Map investor name using dictionary if available
    mapped_name = company_dict.get(investor.lower(), investor)
    
    investor_row = {
        'investor_name': mapped_name,
        'parent_company_of_investor': None  # To be filled if available
    }
    tables_df['Investor'] = pd.concat([tables_df['Investor'], 
                                      pd.DataFrame([investor_row])], 
                                      ignore_index=True)

tables_df["Investor"].dropna(how="all", inplace=True)
tables_df["Investor"].to_csv("Intermediate Results/Investor_AIDDATA.csv", encoding='utf-8', index=False)



# #NOTE transaction table includes all enteries + in intermidate folder
# #NOTE debt is saved to current final tables directly!
# # Create transaction and debt tables
# for idx, row in aid_data.iterrows():
#     # Generate investment_id for this power plant
#     investment_id = create_placeholder(row, 'investment_id', 'DEBTID_AIDDATA_', idx)
    
#     # Get the number of lenders
#     number_of_lenders = len(row['investor_list'])
    
#     # Calculate if investment is cofinanced
#     investment_cofinanced = 1 if number_of_lenders > 1 else 0
    
#     # Create transaction entries
#     for investor in row['investor_list']:
#         # Map investor name using dictionary
#         mapped_investor = company_dict.get(investor.lower(), investor)
        
       
#         transaction_row = {
#             'investment_id': investment_id,
#             'investor_name': mapped_investor,
#             'amount': row['Amount (Constant USD 2021)'] / number_of_lenders if number_of_lenders > 0 else 0,
#             'investment_averaged': "Y",  # Default for AidData
#             'investment_weighted': "N",   # Default for AidData
#             'r_id': None
#         }
#         tables_df['Transaction'] = pd.concat([tables_df['Transaction'], 
#                                             pd.DataFrame([transaction_row])], 
#                                             ignore_index=True)
        
#         # Create debt entry
#         debt_id = create_placeholder({}, 'debt_id', 'DEBTKEY_AIDDATA_', len(tables_df['Debt']))
        
#         debt_row = {
#             'debt_id': debt_id,
#             'PP_key': row['PP_key'],
#             'debt_investment_year': row['Commitment Year'],
#             'debt_investment_amount': row['Amount (Constant USD 2021)'] / number_of_lenders if number_of_lenders > 0 else 0,
#             'investment_cofinanced_bu': investment_cofinanced,
#             'number_of_lenders': number_of_lenders,
#             'bu_id_bu_cgp': None,  # Not applicable for AidData
#             'bu_id': None,         # Not applicable for AidData
#             'i_id': None,          # Not applicable for AidData
#             'r_id': r_id
#         }
#         tables_df['Debt'] = pd.concat([tables_df['Debt'], 
#                                       pd.DataFrame([debt_row])], 
#                                       ignore_index=True)

# # Save intermediate results
# for table_name in investments_tables:
#     output_path = f"{intermediate_folder}/{table_name}_AIDDATA.csv"
#     tables_df[table_name].to_csv(output_path, index=False)
#     print(f"Saved {table_name} table with {len(tables_df[table_name])} rows to {output_path}")


# Match investments
def match_investments(row, threshold_year=1):
    potential,_ = match_power_plants(row, pp_city_country_debt)
    # Convert debt_investment_year to numeric, handling any non-numeric values
   # potential['debt_investment_year'] = pd.to_numeri(potential['debt_investment_year'], errors='coerce')
   # potential.reset_index()
    
    # Create boolean mask handling null values
    mask = (
        (potential['debt_investment_year'] <= row['Commitment Year'] + threshold_year) & 
        (potential['debt_investment_year'] >= row['Commitment Year'] - threshold_year)
    )
    # potential =potential.to_frame()
    # potential = potential.reset_index()
    # potential = potential.reset_index(drop=True)
  
    #print(potential.T)
    potential["debt_investment_year"]
    
    # Process funding agencies for the single matched row
    diff_dict = {}
    for agency in row['Funding Agencies'].split("|"):
        if agency.strip() in company_dict:
            agency_key = company_dict[agency.strip()]
            amount_per_agency = row['Amount (Nominal USD)'] / len(row['Funding Agencies'].split("|"))
            if agency_key in potential:
                diff_dict[agency_key] = abs(amount_per_agency - potential[agency_key] * 1000000)
    
    # Add differences to the Series
    potential['differences_per_bank'] = diff_dict
    return potential

# Create investment, debt, and transaction tables
investment_rows = []
debt_rows = []
transaction_rows = []
umatched_debt_rows = []
debt_columns = ['debt_id', 'PP_key', 'debt_investment_year', 'debt_investment_amount', 'investment_cofinanced_bu', 'number_of_lenders', 'bu_id_bu_cgp', 'bu_id', 'j_id', 'r_id']
[print(x) for x in pp_city_country_debt.columns]
for i, row in aid_data.iterrows():
    matched_pp, _ = match_power_plants(row, pp_city_country_debt)
    if matched_pp is not None:
        matched_investment = match_investments(row)
        if matched_investment is not None:  # Check if we got a match
            
            investment_rows.append([row['AidData Record ID'], row['Commitment Year'], 
                                   row['Funding Agencies'], row['Amount (Nominal USD)']])

            debt_rows.append([matched_pp[c] for c in debt_columns])
    else:
        umatched_debt_rows.append([row['AidData Record ID'], row['Commitment Year'], 
                                    row['Amount (Nominal USD)']])

# Save tables
investment_table = pd.DataFrame(investment_rows, columns=['AidData Record ID','Commitment Year', 'Funding Agencies', 'Amount'])
debt_table = pd.DataFrame(debt_rows, columns=debt_columns)
debt_unmatched_table = pd.DataFrame(umatched_debt_rows, columns=['aiddata_id','debt_investment_year', 'debt_investment_amount'])
debt_unmatched_table["PP_key"] = ""
debt_unmatched_table[['investment_cofinanced_bu', 'number_of_lenders', 'bu_id_bu_cgp', 'bu_id', 'j_id', 'r_id']] = ""
debt_unmatched_table = create_unique_id(debt_unmatched_table, "PP_key", "PLANTKEY_AIDDATA_")
debt_unmatched_table = create_unique_id(debt_unmatched_table, "debt_id", "DEBTID_AIDDATA_")
debt_table["aiddata_id"] = ""
final_debt_table = pd.concat([debt_df, debt_unmatched_table])

# Create Transaction table for the unmatched debt entries
for i, row in debt_unmatched_table.iterrows():
    # Find the corresponding AidData row
    aid_row = aid_data[aid_data['AidData Record ID'] == row['aiddata_id']].iloc[0] if not pd.isna(row['aiddata_id']) else None
    
    if aid_row is not None:
        # Process funding agencies to create transaction entries
        if 'Funding Agencies' in aid_row and not pd.isna(aid_row['Funding Agencies']):
            agencies = aid_row['Funding Agencies'].split('|')
            amount_per_agency = aid_row['Amount (Nominal USD)'] / len(agencies) if len(agencies) > 0 else 0
            
            for agency in agencies:
                agency_name = agency.strip().lower()
                if agency_name:
                    # Create transaction row
                    transaction_row = {
                        'investment_id': row['debt_id'],  # Use debt_id as investment_id
                        'investor_name': agency_name,
                        'amount': amount_per_agency,
                        'investment_averaged': 'N',
                        'investment_weighted': 'N',
                        'r_id': row.get('r_id', ''),
                        'original_source': 'AIDDATA'
                    }
                    transaction_rows.append(transaction_row)

# Create Transaction DataFrame
transaction_table = pd.DataFrame(transaction_rows)



# Save tables

investment_table.dropna(how="all", inplace=True)
transaction_table.dropna(how="all", inplace=True)
final_debt_table.dropna(how="all", inplace=True)

investment_table.to_csv(f"{intermediate_folder}/Investment_AIDDATA.csv", index=False, encoding='utf-8')
final_debt_table.to_csv(f"{current_final_tables_folder}/Debt.csv", index=False, encoding='utf-8')
transaction_table.to_csv(f"{intermediate_folder}/Transaction_AIDDATA.csv", index=False, encoding='utf-8')

print(f"Created Transaction table with {len(transaction_table)} new rows")



# Create empty tables for all entities we need to populate
tables_df = get_tables_to_create_empty(map_table[db_structure_sheet])

# ---------------------- Create Country and City Tables ---------------------------

# Step 1: Extract country and city data from matched entries
print("Creating Country and City tables from AidData...")

# First, handle matched entries that already have a PP_key
for entry in matched_entries:
    # For Country table - use 'Recipient' as the country name
    country_name = entry.get('Recipient', "")
    if country_name != "":
        new_country_row = {}
        for col in tables_df["Country"].columns:
            if col == "country":
                new_country_row[col] = country_name
            else:
                new_country_row[col] = entry.get(col, None)
        
        # Only add if this country doesn't already exist
        if country_name not in tables_df["Country"]["country"].values:
            tables_df["Country"] = pd.concat([tables_df["Country"], 
                                          pd.DataFrame([new_country_row])], 
                                          ignore_index=True)
    
    # For City table
    for column in tables_df["City"].columns:
        if column in entry and entry[column] is not None:
            # For city_key, use the PP_key as in other files
            if column == "city_key" and "PP_key" in entry:
                entry["city_key"] = entry["PP_key"]
            
            # Add the row to City table
            new_city_row = {col: entry.get(col, None) for col in tables_df["City"].columns}
            
            # Fill country column with the respective country from Recipient
            if "country" in tables_df["City"].columns:
                #new_city_row["country"] = entry.get('Recipient', None)
                new_city_row["country"] = country_name
                
            if "city" in new_city_row and new_city_row["city"] is not None:
                tables_df["City"] = pd.concat([tables_df["City"], 
                                           pd.DataFrame([new_city_row])], 
                                           ignore_index=True)

# Step 2: Handle unmatched entries that were assigned new PP_keys
for entry in unmatched_entries:
    # For Country table - use 'Recipient' as the country name
    country_name = entry.get('Recipient', "")
    if country_name != "":
        new_country_row = {}
        for col in tables_df["Country"].columns:
            if col == "country":
                new_country_row[col] = country_name
            else:
                new_country_row[col] = entry.get(col, None)
        
        # Only add if this country doesn't already exist
        if country_name not in tables_df["Country"]["country"].values:
            tables_df["Country"] = pd.concat([tables_df["Country"], 
                                          pd.DataFrame([new_country_row])], 
                                          ignore_index=True)
    
    # For City table
    for column in tables_df["City"].columns:
        if column in entry and entry[column] is not None:
            # For city_key, use the PP_key as in other files
            if column == "city_key" and "PP_key" in entry:
                entry["city_key"] = entry["PP_key"]
            
            # Add the row to City table
            new_city_row = {col: entry.get(col, None) for col in tables_df["City"].columns}
            
            # Fill country column with the respective country from Recipient
            if "country" in tables_df["City"].columns:
                #new_city_row["country"] = entry.get('Recipient', None)
                new_city_row["country"] = country_name
                
            if "city" in new_city_row and new_city_row["city"] is not None:
                tables_df["City"] = pd.concat([tables_df["City"], 
                                           pd.DataFrame([new_city_row])], 
                                           ignore_index=True)

# Step 3: Handle debt_unmatched_table entries that were assigned new PP_keys
for _, row in debt_unmatched_table.iterrows():
    # Find the corresponding AidData row
    aid_row = aid_data[aid_data['AidData Record ID'] == row['aiddata_id']].iloc[0] if not pd.isna(row['aiddata_id']) else None
    
    if aid_row is not None:
        # For Country table - use 'Recipient' column as the country name
        country_name = aid_row.get('Recipient', "")
        if country_name != "":
            new_country_row = {}
            for col in tables_df["Country"].columns:
                if col == "country":
                    new_country_row[col] = country_name
                else:
                    new_country_row[col] = aid_row.get(col, None)
            
            # Only add if this country doesn't already exist
            if country_name not in tables_df["Country"]["country"].values:
                tables_df["Country"] = pd.concat([tables_df["Country"], 
                                              pd.DataFrame([new_country_row])], 
                                              ignore_index=True)
        
        # For City table
        city_name = aid_row.get('city', None)
        if city_name is not None:
            new_city_row = {col: aid_row.get(col, None) for col in tables_df["City"].columns}
            new_city_row["country"] = country_name
            new_city_row["city_key"] = row["PP_key"]  # Use the newly created PP_key
            tables_df["City"] = pd.concat([tables_df["City"], 
                                       pd.DataFrame([new_city_row])], 
                                       ignore_index=True)

# Remove duplicate rows from Country and City tables
tables_df["Country"] = tables_df["Country"].drop_duplicates(subset=["country"], keep="first")
tables_df["City"] = tables_df["City"].drop_duplicates(subset=["city_key", "city"], keep="first")

# Print statistics for the created tables
print(f"Created Country table with {len(tables_df['Country'])} rows")
print(f"Created City table with {len(tables_df['City'])} rows")

# Save Country and City tables to intermediate results
tables_df["Country"].dropna(how="all", inplace=True)
tables_df["City"].dropna(how="all", inplace=True)

tables_df["Country"].to_csv(f"{intermediate_folder}/Country_AIDDATA.csv", index=False, encoding='utf-8')
tables_df["City"].to_csv(f"{intermediate_folder}/City_AIDDATA.csv", index=False, encoding='utf-8')


print("NUMBER OF NULLS IN COUNTRY COLUMN IN CITY TABLE")
print(tables_df["City"].isnull().sum())
print(aid_data_full["Recipient"].isna().sum())
print(tables_df["Country"].isnull().sum())
# for the investments h process 3la el matched w el unmatched el matched hykon 3ndhom nfs el pp_key bta3 el matched bta3to bs
# el fekra en el debt table msh saved in n intermidiate file, lel current final table bs
# fa h add 3leh bs el new debt 
# do it one by one!
# investor
# debt
# investment
# country
# city - DONE