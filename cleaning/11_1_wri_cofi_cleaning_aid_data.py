#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Unified Cleaning Script for AidData

This script:
  1. Loads the original AidData and COFI datasets.
  2. Filters to the ENERGY sector and keeps only relevant columns.
  3. Extracts information from the Title field (fuel, capacity, additional info).
  4. Performs a “round 2” of extraction to fill missing project name/location info
     (using regex and fallback patterns) and extracts location (city/province)
     from Description via GPT.
  5. Extracts missing installed_capacity and primary_fuel from Description via
     regex and GPT and cleans these values.
  6. Adds the AidData Record ID from the original database.
  7. Saves the final output as "AidData - with extracted info and record ID v4.xlsx".

Before running:
  • Ensure the required Excel files are in the correct folders.
  • Install necessary packages: pandas, numpy, openai, tiktoken, spacy, etc.
  • Set your OpenAI API key below.
"""
#-------FIRST PART OF AID DATA CLENAING : POWER PLANT----------#
import pandas as pd
import numpy as np
import re
from statistics import mean
from itertools import combinations
import openai
import tiktoken
import streamlit as st
import os
# Uncomment the following if using spaCy for any additional NLP tasks:
# import spacy
#@TODO update paths

# ==============================
#  CONFIGURATION & API KEYS
# ==============================

# Set your OpenAI API key here:
openai.api_key = os.environ.get("OPENAI_API_KEY")

# Load Master Mapping file
mapping = './Mapping Files/COFI Mapping_columns.xlsx' # note this is 
mapping_source_sheet = 'Source'


# Load the name of source dataset for this code
source_name = 'AidData'

# Load fuel mapping file
fuel_map = './Mapping Files/AidData fuels dictionary.xlsx'

# Cleand file output directory
outname = './Cleaned Data/'

map_df = pd.read_excel(mapping, sheet_name = mapping_source_sheet)
print(map_df["Source_Name"].unique())
map_temp = map_df.loc[map_df['Source_Name'] == source_name]
map_temp = map_temp.reset_index(drop = True)
print(map_temp)
# Load fuel mapping file
fuels_df = pd.read_excel(fuel_map)

if len(map_temp) > 1:
    appended_data = []
    print("FILES SEEN BY CLEANING aidData")
    for i in map_temp.index:
        # load Source Data
        filename = map_temp.loc[i,'File_Name']
        print(filename)
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
    print("SINGLE FILE ", filename)
    starting_row = map_temp.loc[0,'Starting_row']
    sheet = map_temp.loc[0,'Sheet_name']

    if '.xls' in filename:
        if pd.notna(sheet):
            df = pd.read_excel(filename, sheet_name=sheet, skiprows=starting_row)
        else:
            df = pd.read_excel(filename, skiprows=starting_row)
    elif filename.endswith('.csv'):
        df = pd.read_csv(filename)

## GET ENERGY FILE ##

aid = df[df["Sector Name"] == "ENERGY"].copy(deep=True)
aid_full = df.copy(deep=True)


# ==============================
#  UTILITY FUNCTIONS
# ==============================

def adjust_present_column(df):
    """
    Adjust the Present column:
      - If Present is missing or empty, examine Description.
      - Split Description into sentences (by ". ") and select those
        that contain "purpose of the project" (case-insensitive).
      - Join such sentences with " # " and assign to Present.
    """
    if "Present" not in df.columns:
        df["Present"] = ""
    else:
        df["Present"] = df["Present"].fillna("")
    for i, row in df.iterrows():
        if row["Present"] == "":
            description = row.get("Description", "")
            sentences = description.split(". ")
            res = [s.strip() for s in sentences if "purpose of the project" in s.lower()]
            if res:
                df.at[i, "Present"] = " # ".join(res)
    return df

def exctract_fuel(text):
    """Extract fuel from Title using a regex pattern."""
    if "Power Plant" in text:
        pattern = r"(.*?)(Power Plant)"
    else:
        pattern = r"(.*?)(Plant)"
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip().split(" ")[-1]
    return ""

def exctract_capacity_before_parenthesis(text):
    pattern = r"\b(\w+)\s*\("
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip().split(" ")[-1]
    return ""

def do_multiplication(text, x_symbol):
    before = int(text.split(x_symbol)[0])
    after = float(text.split(x_symbol)[-1])
    return str(before * after)

def exctract_capacity(text):
    """Extract installed capacity from a text snippet (from Title, Present, or Description)."""
    pattern_capacity_mw = r"(.*?)(MW)"
    pattern_capacity_gw = r"(.*?)(GW)"
    match = re.search(pattern_capacity_mw, text)
    if match:
        word = match.group(1).strip().split(" ")[-1]
        if "(" in word or ")" in word:
            word = exctract_capacity_before_parenthesis(match.group(1).strip())
        if "x" in word:
            word = do_multiplication(word, "x")
        elif "×" in word:
            word = do_multiplication(word, "×")
        word = word.replace(",", "")
        try:
            return float(word)
        except:
            return ""
    match = re.search(pattern_capacity_gw, text)
    if match:
        word = match.group(1).strip().split(" ")[-1]
        try:
            return float(word) * 1000
        except:
            return ""
    return ""

def extract_words_between_capacity_and_fuels(text, special_words, start_substring):
    pattern = re.compile(rf'{start_substring}\s+(.+?)\s+(?:{"|".join(special_words)})')
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return ""

def extract_words_before_capacity(text):
    pattern = re.compile(r"for\s+(.+?)(?=\s+\d+MW)")
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return ""

def extract_additional_info(df, fuels_keywords):
    """For rows with missing additional_info, try extracting from Title using various patterns."""
    for i, row in df.iterrows():
        if row['additional_info'] == "":
            start_substring = "MW" if "MW" in row['Title'] else ("GW" if "GW" in row['Title'] else "for")
            info = extract_words_between_capacity_and_fuels(row['Title'], fuels_keywords, start_substring)
            if info == "":
                info = extract_words_before_capacity(row['Title'])
            df.at[i, "additional_info"] = info
    # Manually fix a known case
    for i, row in df.iterrows():
        if row['Title'].strip() == "Rositas Hydroelectric Power Plant":
            df.at[i, "additional_info"] = "Rositas"
    return df

def get_response(prompt: str) -> str:
    """Call the OpenAI GPT API with the provided prompt."""
    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant.'},
        {'role': 'user', 'content': prompt}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print("GPT call error:", e)
        return ""

def num_tokens_from_string(string: str, encoding) -> int:
    return len(encoding.encode(string))

def extract_after_colon(text):
    """Extract text that comes after a colon."""
    pattern = re.compile(r":\s+(.*)")
    match = pattern.search(text)
    if match:
        return match.group(1).strip()
    return ""

def replace_custom_values(text, values_map):
    for key, val in values_map.items():
        text = text.replace(key, val)
    return text

# ==============================
#  MAIN PROCESSING
# ==============================

#@TODO ashyl 

    
# Keep only rows likely about power plants (Title contains "Plant")
plant = aid[aid['Title'].str.contains("Plant", case=False, na=False)].reset_index(drop=True)

print(plant.shape)
    # Extract primary_fuel from Title
plant['primary_fuel'] = plant['Title'].apply(exctract_fuel)
fuel_dict = fuels_df.copy(deep=True)
for i, row in plant.iterrows():
        if row['primary_fuel'] in fuel_dict and fuel_dict[row['primary_fuel']] != "":
            plant.at[i, 'primary_fuel'] = fuel_dict[row['primary_fuel']]
    
    # Extract installed_capacity from Title
plant["installed_capacity"] = plant['Title'].apply(exctract_capacity)
    
    # Extract additional_info (project name/location) from Title
plant["additional_info"] = ""

fuels_keywords = [x.replace(")", "") for x in list(fuels_df['Old'])]
for i, row in plant.iterrows():
        start_substring = "MW" if "MW" in row['Title'] else ("GW" if "GW" in row['Title'] else "for")
        info = extract_words_between_capacity_and_fuels(row['Title'], fuels_keywords, start_substring)
        plant.at[i, "additional_info"] = info
print("Saving intermediate file: AidData - with extracted info.xlsx")
    #plant.to_excel("AidData - with extracted info.xlsx", index=False)
    
    # --- Part 2: Round 2 Extraction (Improve Additional Info & Location Extraction) ---
print("Performing round 2 extraction from Title...")
    #aid_round2 = pd.read_excel("AidData - with extracted info.xlsx")
aid_round2 = plant.copy(deep=True)
    
for col in ['primary_fuel', "installed_capacity", "additional_info"]:
        aid_round2[col] = aid_round2[col].fillna("")
    # Update missing additional_info using additional regex patterns
aid_round2 = extract_additional_info(aid_round2, fuels_keywords)
print("Saving intermediate file: AidData - with extracted info v2.xlsx")
    #aid_round2.to_excel("AidData - with extracted info v2.xlsx", index=False)
    
    # --- Part 2b: Extract Location Info from Description via GPT ---
print("Extracting location info (city/province) using GPT...")
    #aid_v2 = pd.read_excel("AidData - with extracted info v2.xlsx")
aid_v2 = aid_round2.copy(deep=True)
print(aid_v2.shape, aid_v2.columns)
aid_v2 = adjust_present_column(aid_v2)
aid_v2['Present'] = aid_v2['Present'].fillna("")
prompt_intro = (
        "\nIn the following text, extract the location of the power plant. Return the city "
        "and the province/district/region in this format:\n"
        "* City: [city]\n* Province: [province]\n"
        "Return 'N/A' if not available.\n"
    )
    # Get GPT responses for unique non-empty 'Present' values
present_answers_dict = {}
for text in aid_v2['Present'].unique():
        if text.strip():
            response = get_response(prompt_intro + text)
            present_answers_dict[text] = response
aid_v2['GPT answer'] = ""
for i, row in aid_v2.iterrows():
        if row['Present'].strip():
            aid_v2.at[i, 'GPT answer'] = present_answers_dict.get(row['Present'], "")
    # For rows where 'Present' is empty, use the Description field
desc_answers_dict = {}
for text in aid_v2.loc[aid_v2['GPT answer'] == ""]['Description'].unique():
        if text.strip():
            response = get_response(prompt_intro + text)
            desc_answers_dict[text] = response
for i, row in aid_v2.iterrows():
        if not row['GPT answer'].strip():
            aid_v2.at[i, 'GPT answer'] = desc_answers_dict.get(row['Description'], "")
    # Parse the GPT answer for city and province
aid_v2['city'] = ""
aid_v2['province'] = ""
for i, row in aid_v2.iterrows():
        for line in row['GPT answer'].split("\n"):
            if "city" in line.lower():
                aid_v2.at[i, "city"] = extract_after_colon(line)
            elif "province" in line.lower():
                aid_v2.at[i, "province"] = extract_after_colon(line)
    # Clean city and province values
aid_v2['city'] = aid_v2['city'].apply(lambda x: x.replace("N/A", "").strip())
aid_v2['province'] = aid_v2['province'].apply(lambda x: x.replace("N/A", "").strip())
aid_v2['city'] = aid_v2['city'].apply(lambda x: replace_custom_values(x, {"commune": ""}))
aid_v2['province'] = aid_v2['province'].apply(lambda x: replace_custom_values(x, {'district': "", "division": "", "region": "", "province": ""}))
print("Saving intermediate file: AidData - with extracted info v3.xlsx")
    #aid_v2.to_excel("AidData - with extracted info v3.xlsx", index=False)
    
    # --- Part 3: Extract Missing Capacity and Fuel from Description via Regex and GPT ---
print("Extracting missing installed_capacity and primary_fuel...")
    #aid_v3 = pd.read_excel("AidData - with extracted info v3.xlsx")
aid_v3 = aid_v2.copy(deep=True)
for col in ['primary_fuel', "installed_capacity", "additional_info", "city", "province"]:
        aid_v3[col] = aid_v3[col].fillna("")
aid_v3['Present'] = aid_v3.get('Present', "").fillna("")
    # For rows missing installed_capacity, try extracting from 'Present' or 'Description'
for i, row in aid_v3.iterrows():
        if row['installed_capacity'] == "":
            if "MW" in row['Present'] or "GW" in row['Present']:
                cap = exctract_capacity(row['Present'])
                if cap != "":
                    aid_v3.at[i, "installed_capacity"] = cap
            elif "MW" in row['Description'] or "GW" in row['Description']:
                cap = exctract_capacity(row['Description'])
                if cap != "":
                    aid_v3.at[i, "installed_capacity"] = cap
    # For rows still missing capacity or primary_fuel, call GPT
missing = aid_v3.loc[(aid_v3['installed_capacity'] == "") | (aid_v3['primary_fuel'] == "")].reset_index(drop=True)
prompt_capacity = (
    "\nIn the following text, return the installed capacity of the whole power plant (in MW) "
    "in the format:\n* capacity: [result]\nIf no information, return 'N/A'.\n"
)
prompt_fuel = (
    "\nIn the following text, return the primary fuel of the power plant in the format:\n* fuel: [result]\n"
    "If no information, return 'N/A'.\n"
)
prompt_both = (
    "\nIn the following text, return both the primary fuel and the installed capacity of the power plant "
    "in the format:\n* fuel: [result]\n* capacity: [result]\nIf no information, return 'N/A'.\n"
)
missing['CPG answer'] = ""
for i, row in missing.iterrows():
    if row['installed_capacity'] == "" and row['primary_fuel'] != "":
        response = get_response(prompt_capacity + row['Description'])
        missing.at[i, 'CPG answer'] = response
    elif row['installed_capacity'] != "" and row['primary_fuel'] == "":
        response = get_response(prompt_fuel + row['Description'])
        missing.at[i, 'CPG answer'] = response
    elif row['installed_capacity'] == "" and row['primary_fuel'] == "":
        response = get_response(prompt_both + row['Description'])
        missing.at[i, 'CPG answer'] = response
# Parse GPT answers to update capacity and fuel
for i, row in missing.iterrows():
    for line in row.get('CPG answer', "").split("\n"):
        if "capacity" in line.lower() and missing.at[i, "installed_capacity"] == "":
            missing.at[i, "installed_capacity"] = extract_after_colon(line)
        elif "fuel" in line.lower() and missing.at[i, "primary_fuel"] == "":
            missing.at[i, "primary_fuel"] = extract_after_colon(line)
# Clean installed_capacity: remove units, handle ranges/multiplications, etc.
missing['installed_capacity'] = missing['installed_capacity'].astype(str)\
    .apply(lambda x: x.replace("N/A", "").replace("MW", "").replace("megawatts", "").replace("megawatt", "").strip())
missing['installed_capacity'] = missing['installed_capacity'].apply(lambda x: "" if "tons" in x or "kV" in x else x)
for i, row in missing.iterrows():
    cap_str = row['installed_capacity']
    if "-" in cap_str:
        try:
            parts = [float(num.strip()) for num in cap_str.split("-")]
            missing.at[i, "installed_capacity"] = str(mean(parts))
        except:
            pass
    if "*" in cap_str:
        try:
            parts = [float(num.strip()) for num in cap_str.split("*")]
            missing.at[i, "installed_capacity"] = str(parts[0] * parts[1])
        except:
            pass
    if "kW" in cap_str:
        try:
            num = float(cap_str.replace("kW", "").strip())
            missing.at[i, "installed_capacity"] = str(num * 0.001)
        except:
            pass
    missing.at[i, "installed_capacity"] = missing.at[i, "installed_capacity"].replace(",", "")
# Convert capacity strings to float when possible
for i, row in missing.iterrows():
    if row['installed_capacity'] != "":
        try:
            missing.at[i, "installed_capacity"] = float(row['installed_capacity'])
        except:
            pass
# Clean primary_fuel using a new fuels dictionary
#fuels_df_new = pd.read_excel("New fuels dictionary.xlsx").fillna("")
fuels_df.fillna("", inplace=True)
fuel_dict_new = {row['Old'].strip(): row['New'].strip() for i, row in fuels_df.iterrows()}
for i, row in missing.iterrows():
    if row['primary_fuel'] in fuel_dict_new and fuel_dict_new[row['primary_fuel']] != "":
        missing.at[i, 'primary_fuel'] = fuel_dict_new[row['primary_fuel']]
    missing.at[i, 'primary_fuel'] = row['primary_fuel'].replace("N/A", "").replace("Unknown", "").replace("Not specified", "").strip()
print("Saving capacity and fuel cleaned file: Capacity and fuel extracted - all cleaned.xlsx")
missing.to_excel("Capacity and fuel extracted - all cleaned.xlsx", index=False)
# Merge cleaned missing rows back into the main dataset (matching on Title)
for i, row in missing.iterrows():
    title = row['Title']
    aid_v3.loc[aid_v3['Title'] == title, "installed_capacity"] = row['installed_capacity']
    aid_v3.loc[aid_v3['Title'] == title, "primary_fuel"] = row['primary_fuel']
print("Saving intermediate file: AidData - with extracted info v4.xlsx")
#aid_v3.to_excel("AidData - with extracted info v4.xlsx", index=False)

# --- Part 4: Add AidData Record ID ---
print("Adding AidData Record ID from original database...")
#new_aid = pd.read_excel("AidData - with extracted info v4.xlsx")
new_aid = aid_v3.copy(deep=True)
db = df.copy(deep=True)
new_aid["AidData Record ID"] = ""
for i, row in new_aid.iterrows():
    aid_match = db.loc[db['Title'] == row['Title']]
    if not aid_match.empty:
        new_aid.at[i, "AidData Record ID"] = aid_match['AidData Record ID'].values[0]
# Optionally, reorder columns so that the Record ID comes first.
cols = [new_aid.columns[-1]] + list(new_aid.columns[:-1])
new_aid = new_aid[cols]
final_filename = "AidData - with extracted info and record ID v4_generated.xlsx"
new_aid.to_excel(final_filename, index=False)
print("Final file saved:", final_filename)



#@TODO "Present" column geh mnen?????????
#@TODO between v2 and v3 initialize present then set null to empty string