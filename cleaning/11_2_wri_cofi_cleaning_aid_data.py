

import re
import pandas as pd
import numpy as np
import re
import datetime
from statistics import mean
import openai
import tiktoken

#@TODO Update paths

#-------SECONDPART OF AID DATA CLENAING : ENERGY----------#

# IMPORT LOCAL FUNCTIONS
#from COFI_cleaning_functions import extract_plant, extract_capacity, extract_fuel, extract_fuel_proxy, region_clean

"""### Set Variables
These variables can be adjusted upfront or via frontend platform without visiting the algorithm code
"""

# Set source data location
#path = r'G:\My Drive\ZFG Insights\Project Folder\51. WRI\1. COFI NLP Project\4. Cleaned Data Source'

# Load Master Mapping file
mapping = './Mapping Files/COFI Mapping_columns.xlsx'
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

aid_en = df[df["Sector Name"] == "ENERGY"].copy(deep=True)


##############################################################################
# STEP 1: LOAD & FILTER (from "1 - extracting info - name, capacity, fuel REGEX")
##############################################################################

# Keep only relevant columns
columns_to_keep_aid = [
    "AidData Record ID", "Title", "Recipient ISO-3", "Commitment Year",
    "Funding Agencies", "Co-financing Agencies", "Amount (Nominal USD)",
    "Amount (Constant USD 2021)", "Adjusted Amount (Constant USD 2021)",
    "Flow Type", "M&A", "Description"
]
aid = aid_en[columns_to_keep_aid].copy(deep=True)
aid = aid.reset_index(drop=True)

# 2. Filter the data to pick only rows that describe new power plants
#    (Solar, wind, hydro, etc.). We mimic the logic from the script.
keywords = ["wind", "solar", "solar pv", "photovoltaic", "hydropower",
            "coal", "gas", "oil", "waste to energy", "biomass", "geothermal"]

def has_special_word_project(text, special_words, ending_word=" project"):
    pattern = rf"\b({'|'.join(special_words)})\b{ending_word}"
    return bool(re.search(pattern, text, flags=re.IGNORECASE))

aid["to_keep"] = aid["Title"].apply(lambda x: has_special_word_project(x, keywords))
aid["reason"] = aid['to_keep'].apply(lambda x: "new pattern" if x is True else "")

# Next, handle other patterns that mention "power station", "power project", "Dam" logic, etc.
def is_also_about_dam(text):
    # checks whethre the entry with "Transmission Line" should be kept
    words = text.split(" ")

    # if there is no "and" then the investment is only about the transmission line
    if "and" not in words:
        return False
    # check the indexes
    if words.index("Dam") < words.index("and") and words.index("and") < words.index("Transmission"):
        return True
    else:
        return False

for i, row in aid.iterrows():
    if row["to_keep"] is True:
        continue
    
    title_lower = row["Title"].lower()
    # Additional patterns
    if "power station" in title_lower:
        aid.at[i, "to_keep"] = True
        aid.at[i, "reason"] = "Power Station"
    elif "power project" in title_lower:
        aid.at[i, "to_keep"] = True
        aid.at[i, "reason"] = "Power Project"
    elif "hydroelectric station" in title_lower:
        aid.at[i, "to_keep"] = True
        aid.at[i, "reason"] = "Hydroelectric Station"
    elif "Dam" in row["Title"]:
        if "Transmission" not in row["Title"]:
            aid.at[i, "to_keep"] = True
            aid.at[i, "reason"] = "Dam"
        else:
            # Keep only if "Dam" < "and" < "Transmission"
            if is_also_about_dam(row["Title"]):
                aid.at[i, "to_keep"] = True
                aid.at[i, "reason"] = "Dam"

print(aid[aid["to_keep"] == True]["reason"].value_counts())

# Subset the data
plant_new = aid.loc[aid["to_keep"] == True].copy()
plant_new = plant_new.reset_index(drop=True)

print("Shape plant_new right after 1 - extracting info - name, capacity, fuel REGEX : ", plant_new.shape)




##############################################################################
# STEP 2: EXTRACT PRIMARY FUEL, CAPACITY, and NAME via REGEX (round 1 logic)
##############################################################################

# 2.1 Load the fuels dictionary
fuels_dict = dict(zip(fuels_df["Old"], fuels_df["New"]))
fuels_list = sorted(list(fuels_df["Old"]), key=len, reverse=True)  # remove them in order of decreasing length

# 2.2 Define a function to extract the primary fuel via rule-based approach
def extract_fuel_rule_based(text):
    tmp = text
    found_fuels = []
    for f in fuels_list:
        if f in tmp:
            found_fuels.append(f)
            tmp = tmp.replace(f, "")

    # For Dam
    if len(found_fuels) == 0 and "Dam" in text:
        # Assume hydro if not otherwise found
        return "Hydroelectric"

    # If nothing found
    if len(found_fuels) == 0:
        return ""

    # If only one
    if len(found_fuels) == 1:
        return fuels_dict.get(found_fuels[0], found_fuels[0])

    # If two, handle known combos
    if len(found_fuels) == 2:
        f1, f2 = found_fuels
        # Make sure we handle in a consistent order
        pair = set([f1.lower(), f2.lower()])
        if pair == set(["photovoltaic", "solar"]):
            return fuels_dict.get("Solar", "Solar")
        if pair == set(["combined cycle", "gas"]):
            return fuels_dict.get("Gas", "Gas")
        if pair == set(["thermal", "coal"]):
            return fuels_dict.get("Thermal", "Thermal")
        # Otherwise join them
        return "|".join([fuels_dict.get(x, x) for x in found_fuels])

    # More than 2
    return "|".join([fuels_dict.get(x, x) for x in found_fuels])

plant_new["primary_fuel"] = plant_new["Title"].apply(extract_fuel_rule_based)

# 2.3 Define capacity-extraction logic
capacity_pattern_mw = r"(.*?)(MW)"
capacity_pattern_gw = r"(.*?)(GW)"
capacity_pattern_kw = r"(\w+)(?:\s*(kW|KW))"

def exctract_capacity_before_parenthesis(text):
    pattern = r"\b(\w+)\s*\("
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip().split(" ")[-1]
    else:
        return ""

def do_multiplication(text, x_symbol):
    before = float(text.split(x_symbol)[0].replace(",", "").strip())
    after = float(text.split(x_symbol)[1].replace(",", "").strip())
    return str(before * after)

def do_average(text):
    values = [float(v.strip()) for v in text.split("-")]
    return str(mean(values))

def extract_capacity(text):
    # Try MW
    match = re.search(capacity_pattern_mw, text)
    if match:
        word_before = match.group(1).strip().split(" ")[-1]
        if "(" in word_before or ")" in word_before:
            word_before = exctract_capacity_before_parenthesis(match.group(1).strip())
        if "x" in word_before:
            word_before = do_multiplication(word_before, "x")
        elif "×" in word_before:
            word_before = do_multiplication(word_before, "×")
        if "-" in word_before:
            word_before = do_average(word_before)
        word_before = word_before.replace(",", "")
        try:
            return float(word_before)
        except:
            return ""
    # Try GW
    match = re.search(capacity_pattern_gw, text)
    if match:
        word_before = match.group(1).strip().split(" ")[-1]
        word_before = word_before.replace(",", "")
        try:
            return float(word_before)*1000  # Convert GW -> MW
        except:
            return ""

    # Try kW
    match = re.search(capacity_pattern_kw, text)
    if match:
        word_before = match.group(1).strip().split(" ")[-1]
        word_before = word_before.replace(",", "")
        try:
            return float(word_before)*0.001 # Convert kW -> MW
        except:
            return ""
    return ""

plant_new["installed_capacity"] = plant_new["Title"].apply(extract_capacity)

# 2.4 Extract potential name info (label as 'additional_info')
#    This chunk tries multiple patterns in sequence: [MW/GW -> fuels], [MW/GW -> Dam], etc.
#    For simplicity, we combine them directly (this is a condensed approach from your script).
def extract_additional_info(title, fuels_keywords):
    """
    Combine multiple search patterns to approximate the 'additional_info'
    from the 'Title' as done in the original notebooks.
    """
    # We'll do a simpler layered approach for demonstration:
    # 1) [MW/GW...some text...fuel keywords], or [MW...some text...Power Station], etc.
    # 2) [for/on/fund <name> Dam], etc.
    # For brevity, we replicate only the main patterns from your script.

    def extract_between(text, start_list, end_list):
        """Generic pattern [start_list] (.+?) [end_list]. Returns the first match or ""."""
        pattern = rf'\b({"|".join(start_list)})\s+(.+?)\s+\b({"|".join(end_list)})\b'
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return match.group(2).strip()
        return ""

    # 1) Attempt [MW|GW] -> [some text] -> [Fuel keywords]
    start_cap = []
    if "MW" in title:
        start_cap.append("MW")
    if "GW" in title:
        start_cap.append("GW")
    if "kW" in title:
        start_cap.append("kW")
    if "KW" in title:
        start_cap.append("KW")

    # If we found a capacity pattern
    if start_cap:
        # e.g. [MW] (some text) [fuel]
        fuel_info = extract_between(title, start_cap, fuels_keywords)
        if fuel_info:
            return fuel_info

        # e.g. [MW] (some text) [Power Station, Power Project, Dam]
        possible_ends = ["Power Station", "Power Project", "Dam"]
        station_info = extract_between(title, start_cap, possible_ends)
        if station_info:
            return station_info

    # 2) Attempt dam pattern: [for/on/fund/finance] <name> "Dam"
    dam_start = ["for", "on", "fund", "finance"]
    dam_end = ["Dam"]
    dam_info = extract_between(title, dam_start, dam_end)
    if dam_info:
        return dam_info

    # 3) Attempt [at/in <text> ... ] at end
    def find_at_in_end(text, keywords):
        for k in keywords:
            pattern = rf'\b({k})\s+(.+)$'
            m = re.search(pattern, text)
            if m:
                return m.group(2).strip()
        return ""
    at_in_info = find_at_in_end(title, ["at", "in"])
    if at_in_info:
        # minimal cleanup
        # If it's obviously just a country or something else you don't want, you could do more filtering.
        return at_in_info

    return ""

# Fuel keywords for name extraction (remove parentheses to avoid regex confusion)
fuels_keywords_cleaned = [f.replace(")", "") for f in fuels_list]

plant_new["additional_info"] = plant_new["Title"].apply(
    lambda x: extract_additional_info(x, fuels_keywords_cleaned)
)

# Save intermediate result (optional, for debugging)
# plant_new.to_excel("AidData - New Plants with extracted info (regex).xlsx", index=False)

print("Plant new shape after EXTRACT PRIMARY FUEL, CAPACITY, and NAME via REGEX (round 1 logic)", plant_new.shape)

##############################################################################
# STEP 3: FILL MISSING CAPACITY/FUEL USING GPT (from "2 - extracting info - capacity and fuel GPT")
##############################################################################

# For brevity, we keep the essential approach:
#  - We check if 'installed_capacity' is missing or 'primary_fuel' is missing.
#  - We call GPT with a prompt that requests the missing fields from 'Description'.

# Setup your OpenAI key and model as needed

openai.api_key = os.environ.get("OPENAI_API_KEY")
# GPT prompts
prompt_capacity = (
    "In the following text, return the information regarding the installed capacity "
    "of the power plant whose investment the text discusses. The capacity must be the "
    "capacity of the whole plant, not the additional capacity. Return in format:\n"
    "* capacity: [result]\n\n"
    "If none found, return 'N/A'.\n\n"
)
prompt_fuel = (
    "In the following text, return the information regarding the primary fuel of the power "
    "plant whose investment the text discusses. Return in format:\n"
    "* fuel: [result]\n\n"
    "If none found, return 'N/A'.\n\n"
)
prompt_both = (
    "In the following text, return the information regarding the primary fuel and the "
    "installed capacity of the power plant whose investment the text discusses. The "
    "capacity must be the capacity of the whole plant, not additional capacity. Return in format:\n"
    "* fuel: [result]\n"
    "* capacity: [result]\n\n"
    "If none found, return 'N/A'.\n\n"
)

def get_chatgpt_response(prompt_text):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt_text},
    ]
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages
        )
        return response["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error calling GPT: {e}")
        return ""

# Identify rows missing capacity/fuel
need_gpt = plant_new.loc[(plant_new["installed_capacity"] == "") | (plant_new["primary_fuel"] == "")]
need_gpt = need_gpt.reset_index(drop=True)

# Prepare a new column to store GPT raw answers
need_gpt["GPT answer - capacity/fuel"] = ""

# 3.1 Query GPT for each row needing capacity/fuel
for i, row in need_gpt.iterrows():
    desc = row["Description"] if pd.notna(row["Description"]) else ""
    if row["installed_capacity"] == "" and row["primary_fuel"] != "":
        # Need capacity only
        query_prompt = prompt_capacity + desc
    elif row["installed_capacity"] != "" and row["primary_fuel"] == "":
        # Need fuel only
        query_prompt = prompt_fuel + desc
    else:
        # Need both
        query_prompt = prompt_both + desc

    gpt_response = get_chatgpt_response(query_prompt)
    need_gpt.at[i, "GPT answer - capacity/fuel"] = gpt_response

# 3.2 Parse GPT responses into 'installed_capacity' and 'primary_fuel'
def extract_value_after_colon(line):
    """
    Extract text after colon. E.g. "* capacity: 450 MW" -> "450 MW"
    """
    match = re.search(r":\s+(.*)", line)
    if match:
        return match.group(1).strip()
    return ""

for i, row in need_gpt.iterrows():
    lines = row["GPT answer - capacity/fuel"].split("\n")
    updated_capacity = row["installed_capacity"]
    updated_fuel = row["primary_fuel"]

    for ln in lines:
        lower_ln = ln.lower()
        if "capacity" in lower_ln and updated_capacity == "":
            extracted = extract_value_after_colon(ln)
            updated_capacity = extracted
        elif "fuel" in lower_ln and updated_fuel == "":
            extracted = extract_value_after_colon(ln)
            updated_fuel = extracted

    need_gpt.at[i, "installed_capacity"] = updated_capacity
    need_gpt.at[i, "primary_fuel"] = updated_fuel

# 3.3 Clean the newly populated capacity/fuel columns
def clean_capacity(val):
    if not isinstance(val, str):
        return val
    val = val.replace("N/A", "").replace("Unknown", "").strip()
    val = val.replace("MW", "").replace("megawatts", "").replace("megawatt", "").strip()
    # remove units not relevant
    for forbidden in ["tons", "kV", "barrel-per-day", "million cubic meters"]:
        if forbidden in val.lower():
            return ""
    # convert "2,500" -> "2500"
    val = val.replace(",", "")
    # handle dash (e.g. "700-750" -> average
    if "-" in val:
        try:
            parts = [float(x) for x in val.split("-")]
            return str(mean(parts))
        except:
            return ""
    # handle "x" or "*"
    if "*" in val:
        try:
            parts = [float(x) for x in val.split("*")]
            return str(parts[0] * parts[1])
        except:
            return ""
    # handle "kW" or "KW"
    if "kw" in val.lower():
        # remove possible "kW" text:
        val = re.sub(r"(?i)kw", "", val).strip()
        # interpret as kW, so convert to MW
        try:
            num_val = float(val)
            return str(num_val * 0.001)
        except:
            return ""
    # final check if numeric
    try:
        float_val = float(val)
        return str(float_val)
    except:
        return ""
    
def clean_fuel(val, fuel_dictionary):
    """
    Map GPT fuel strings to standardized COFI or your dictionary entries.
    """
    if not isinstance(val, str):
        return ""
    val = val.replace("N/A", "").replace("Unknown", "").replace("Not specified", "").strip()
    return fuel_dictionary.get(val, val)  # If GPT produced an exact match in your dictionary

# Build a more extensive dictionary if needed from your local "New fuels dictionary - Round 3.xlsx"
# For demonstration, let's assume the original 'fuels_dict' from Round 1 plus some expansions.
expanded_fuel_mapping = dict(fuels_dict)  # you can update with new keys from GPT if needed

for i, row in need_gpt.iterrows():
    # Clean capacity
    updated_cap = clean_capacity(row["installed_capacity"])
    need_gpt.at[i, "installed_capacity"] = updated_cap

    # Clean fuel
    updated_fuel = clean_fuel(row["primary_fuel"], expanded_fuel_mapping)
    need_gpt.at[i, "primary_fuel"] = updated_fuel

# 3.4 Merge GPT-updated rows back into the main dataframe
#    We'll join on the unique combination (Title, possibly ID if available).
plant_new.set_index("Title", inplace=True)
need_gpt.set_index("Title", inplace=True)

for idx in need_gpt.index:
    if idx in plant_new.index:
        plant_new.at[idx, "installed_capacity"] = need_gpt.at[idx, "installed_capacity"]
        plant_new.at[idx, "primary_fuel"] = need_gpt.at[idx, "primary_fuel"]

plant_new.reset_index(inplace=True)

print("Plant new shape after 2 - extracting info - capacity and fuel GPT ", plant_new.shape)
##############################################################################
# STEP 4: FILL MISSING CITY/PROVINCE USING GPT (from "3 - extracting info - city and province")
##############################################################################

prompt_location = (
    "In the following text, extract the information regarding the location of the "
    "power plant being invested in. Return the city of the power plant and the province (or district/region). "
    "Use the format:\n\n"
    "* City: [answer]\n"
    "* Province: [answer]\n\n"
    "Return 'N/A' if no information.\n\n"
)

def get_city_province_from_gpt(description):
    resp = get_chatgpt_response(prompt_location + str(description))
    return resp

plant_new["GPT answer - city and province"] = ""

for i, row in plant_new.iterrows():
    gpt_reply = get_city_province_from_gpt(row["Description"])
    plant_new.at[i, "GPT answer - city and province"] = gpt_reply

# Parse out city & province from GPT reply
def extract_after_colon(line):
    m = re.search(r':\s+(.*)', line)
    if m:
        return m.group(1).strip()
    return ""

plant_new["city"] = ""
plant_new["province"] = ""

for i, row in plant_new.iterrows():
    lines = row["GPT answer - city and province"].split("\n")
    city_val, prov_val = "", ""
    for ln in lines:
        lower_ln = ln.lower()
        if "city" in lower_ln:
            city_val = extract_after_colon(ln)
        elif "province" in lower_ln:
            prov_val = extract_after_colon(ln)
    plant_new.at[i, "city"] = city_val
    plant_new.at[i, "province"] = prov_val

# Clean city/province "N/A"
plant_new["city"] = plant_new["city"].apply(lambda x: "" if "N/A" in str(x) else x.strip())
plant_new["province"] = plant_new["province"].apply(lambda x: "" if "N/A" in str(x) else x.strip())

print("plant new shape after 3 - extracting info - city and province ", plant_new.shape )

##############################################################################
# STEP 5: SAVE THE FINAL FILE
##############################################################################

plant_new.to_excel("AidData - New Plants with extracted info - all FINAL_v2.xlsx", index=False)

print("Done! The final file is saved as 'AidData - New Plants with extracted info - all FINAL.xlsx'.")


#@TODO last step , merging the file

##############################################################################
# STEP 6: MERGE WITH POWER PLANT
##############################################################################
