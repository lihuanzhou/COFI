import pandas as pd
import numpy as np
import re
import datetime


''' Load mapping files '''
# Set source data location
# path = r'G:\My Drive\ZFG Insights\Project Folder\51. WRI\1. COFI NLP Project\4. Cleaned Data Source'
# Load fuel mapping file
fuel_map = './Mapping Files/STI-reference_primary-fuel-mapping.xlsx'

# Set master path
# import os
# os.chdir(path)

# Load fuel mapping file
fuel_map = pd.read_excel(fuel_map)




# Function to extract power plant / project name from data
    
''' Logic as: 
    1. if parenthesis exist with capacity inside, then extract all text before the last parenthesis. e.g. Coopers Gap (453MW) and Rye Park (396MW) Wind Farms Refinancing 2023 <> Coopers Gap and Rye Park Wind Farms 
    2. if capacity unit exist, then extract between capacity unit and the mention of "plant", with optional location name mentioned afterwards;
    3. if capacity doesn't exist, then extract between the last stopword and the mention of "plant", with optional location name mentioned afterwards
    4. remove words like "refinance"
'''

# Function to detect capacity unit in parenthesis
def contains_capacity_unit(text):
    # Regular expression to find patterns of the form (###UNIT)
    pattern = r'\(\d+(\.\d+)?\s*(MW|GW|KW)\)'
    
    # Search for the pattern in the text
    match = re.search(pattern, text)
    
    # Return True if a match is found, False otherwise
    return bool(match)


def split_by_last_capacity_unit(text):
    # Regular expression to find patterns of the form (###UNIT)
    pattern = r'\(\d+(\.\d+)?\s*(MW|GW|KW)\)'
    
    # Find all matches of the pattern
    matches = list(re.finditer(pattern, text))
    
    # If matches were found, split the text at the last match
    if matches:
        # Get the index of the start of the last match
        last_match_start = matches[-1].start()
        # Return the part of the text before the last match
        return text[:last_match_start]

# Extract powerplant name if capacity unit exist in string and not in parenthesis
# def extract_powerplant_w_unit(text):
#     # Step 1: Split the string by the first occurrence of any of the capacity unit
#     match = re.split(r'(\d+\.?\d*\s*(MW|KW|GW))', text, 1)
#     match = [item for item in match if item and len(item.strip()) > 1]
#     # Step 2: iterate the split parts, and get power plant name
#     for x in match:
#         pattern = (r"(plant(s)?|station(s)?|project(s)?|facility(s)?|farm(s)?|generation(s)?|field(s)?|generator(s)?|dam(s)?|LNG|pole(s)?|hydropower|drill(ing)?|park(s)?)")
#         x = x.strip()
#         # Search for the pattern in the text
#         match1 = re.search(pattern, x, re.IGNORECASE)
#         if match1:
#             # Return the matched text, which includes capacity and location
#             first_part = x
            
#             # Step 3: Find the last occurrence of 'of', 'from', 'the' in the first part (case insensitive)
#             last_preposition_match = list(re.finditer(r'\b(of|from|the|for|in)\b', first_part, flags=re.IGNORECASE))

#             if last_preposition_match:
#                 # Step 3: Extract the text after the last occurrence of 'of', 'from', 'the'
#                 last_preposition_start = last_preposition_match[-1].end()
#                 result =  first_part[last_preposition_start:].strip()
#             else:
#                 # If no such prepositions or articles found, return the first part
#                 result =  first_part.strip()

#             return result
#         break

# def extract_powerplant_w_unit(text):
#     # Step 1: Split the string by the first occurrence of any of the capacity unit
#     match = re.split(r'(\d+\.?\d*\s*(MW|KW|GW))', text, 1)
#     match = [item for item in match if item and len(item.strip()) > 1]
#     # Step 2: iterate the split parts, and get power plant name
#     for x in match:
#         pattern = (r"(plant(s)?|station(s)?|project(s)?|facility(s)?|farm(s)?|generation(s)?|field(s)?|generator(s)?|dam(s)?|LNG|pole(s)?|hydropower|drill(ing)?|park(s)?|onshore)")
#         x = x.strip()
#         # Search for the pattern in the text
#         match1 = re.search(pattern, x, re.IGNORECASE)
#         if match1:
#             # Return the matched text, which includes capacity and location
#             first_part = x

#             # Step 3: Find the SECOND occurrence of 'of', 'from', 'the' in the first part (case insensitive)
#             pattern2 = r'\b(of|from|the|for|and)\b'
#             matches = list(re.finditer(pattern2, first_part, flags=re.IGNORECASE))

#             # Check if there are at least two stopwords
#             if len(matches) > 1:
#                 # Find the position right after the second stopword
#                 end_pos = matches[1].end()
#                 # Return the truncated text
#                 truncated_text = first_part[:end_pos]
#                 # Remove any trailing stopwords
#                 truncated_text = re.sub(pattern2, '', truncated_text)
#                 return truncated_text.strip()
#             # If less than two stopwords are found, return the original text
#             return first_part.strip()
#             break  

    
# Extract powerplant name if capacity unit doesn't exist
def extract_substring_powerplant(text):
    # replace refinancing faclity
    text = re.sub(r'refinancing facility', '', text, flags=re.IGNORECASE)
    
    # Step 1: Split the string by the first occurrence of any of the specified keywords (case insensitive)
    match = re.split(r'\b(plant(s)?|station(s)?|project(s)?|(power|generation|generating) facility(s)?|farm(s)?|generation(s)?|field(s)?|generator(s)?|dam(s)?|LNG|pole(s)?|hydropower|drill(ing)?|park(s)?|onshore|hydroelectric|ccgt)\b', text, 1, flags=re.IGNORECASE)
    match = [item for item in match if item and len(item.strip()) > 1]
    if len(match) >= 2:
        # Concatenate the parts to include the keyword in the first part
        first_part = match[0] + match[1]
    else:
        # Return an empty string if no keyword is found
        return ""
    
    pattern2 = r'\b(of|from|the|for|and|by)\b'
    # Step 2: Find the last occurrence of 'of', 'from', 'the' in the first part (case insensitive)
    last_preposition_match = list(re.finditer(pattern2, first_part, flags=re.IGNORECASE))
    
    if last_preposition_match:
        # Step 3: Extract the text after the last occurrence of 'of', 'from', 'the'
        last_preposition_start = last_preposition_match[-1].end()
        result =  first_part[last_preposition_start:].strip()
    else:
        # If no such prepositions or articles found, return the first part
        result =  first_part.strip()
        
    # Step 3: include the optional location after powerplant name
    if len(match)>2:
        if match[2].startswith((" in", " at")):
            result = result + match[2]
            
#     # Step 4: Find the SECOND occurrence of 'of', 'from', 'the' in the first part (case insensitive)
    matches = list(re.finditer(pattern2, result, flags=re.IGNORECASE))
    # Check if there are at least two stopwords
    if len(matches) > 1:
        # Find the position right after the second stopword
        end_pos = matches[1].end()
        # Return the truncated text
        result = result[:end_pos]
        # Remove any trailing stopwords
        result = re.sub(pattern2, '', result)
    
    # remove the second sentence if exists
    result = result.split('.')[0]
    
    return result.strip()


# Extract powerplant name if capacity unit exist in string and not in parenthesis
def extract_powerplant_w_unit(text):
    # Step 1: Split the string by the first occurrence of any of the capacity unit
    match = re.split(r'(\d+\.?\d*\s*(MW|KW|GW))', text, 1)
    match = [item for item in match if item and len(item.strip()) > 1]
    # Step 2: iterate the split parts, and get power plant name
    for x in match:
        pattern = (r"(plant(s)?|station(s)?|project(s)?|(power|generation|generating) facility(s)?|farm(s)?|generation(s)?|field(s)?|generator(s)?|dam(s)?|LNG|pole(s)?|hydropower|drill(ing)?|park(s)?|onshore|hydroelectric|ccgt)")
        x = x.strip()
        # Search for the pattern in the text
        match1 = re.search(pattern, x, re.IGNORECASE)
        if match1:
            # Return the matched text, which includes capacity and location
            first_part = x
            return extract_substring_powerplant(first_part)
            break  



# Main function to extract powerplant
def extract_plant (text):  
    # Ensure text is a string
    if pd.isna(text):
        return None
    text = str(text)  # Convert to string in case of non-string types
    
    # Logic 1
    if contains_capacity_unit(text):
        text = split_by_last_capacity_unit(text)
    
    else:
        # Logic 2
        # remove parentheses
        text = re.sub(r'\([^)]*\)', '', text).replace('  ','')

        # if capacity exist, extract between capacity unit and "Plant", with optional location afterwards
        if any(x in text for x in ['MW', 'GW', 'KW']):
            text = extract_powerplant_w_unit(text)
        else:
            text = extract_substring_powerplant(text)

        # remove terms like "Phase", "Upgrade", or "Renewal"
        try:
            if 'phase' in text.lower():
                text = re.sub(r'[ ,]*(phase|upgrade|renewal).*', '', text.lower())
        except:
            pass
        
        # remove the result if project finance or refiennce exist
        try:
            if re.search(r'\b(project finance|refinance|refinances|refinancing)\b', text, flags=re.IGNORECASE):
                return None
        except:
            pass
        
    # Keep only the powerplant name from the extracted power plant names
    try:
        text = re.sub('(?<!^)\b\w*(wind farm|solar.{0,10}farm|power plant|power project|dam|power.{0,10}station)(.*)?','',text,flags=re.I).strip()
        text = re.sub(r'\s*\b(' + '|'.join(fuel_map['fuel_original']) + r')\b$','',text,flags=re.I).strip()
    except:
        pass

    return text



# Function to extract capacity from project name
def extract_capacity(text):
    # Ensure text is a string
    if pd.isna(text):
        return None
    text = str(text)  # Convert to string in case of non-string types
    
    # Updated pattern to capture numbers with commas and optional decimal parts
    #pattern = r'(\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?\s*(MW|GW|KW)'
    pattern = r'((?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?\s*(MW|GW|KW))'
    matches = re.findall(pattern, text)

    capacities = []
    for match in matches:
        # Directly append the matched string, which includes the unit
        capacities.append(match[0])
        #capacities.append(''.join(match))

    # Joining the list to a single string, if needed
    capacity = ', '.join(capacities)
    return capacity



# Function to extract primary fuel, exact match
def extract_fuel (text):
    # Ensure text is a string
    if pd.isna(text):
        return None
    text = str(text)  # Convert to string in case of non-string types

    # Convert fule mapping into a dictionary 
    mapping_dict = dict(zip(fuel_map['fuel_original'], fuel_map['fuel_mapped']))
    
    # Map fuels
    for original_fuel, mapped_fuel in mapping_dict.items():
        pattern = r'\b{}\b'.format(re.escape(original_fuel.lower()))
        text = re.sub(pattern, mapped_fuel, text, flags=re.IGNORECASE)
    
    return text.lower()


# Function to extract primary fuel, contains fuel text
def extract_fuel_proxy (text):
    # Ensure text is a string
    if pd.isna(text):
        return None
    text = str(text)  # Convert to string in case of non-string types

    # Convert fule mapping into a dictionary 
    mapping_dict = dict(zip(fuel_map['fuel_original'], fuel_map['fuel_mapped']))
    
    # Map fuels
    for fuel_original, fuel_mapped in mapping_dict.items():
        if re.search(r'\b{}\b'.format(fuel_original), text.lower()):
            return fuel_mapped
    return None


# Function to clean region and city variables 
def region_clean (text):
    # Ensure text is a string
    if pd.isna(text):
        return None
    text = str(text)  # Convert to string in case of non-string types
    
    text = text.replace('Not Specified','')
    return text


