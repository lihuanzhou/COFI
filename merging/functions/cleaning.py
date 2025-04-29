import pandas as pd
import string
import numpy as np
import re
import os
import time
from datetime import datetime
from thefuzz import fuzz


####################### ACTUALLY DOING SOME CLEANING #######################

def my_reset_index(db):
    # resets the index one time and then removes the column containing the old index
    db = db.reset_index()
    db = db.drop(columns=['index'])
    return db

def preprocess_text(text, remove_punctuation = True):
  # Implement your preprocessing logic here (e.g., lowercase conversion)

    new_string = text.lower()
    new_string = new_string.strip()

    new_string = re.sub(' +', ' ', new_string)

    if remove_punctuation is True:
        # Create a translation table 
        # translator = str.maketrans('', '', string.punctuation) # substitute the punctuations with empty string
        translator = str.maketrans(string.punctuation, ' ' * len(string.punctuation)) # substitute the punctuations with white space
        # Remove punctuation 
        new_string = new_string.translate(translator) 
        new_string = re.sub(' +', ' ', new_string)

    return new_string

def extract_between_underscore_period(text):
  """
  This function extracts the text between an underscore and a period using regex.

  Args:
      text: The string to search.

  Returns:
      The extracted text between the underscore and period, or None if not found.
  """
  pattern = r"(?<=_).+?(?=\.)"  # Positive lookbehind and lookahead for capturing
  match = re.search(pattern, text)
  return match.group(0) if match else None

def extract_between_outside_underscores(text):
  """
  This function extracts the data between the two outermost underscores in a string.

  Args:
      text: The string to search.

  Returns:
      The extracted data between the outermost underscores, or None if no underscores are found or there's only one underscore.
  """
  pattern = r"(?<=_)(.*)(?=_[^_]*)"  # Positive lookbehind and lookahead with negative lookahead
  match = re.search(pattern, text)
  return match.group(1) if match else None

def my_read_excel(path):
    # reads the excel in path
    # returns it without any unneeded columns (such as the annoying "Unnamed: 0" column)
    db = pd.read_excel(path)
    if "Unnamed: 0" in db:
        db = db.drop(columns=["Unnamed: 0"])
    return db

def my_read_excel(folder_path, file_name, verbose=True, low_memory=None):
    # reads an excel/csv file, prints when it was last modified, and
    # returns it without the unwanted "Unnamed: 0" if present
    file_path = folder_path + file_name
    if verbose == True:
        last_modified = datetime.fromtimestamp(os.path.getmtime(file_path))
        print(f"{file_name}: last modified on: {last_modified.year}-{last_modified.month}-{last_modified.day}")

    if "xlsx" in file_name:
        db = pd.read_excel(file_path)
    elif "csv" in file_name:
        if low_memory is None:
            db = pd.read_csv(file_path)
        else:
            db = pd.read_csv(file_path, low_memory=low_memory)
    else:
        print("ERROR: the extension of the file should be 'xlsx' or 'csv'")
        return None

    if "Unnamed: 0" in db.columns:
        db = db.drop(columns=['Unnamed: 0'])

    return db

def change_weird_apostrophe(text):
    text = text.replace("’", "'").replace("’", "'")
    return text

####################### SUPPORT FUNCTIONS TO PRINT STUFF #######################

def print_last_modified_file_date_in_folder(folder_name):
    # prints when the files in the folder were last modified
    if folder_name[-1] != "/":
        folder_name = folder_name + "/"
    files = [x for x in os.listdir(folder_name)]
    last_modified = max([time.strptime(time.ctime(os.path.getmtime(folder_name + x))) for x in files])
    print(f"Data was last modified on: {last_modified.tm_year}-{last_modified.tm_mon}-{last_modified.tm_mday}")

def print_final_statistics(tables_df, tables_to_create):
    # prints how many rows per table we created
    # and prints some checks on the number of rows

    # get how many rows per table
    sizes = {}
    for ent in tables_to_create:
        sizes[ent] = tables_df[ent].shape[0]
    # print
    for ent in sizes:
        print(f"{ent}: {sizes[ent]}")
    print("")
    # do checks based on whether whether the tables were indeed created
    # size Power Plant == size City == size Country as we 
    # created them from the same subset of data and at the same time
    if "Power Plant" in sizes:
        print(f"Power Plant, City, and Country have same amount of rows: {sizes['Power Plant'] == sizes['City'] <= sizes['Country']}")
    # there are more rows in Debt that in Transaction and Investor because
    # TODO: for IJ_Global the checks for Equity do not work because in Transaction tehre is both Equity and Debt rows!!
    # we already merged the information in Debt
    if "Debt" in sizes:
        print(f"Debt has been merged already: {(sizes['Debt'] >= sizes['Transaction'])}")
    # same as for Debt
    if "Equity" in sizes:
        print(f"Equity has been merged already: {(sizes['Equity'] >= sizes['Transaction'])}")
    # in Investor there are the same amount or more of rows than Transaction 
    # because each Transaction has at least one investor
    if "Transaction" in sizes:
        print(f"There are enough Transaction rows for the Investors: {sizes['Investor'] >= sizes['Transaction']}")

def count_sources_in_id(db, id_col, db_name="data"):
    # count how many rows there are in db from each data source by using id_col (e.g., "PP_key")

    # extract the source name
    sources_list = [extract_between_outside_underscores(x) for x in db[id_col]]
    # for each unique source then compute
    for source in sorted(set(sources_list)):
        print(f"Rows in {db_name} from {source}: {sources_list.count(source)}")


####################### SUPPORT FUNCTIONS TO SET UP OTHER PROCESSES #######################

def get_all_files(folder, name):
    # gets all the files in a folder that have a certain word in the name (e.g., it could be a table name such as "Country")
    # return [x for x in os.listdir(folder) if name in x]
    return [x for x in get_sorted_files(folder) if name in x]

def get_sorted_files(folder_path):
  # returns the files in folder_path ordered from last modified to oldest modified
  files = []
  for filename in os.listdir(folder_path):
    file_path = os.path.join(folder_path, filename)
    # Check if it's a regular file (not a directory)
    if os.path.isfile(file_path):
      # Get the last modification time
      last_modified = os.path.getmtime(file_path)
      # Convert timestamp to datetime object (optional for sorting by date)
      last_modified_dt = datetime.fromtimestamp(last_modified)
      files.append((filename, last_modified_dt))  # Add filename and modification time

  # Sort the list by the last modification time (descending)
  files.sort(key=lambda x: x[1], reverse=True)

  # Return only the filenames (optional)
  return [filename for filename, _ in files]

def get_variables(vertical_slicing, source_name, tables):
    # function that gets all the variables to keep given a database for the tables that need to be created
    # returns the variables in a lits format
    # vertical_slicing: the dataset that contains the information of which variables belong to which table
    # source_name: the name of the database that we are working on (e.g., WEPP)
    # tables: the tables that we want to create in the end

    # read only the variables that refer to the tables that we are creating here & that refer to the database of interest
    variables_list = vertical_slicing.loc[vertical_slicing['Name of new table'].isin(tables)][source_name].to_list()

    # remove the "0" which are the empty lines if we create the Varibles directly in Excel
    # if we download it from Google Sheet then it is filled with np.nan for the empty lines
    # TODO: this is done only because this sheet that I am reading is a copy of another sheet, probably in the final version this shouldn't be the case so this needs to be updated
    while 0 in variables_list:
        variables_list.remove(0)
    while np.nan in variables_list: 
        variables_list.remove(np.nan)

    # in the dataframe there could be lines where there are multiple variables written there
    variables = []
    for el in variables_list:
        variables.extend([x.strip() for x in el.split(";")])

    # clean the names
    variables = [x.strip() for x in variables]
    
    return list(set(variables))

def get_variables_advanced(vertical_slicing,  source_name, tables) -> dict:
    # function that gets all the variables to keep and how they should be renamed given a database for the tables that need to be created
    # returns the variables in a dictionary format
    # vertical_slicing: the dataset that contains the information of which variables belong to which table
    # source_name: the name of the database that we are working on (e.g., WEPP)
    # tables: the tables that we want to create in the end
    
    # read only the variables that refer to the tables that we are creating here & that refer to the database of interest
    variables_df = vertical_slicing.loc[vertical_slicing['Name of new table'].isin(tables)][["Variables", source_name]]

    # create the dictionary
    variables_dict = {}
    for i, row in variables_df.iterrows():
        if isinstance(row[source_name], str) and row[source_name] != 0:
            vars = [x.strip() for x in row[source_name].split(";")]
            for var in vars:
                variables_dict[var] = row["Variables"]


    # remove the "0" which are the empty lines
    # TODO: this is done only because this sheet that I am reading is a copy of another sheet, probably in the final version this shouldn't be the case so this needs to be updated
    while 0 in variables_dict:
        variables_dict.pop(0, None)
    
    # while np.nan in variables_dict: 
    #     variables_dict.remove(np.nan)

    # clean the names
    variables_dict = {x.strip():variables_dict[x].strip() for x in variables_dict}
    
    return variables_dict

def get_thresholds_methods_dicts(thresholds_df, methods_df, parameters_df, dataset_name, matching_situation):
    # functions thet returns the thresholds and methods to use when matching for the situation
    # described in matching_situation (the methods are already the functions that can be directly
    # used). It returns two dictionaries, one for the thresholds, one for the methods
    # thresholds_df: info about the threshold
    # methods_df: info about the methods
    # dataset_name: the source being processed
    # matching_situtation: "Debt", "Equity", or "Power Plant". It describes where the thresholds
    # and methods are being used (e.g., "Power Plant" means that we need the thresholds and methods
    # used to find matches in already existing Power Plants)
    # returns: two dictionaries, one for the thresholds, one for the methods. Both are structured like
    # this: {column for which the threshold/method applies: threshold/method} (e.g., {"power_plant_name": 80}
    # is used to get all the plants whose similarity score is higher than 80))


    # process the thresholds_df
    thrs_dict = turn_table_in_dict(thresholds_df, dataset_name, matching_situation, "Threshold")

    # process the methods_df 
    mtds_dict = turn_table_in_dict(methods_df, dataset_name, matching_situation, "Method")
    # for methods_df: already put there the actual functions
    for key in mtds_dict:
        mtds_dict[key] = get_function(mtds_dict[key])

    # process the parameters_df
    prmts_dict = turn_table_in_dict(parameters_df, dataset_name, matching_situation, "Value")

    return thrs_dict, mtds_dict, prmts_dict

def turn_table_in_dict(table, dataset_name, matching_situation, threshold_or_method):
    # creates a dictionary that summarize the given table. Each key is a database column
    # and the corresponding value is the corresponding threshold/method 
    # table: eitehr thresholds_df or methods_df or parameters_df
    # dataset_name: the source being processed
    # matching_situtation: "Debt", "Equity", or "Power Plant"
    # threshold_or_method: either "Threshold" for the trhesholds_df or "Method" for methods_df or "Value" for parameters_df

    final_dict = {} # final result
    
    # select only the rows that apply to the dataset_name and to the matching_situation
    relevant_rows = table.loc[(table['Data'] == dataset_name) & (table['Matching situation'] == matching_situation)]

    # iterate through these rows to fill up the dictionary
    for i, row in relevant_rows.iterrows():
        if not isinstance(row[threshold_or_method], str) or "[" not in row[threshold_or_method]:
            final_dict[row['Column'].strip()] = row[threshold_or_method]
        else:
            # we are dealing with a list of integers (which is present in parameters_df)
            # so convert it to list manually
            text = row[threshold_or_method].replace("[", "").replace("]", "")
            final_dict[row['Column'].strip()] = [int(x) for x in text.split(",")]

    return final_dict


def get_function(method_name):
    # transforms the method_name in the actual function
    # method_name: a string-based name of a function
    # returns: the function to use

    if method_name == "fuzz.partial_ratio":
        scorer = fuzz.partial_ratio
    elif method_name == "fuzz.token_set_ratio":
        scorer = fuzz.token_set_ratio
    elif method_name == "fuzz.ratio":
        scorer = fuzz.ratio
    else:
        # defualt method
        scorer = fuzz.partial_ratio

    return scorer

def get_joining_info(joining_info, name_row):
    """ Returns the list of variables to use to perform the joining between the datasources described in name_row.

    joining_info: db containing the joining information
    name_row: the name of the datasources to join

    returns: the list to use to perform the joining (in that order).
    """
    variables_text = joining_info.loc[joining_info['Databases'] == name_row]['Variables'].values[0]
    return [x.strip() for x in variables_text.split(",")]

def get_comm_year_thr(commissioning_year_thresholds_info, situation):
    # gets the threshold for the commissioning_year for aggregation or joining purposes
    return commissioning_year_thresholds_info.loc[commissioning_year_thresholds_info['Situation'] == situation]['Value'].values[0]


def prepare_for_joining(db):
    # creates the columns in db for the joining using a range for joining the commissioning_year
    db['year_range'] = db['year_range'].fillna("")
    db['year_range'] = db.apply(lambda row: f"{row['commissioning_year']}-{row['commissioning_year']}" if row['year_range'] == "" else row['year_range'], axis=1)

    db['lower_limit'] = db['year_range'].apply(lambda x: float(x.split("-")[0]) if "missing" not in x else np.nan)
    db['upper_limit'] = db['year_range'].apply(lambda x: float(x.split("-")[1]) if "missing" not in x else np.nan)

    return db


def find_equal_rows(df, target_row, columns):
    # gets the indexes of all the rows that are equals to the target_row based on the subset of columns
    return df[(df[columns] == target_row[columns]).all(axis=1)].index.to_list()

####################### FUNCTIONS THAT CREATE AND DEAL WITH UNIQUE IDs/PLACEHOLDERS #######################

def create_placeholder(row, col, prefix: str, unique_number: int | str):
    # returns an UNIQUE placeholder for the col cell in the given row if needed.
    # this take into account if the row already has a valid value for that column
    # the placeholder will be: prefix + unique_number
    # note that both prefix can be empty values
    # row: a DataFrame's row.
    # col: the column that needs to be fixed
    # prefix: the prefix that the placeholder will have
    # unique_number: a number to put in the placeholder. Ideally you'd want this number to be unique in the whole column
    # returns a placehholder if the col cell is empty, otherwise it returns the original value

    # if there is no value in col in the row
    if pd.isna(row[col]) or row[col] == "":
        # placeholder: prefix + unique_number

        # we make sure that if a number or string are given to the function, the function can deal with both
        if isinstance(unique_number, str):
            return prefix + unique_number
        else:
            return prefix + str(unique_number)
    
    # if there was already a valid "province", then return it
    return row[col]

def create_unique_id(db, col, prefix):
    # creates a unique id for each row as prefix + index of row
    # this doesn't take into account if the cell already has an ID or not
    # db: the db to add the unique id
    # col: the name of the column where the unique_ids will be
    # prefix: the prefix of the unique ids
    # returns: the db with the unique_ids
    
    for i, row in db.iterrows():
        db.at[i, col] = prefix + str(i)

    return db

def fill_column_in_rows(db, col, prefix: str):
    # function that given a dataset adds a UNIQUE placeholder in each row for the given col IF NEEDED
    # this function takes into account if there is already a valid value in the col:
    # if there is, it doesn't change it, otherwise it adds a unique placeholder
    db = db.copy(deep=True)
    for i, row in db.iterrows():
        db.at[i, col] = create_placeholder(row, col, prefix, unique_number = i)

    return db

def make_city_key(text, old_prefix = "PLANTKEY_", prefix = "CITYKEY_"):
    # creates the city_key from the PP_key by replacing the prefix present in PP_key (but not the name of the datasource)
    # text: the text that is the base for the new city_key. It is the old PP_key
    # old_prefix: the prefix to remove
    # prefix: the new prefix to add
    # returns the newly created city_key
    number = text.partition(old_prefix)[-1]
    return prefix + number


####################### FUNCTIONS THAT DEAL WITH STANDARDIZING THE "country" COLUMN #######################

def update_country_names(country_names_dict_df, data):
    counter = 0

    # create a dictionary which contains the renamings 
    country_names_dict = {}
    for i, row in country_names_dict_df.iterrows():
        country_names_dict[row['original name']] = row['UN name'].strip()

    # pre-process the string to faciliate comparison
    data['country'] = data['country'].apply(lambda x: x.lower())

    # make the change
    for i, row in data.iterrows(): # for each country in the data
        # if the country name needs changing
        if row['country'] in country_names_dict:
            new_name = country_names_dict[row['country']]
            new_name = change_weird_apostrophe(new_name)
            data.at[i, 'country'] = new_name
            counter+=1
        elif "’" in row['country'] or "’" in row['country']:
            data.at[i, 'country'] = change_weird_apostrophe(row['country'])

    print(f"Entries updated: {counter}")

    return data

def check_if_new_country_names(countries, un_data, country_names_dict_df):
    # this function is needed to check if we need to add some new renamings rule
    # in "country names dictionary" sheet in "Variables.xlsx"
    # TODO: this one doesn't require any pre-processing for countries,
    # so in step 1-4 and in the step cleaning remove the lowering before using this function!

    counter = 0
    new_name = []

    # lower the countries names
    countries = [change_weird_apostrophe(x.lower()) for x in countries]

    # a country name can either be a UN official country name or a synonym 
    un_names = [x.lower() for x in un_data['Country or Area'].to_list()]
    # we need to change this weird apostrophe that's in the "Côte d'Ivoire" name
    # that I can't seem to change
    un_names = [change_weird_apostrophe(x) for x in un_names]
    synonyms = country_names_dict_df['original name'].to_list()

    # check if the country name is an official name or an already known synonym
    for country in countries:   
        if (country not in un_names) and (country not in synonyms):
            # if it is not then it must be a new synonym!
            counter += 1
            new_name.append(country)
    return new_name


####################### FUNCTION THAT DEAL WITH STANDARDIZING THE "primary_fuel" COLUMN #######################

def clean_primary_fuel_column(db, primary_fuel_column_name = "primary_fuel", words_to_fix = ['other', "unknown", "unspecified"], additional_words_to_fix = []):
    """Cleans up the primary fuel column by turning the words that refer to unknown fuel types
    to np.NaNs.
    
    db: the dataframe to update
    primary_fuel_column_name: the column that contains the fuel types to update.
        The default value is "primary_fuel"
    words_to_fix: the words to turn into NaNs. The default ones are ['other', "unknown", "unspecified"].
    additional_words_to_fix: other words to turn into NaNs (in a list format). If words are specified here, the algorithm will clen
        the default ones and these ones as well.

    returns: the updated db

    """
    # add the new words
    words_to_fix = words_to_fix + additional_words_to_fix
    # print(words_to_fix)

    for to_fix in words_to_fix:
        db[primary_fuel_column_name] = db[primary_fuel_column_name].apply(lambda x: np.nan if x == to_fix else x)

    return db


def convert_primary_fuel(fuel_dict_df, db, other_value = None, other_values = None):
    # convert the primary_fuel values to follow what's in the dictionary
    # fuel_dict_df: a dataframe containing the conversions to do 
    # db: the db where to put the new values
    # other_value: this is the value that is used in the entries where it is not known the fuel
        # but it is not a NaN value (e.g., "other"). If it is passed by the user, then we change this value to
        # be NaN
        # the goal is to have all the fuel types that are not specified to be nan values
        # this facilitates then when we do the merging
    # other_values: a list of other_value to change
    # retuns: the updated db
    
    # turn it in a dictionary
    fuel_dict = {}
    for i, row in fuel_dict_df.iterrows():
        fuel_dict[row['Old']] = row['New']

    # make conversion
    db['primary_fuel'] = db['primary_fuel'].apply(lambda x: fuel_dict[x] if(x in fuel_dict) else x)

    if other_value != None:
        db['primary_fuel'] = db['primary_fuel'].apply(lambda x: np.nan if(x == other_value) else x)
    
    if other_values != None:
        for val in other_values:
            db['primary_fuel'] = db['primary_fuel'].apply(lambda x: np.nan if(x == val) else x)
    return db

####################### FUNCTION THAT DEAL WITH STANDARDIZING THE "equity_investment_type" COLUMN #######################


def convert_equity_investment_type(investment_type_df, db):
    # convert the equity_investment_type values to follow what's in the dictionary
       
    # turn it in a dictionary
    investment_dict = {}
    for i, row in investment_type_df.iterrows():
        investment_dict[row['Old']] = row['New']

    # make conversion
    db['equity_investment_type'] = db['equity_investment_type'].apply(lambda x: investment_dict[x] if(x in investment_dict) else x)

    return db

####################### FUNCTION THAT DEAL WITH STANDARDIZING THE "investor_name" COLUMN #######################

def fix_investor_names(db, investor_column, change_parent, parent_column = None, drop_duplicates=True):
    # deals with making a conversion in a dataframe
    # investor_column = the column with the investor names to change
    # change_parent = True if there is a parent column to change, False otherwise
    # parent_column = the column with the parent companies to update
    # drop_duplicates = True if we want to drop the investor duplicates, False otherwise

    original_rows_no = db.shape[0]

    # pre-process things
    db[investor_column] = db[investor_column].apply(lambda x: preprocess_text(x, True))

    # load custom dictionary
    custom = pd.read_excel("custom_investor_dictionary_final.xlsx")
    for col in ["Old", "New"]: # fix in case there are irrelevant white spaces that are left around
        custom[col] = custom[col].apply(lambda x: x.strip())
    # make conversion
    db = convert_custom_bank_dictionary(custom, db, investor_column)
    # drop duplicates
    if drop_duplicates == True:
        db = db.drop_duplicates(subset=investor_column)
        db = my_reset_index(db)

    # load equity investor dictionary
    mapping = pd.read_excel("equity investor master directory.xlsx")
    mapping = mapping.rename(columns={col:col.strip() for col in mapping.columns}) # remove the unneeded white spaces in the columns names
    mapping['List of equity investors'] = mapping['List of equity investors'].apply(lambda x: preprocess_text(x, True))
    # make conversion using the equity investor file
    db = convert_mapping_equity_investor(mapping, db, investor_column, False, None)
    # clean-up the new names
    db[investor_column] = db[investor_column].apply(lambda x: preprocess_text(x, True))   
    # drop duplicates
    if drop_duplicates == True:
        db = db.drop_duplicates(subset=investor_column)
        db = my_reset_index(db)

    # add parent compnay stuff
    if change_parent == True:
        db[parent_column] = db[parent_column].fillna("")
        db = change_parent_company(mapping, db, investor_column, parent_column)
    
    # print statistics
    new_rows_no = db.shape[0]
    diff_no = original_rows_no - new_rows_no
    print(f"New size: {new_rows_no}")
    print(f"Rows dropped #: {diff_no}")
    print(f"Rows dropped %: {diff_no / original_rows_no * 100}")

    return db

def convert_mapping_equity_investor(mapping_df, db, investor_column, change_parent, parent_column):
    # convert the equity investor names using the "equity investor master directory.xlsx" file
       
    # turn it in a dictionary
    investor_name = {}
    for i, row in mapping_df.iterrows():
        if not pd.isna(row['standard company name']):
            investor_name[row['List of equity investors']] = row['standard company name']

    # also get the parents company
    parent_name = {}
    for i, row in mapping_df.iterrows():
        parent_name[row['List of equity investors']] = row['standard parent company name']

    # make conversion for the parent company if needed
    if change_parent == True:
        for i, row in db.iterrows():
            if row[investor_column] in parent_name:
                db.at[i, parent_column] = parent_name[row[investor_column]]
    db[investor_column] = db[investor_column].apply(lambda x: investor_name[x] if(x in investor_name) else x)
    
    return db


def change_parent_company(mapping_df, db, investor_column, parent_column):
    # updates the parent company name with the values in "equity investor master directory.xlsx")
    
    if mapping_df is None:
        # load the file if it's not there
        print(f"Loading mappings from 'equity investor master directory.xlsx'")
        mapping_df = pd.read_excel("equity investor master directory.xlsx")
        mapping_df = mapping_df.rename(columns={col:col.strip() for col in mapping_df.columns}) # remove the unneeded white spaces in the columns names
        mapping_df['List of equity investors'] = mapping_df['List of equity investors'].apply(lambda x: preprocess_text(x, True))

    # transform it in a dictionary
    parent_name = {}
    for i, row in mapping_df.iterrows():
        parent_name[row['List of equity investors']] = row['standard parent company name']

    # make the conversion
    for i, row in db.iterrows():
        if row[investor_column] in parent_name:
            db.at[i, parent_column] = parent_name[row[investor_column]]

    return db


def convert_custom_bank_dictionary(custom_df, db, investor_column):
    # convert the investor names using the custom dicitonary
       
    # turn it in a dictionary
    investor_name = {}
    for i, row in custom_df.iterrows():
        investor_name[row['Old']] = row['New']

    # make conversion
    db[investor_column] = db[investor_column].apply(lambda x: investor_name[x] if(x in investor_name) else x)
    
    return db

####################### FUNCTION THAT DEALS WITH ASSIGNING THE SAME PP_KEY TO DUPLICATED ROWS #######################

def print_duplicated_power_plants_keys_info(db, full_subset_columns, pp_key_new, return_examples = False):
    # prints how many rows are duplicated 8but have different PP_keys) and some examples 

    # get the power plant columns that are in db (these are used to then determine the duplicated rows)
    full_subset_columns_in_db = list(set(full_subset_columns) & set(db.columns))
    subset_no_key = [x for x in full_subset_columns_in_db if x != "PP_key"] # the duplicated rows have now different PP_keys so we need to remove "PP_key" from the list
    print(f"The similarity of rows is computed on this columns: {subset_no_key}")

    # get the rows that wer not matched to pp_full's rows and determine if these rows were duplicated or not
    missing_power_plants_to_fix = db.loc[(db['PP_key'].str.contains(pp_key_new)) & (db['power_plant_name'] != "")].copy(deep=True)
    # the rows are duplicated if they share the same information with another row except for PP_key
    missing_power_plants_to_fix['duplicated'] = missing_power_plants_to_fix.duplicated(subset=subset_no_key) 

    # compute the number of rows that in the end will be dropped in the final Power Plant table because they are duplicated
    before_dropping = missing_power_plants_to_fix.shape[0]
    after_dropping = missing_power_plants_to_fix.drop_duplicates(subset = subset_no_key).shape[0]
    diff = before_dropping - after_dropping

    # if dropping doesn't lead to any change in the size of the table then there is no duplicated row
    # and we can just avoid running the following code
    if diff == 0:
        print("No need to fix!")
        return db, 0
    
    # if we are here then there is need to fix the PP_keys of the duplicated rows
    power_plants_duplicated = list(missing_power_plants_to_fix.loc[(missing_power_plants_to_fix['duplicated'] == True)]['power_plant_name'].unique())
    print(f"Duplicated rows #: {diff}")
    print(f"Examples of duplicated rows' power plants: {power_plants_duplicated[0:5]}")

    if return_examples == True:
        return power_plants_duplicated



def fix_duplicated_power_plants_keys(db, full_subset_columns, pp_key_new):
    # sets the same PP_key to rows that have the same values for the full_subset_columns other than PP_key
    
    db = db.copy(deep=True)

    # get the power plant columns that are in db (these are used to then determine the duplicated rows)
    full_subset_columns_in_db = list(set(full_subset_columns) & set(db.columns))
    subset_no_key = [x for x in full_subset_columns_in_db if x != "PP_key"] # the duplicated rows have now different PP_keys so we need to remove "PP_key" from the list
    print(f"The similarity of rows is computed on this columns: {subset_no_key}")

    # get the rows that wer not matched to pp_full's rows and determine if these rows were duplicated or not
    missing_power_plants_to_fix = db.loc[(db['PP_key'].str.contains(pp_key_new)) & (db['power_plant_name'] != "")].copy(deep=True)
    # the rows are duplicated if they share the same information with another row except for PP_key
    missing_power_plants_to_fix['duplicated'] = missing_power_plants_to_fix.duplicated(subset=subset_no_key) 

    # compute the number of rows that in the end will be dropped in the final Power Plant table because they are duplicated
    before_dropping = missing_power_plants_to_fix.shape[0]
    after_dropping = missing_power_plants_to_fix.drop_duplicates(subset = subset_no_key).shape[0]
    diff = before_dropping - after_dropping

    # if dropping doesn't lead to any change in the size of the table then there is no duplicated row
    # and we can just avoid running the following code
    if diff == 0:
        print("No need to fix!")
        return db, 0
    
    # if we are here then there is need to fix the PP_keys of the duplicated rows
    print(f"Duplicated rows #: {diff}")
    print(f"Duplicated rows %: {diff / db.shape[0] * 100}")
    print(f"Examples of duplicated rows' power plants: {missing_power_plants_to_fix.loc[(missing_power_plants_to_fix['duplicated'] == True)]['power_plant_name'].to_list()[0:5]}")
    print("\nFixing PP_keys....")

    # put placeholders where there are NaNs (otherwise find_equal_works won't work properly)
    for col in subset_no_key:
        db[col] = db[col].fillna("placeholder")

    # let's fix the rows that are duplicated
    # print(missing_power_plants_to_fix.loc[(missing_power_plants_to_fix['duplicated'] == True)]['PP_key'].unique())
    for key in missing_power_plants_to_fix.loc[(missing_power_plants_to_fix['duplicated'] == True)]['PP_key'].unique():
        # if this row has been already fixed (because there is more than 2 rows equal to one another), then doing this filtering leads to having a shape of 0
        if db.loc[db['PP_key'] == key].shape[0] == 0:
            continue
        # get the row that we want to update
        row = db.iloc[db.loc[db['PP_key'] == key].index[0]]
        # get all the rows that are similar
        # print(find_equal_rows(db, row, subset_no_key))
        for i in find_equal_rows(db, row, subset_no_key):
            # and assign the same key to all these rows
            db.at[i, "PP_key"] = key

    # clean up the placeholders that we put before doing the fix
    for col in subset_no_key:
        db[col] = db[col].apply(lambda x: np.nan if x == "placeholder" else x)
    
    # check: if we drop now but on ALL power plant columns INCLUDING PP_key, we lose as many rows as we computed before (when we excluded PP_key from the dropping)
    print(f"Done. Rows were changed correctly: {db.loc[(db['PP_key'].str.contains(pp_key_new)) & (db['power_plant_name'] != '')].drop_duplicates(subset = full_subset_columns_in_db).shape[0] == after_dropping}")

    # returns the updated db and the number of rows that are duplicated (we can use to check the dropping later on)
    return db, diff