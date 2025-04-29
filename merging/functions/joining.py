import pandas as pd
from functions.cleaning import print_last_modified_file_date_in_folder, my_read_excel
import numpy as np

# output_folder = "Final tables/" # where we store the final tables (e.g. Power Plant)
OUTPUT_FOLDER_DEFAULT = "Current Final Tables/"
# already exising final data
# power_plant_file = "Power Plant.xlsx"
# city_file = "City.xlsx"
# country_file = "Country.xlsx"
# bridge_file = "CITYKEY_BRIDGE_PP_C.xlsx"
# equity_file = "Equity.xlsx"
# transaction_file = "Transaction.xlsx"
# debt_file = "Debt.xlsx"
power_plant_file = "Power Plant.csv"
city_file = "City.csv"
country_file = "Country.csv"
bridge_file = "CITYKEY_BRIDGE_PP_C.csv"
equity_file = "Equity.csv"
transaction_file = "Transaction.csv"
debt_file = "Debt.csv"

def join_pp_until_country(columns_to_keep = None, new_output_folder = None, get_commissioning_year=False):
    # functions that joins the Power Plant table to the bridging table to the City table to the Country table
    # columns_to_keep: the columns that the returned table will only have
    # returns: the joined data with only the specified tables

    # set up the folder where to the retrieve the data from
    if new_output_folder != None:
        output_folder = new_output_folder
    else: # default
        output_folder = OUTPUT_FOLDER_DEFAULT

    print(f"Retriving data from folder \"{output_folder}\"....")
    print_last_modified_file_date_in_folder(output_folder)

    # Update the columns_to_keep with commissioning_year if needed
    if get_commissioning_year == True:
        # only add if it's not already in the dataset
        if "commissioning_year" not in columns_to_keep:
            columns_to_keep += ['commissioning_year']
        if "year_range" not in columns_to_keep:
            columns_to_keep += ['year_range']

    # read the data
    pp = my_read_excel(output_folder , power_plant_file, verbose=False, low_memory=False)
    bridge_pp_city = my_read_excel(output_folder , bridge_file, verbose=False)
    city = my_read_excel(output_folder , city_file, verbose=False)
    country = my_read_excel(output_folder , country_file, verbose=False)

    # remove unwanted columns
    # if "Unnamed: 0" in pp.columns:
    #     pp = pp.drop(columns=['Unnamed: 0'])
    # if "Unnamed: 0" in bridge_pp_city.columns:
    #     bridge_pp_city = bridge_pp_city.drop(columns=['Unnamed: 0'])
    # if "Unnamed: 0" in city.columns:
    #     city = city.drop(columns=['Unnamed: 0'])
    # if "Unnamed: 0" in country.columns:
    #     country = country.drop(columns=['Unnamed: 0'])

    # since I haven't updated the Step 1 with the new keys naming convention, then thre are slight discrepancies between pp and brige
    # in bridge the keys that came from WEPP-GPPD are now named "PLANTKEY_WEPPPGPPD_" whereas in PP they are still called "PLANTKEY"
    # whereas the power plants stemming from BUCGP already have the right naming conventions
    # # TODO: this step is not needed once Step 1 is updated with the same naming conventions of keys as step 2
    # for i, row in pp.iterrows():
    #     if "BUCGP" not in row['PP_key']:
    #         pp.at[i, "PP_key"] = row['PP_key'].replace("PLANTKEY", "PLANTKEY_WEPPGPPD_")

    # join pp with bridge
    pp_bridge = pp.merge(bridge_pp_city, left_on="PP_key", right_on="PP_key")
    # check that all the power plants have been preserved in the merge
    print("Joining Power Plant and briding table, all power plants preserved: " + str(pp_bridge.shape[0] == pp.shape[0]))

    # join pp_bridge with city
    pp_city = pp_bridge.merge(city, left_on="city_key", right_on="city_key")
    # check that all the power plants have been preserved in the merge
    print("Joining with city, all power plants preserved: " + str(pp_city.shape[0] == pp_bridge.shape[0]))

    # join pp_city with country to get all the data together
    pp_full = pp_city.merge(country, left_on="country", right_on="country")
    # check that all the power platns have been preserved in the merge
    print("Joining with country, all power plants preserved: " + str(pp_full.shape[0] == pp_city.shape[0]))

    # if columns_to_leep is specified by the user, then return the pp_full dataframe
    # with the specififed columns
    if columns_to_keep != None:
        return pp_full[columns_to_keep]
    # otherwise return the dataset with all the columns
    return pp_full

def join_equity_transaction(columns_to_keep=None, new_output_folder = None):
    # joins together Equity and Transaction table 

    # set up the folder where to the retrieve the data from
    if new_output_folder != None:
        output_folder = new_output_folder
    else: # default
        output_folder = OUTPUT_FOLDER_DEFAULT

    print(f"Retriving data from folder \"{output_folder}\"....")
    print_last_modified_file_date_in_folder(output_folder)
    
    # read the data
    equity = my_read_excel(output_folder , equity_file, verbose=False)
    transaction = my_read_excel(output_folder , transaction_file, verbose=False)
    
    # join
    equity_full = equity.merge(transaction, left_on="equity_id", right_on="investment_id")
    print(f"Joining Equity and Transaction, all Equity entries had matching transaction: {equity_full.shape[0] >= equity.shape[0]}")
    # Note: when joining there could be more entries in Transaction for one Equity entry, so that's why we use the ">="

    # if columns_to_leep is specified by the user, then return the equity_full dataframe
    # with the specififed columns
    if columns_to_keep != None:
        return equity_full[columns_to_keep]
    # otherwise return the dataset with all the columns
    return equity_full

def join_debt_transaction(columns_to_keep=None, new_output_folder = None):
    # joins together Debt and Transaction table 

        # set up the folder where to the retrieve the data from
    if new_output_folder != None:
        output_folder = new_output_folder
    else: # default
        output_folder = OUTPUT_FOLDER_DEFAULT

    print(f"Retriving data from folder \"{output_folder}\"....")
    print_last_modified_file_date_in_folder(output_folder)
    
    # read the data
    debt = my_read_excel(output_folder , debt_file, verbose=False)
    transaction = my_read_excel(output_folder , transaction_file, verbose=False)

    # join
    debt_full = debt.merge(transaction, left_on="debt_id", right_on="investment_id")
    print(f"Joining Debt and Transaction, all Debt entries had matching transaction: {debt_full.shape[0] >= debt.shape[0]}")
    # Note: when joining there could be more entries in Transaction for one Equity entry, so that's why we use the ">="

    # if columns_to_leep is specified by the user, then return the debt_full dataframe
    # with the specififed columns
    if columns_to_keep != None:
        return debt_full[columns_to_keep]
    # otherwise return the dataset with all the columns
    return debt_full

def join_pp_full_debt_full(columns_to_keep_pp=None, columns_to_keep_debt=None, new_output_folder = None, get_commissioning_year=False):
    # returns both pp_full and the Debt+Transaction data already joined with pp_full
    pp_full = join_pp_until_country(columns_to_keep_pp, new_output_folder, get_commissioning_year)
    print("\n")
    debt_full = join_debt_transaction(columns_to_keep_debt, new_output_folder)

    db_pp = debt_full.merge(pp_full, left_on="PP_key", right_on="PP_key")
    print(f"Joining Debt full and Power Plant full, all Debt entries are preserved: {db_pp.shape[0] == debt_full.shape[0]}")

    return pp_full, db_pp

def join_pp_full_equity_full(columns_to_keep_pp=None, columns_to_keep_equity=None, new_output_folder = None, get_commissioning_year=False):
    # returns both pp_full and the Equiy+Transaction data already joined with pp_full
    pp_full = join_pp_until_country(columns_to_keep_pp, new_output_folder, get_commissioning_year)
    print("\n")
    equity_full = join_equity_transaction(columns_to_keep_equity, new_output_folder)

    eq_pp = equity_full.merge(pp_full, left_on="PP_key", right_on="PP_key")
    print(f"Joining Equity full and Power Plant full, all Equity entries are preserved: {eq_pp.shape[0] == equity_full.shape[0]}")

    return pp_full, eq_pp

def join_pp_full_debt_full_equity_full(columns_to_keep_pp=None, columns_to_keep_debt=None, columns_to_keep_equity=None, new_output_folder = None, get_commissioning_year=False):
    # returns pp_full, Debt+Transaction data already joined with pp_full, and Equiy+Transaction data already joined with pp_full
    
    # power plant full
    pp_full = join_pp_until_country(columns_to_keep_pp, new_output_folder, get_commissioning_year)
    print("\n")

    # debt full
    debt_full = join_debt_transaction(columns_to_keep_debt, new_output_folder)
    db_pp = debt_full.merge(pp_full, left_on="PP_key", right_on="PP_key")
    print(f"Joining Debt full and Power Plant full, all Debt entries are preserved: {db_pp.shape[0] == debt_full.shape[0]}")
    print("\n")

    # equity full
    equity_full = join_equity_transaction(columns_to_keep_equity, new_output_folder)
    eq_pp = equity_full.merge(pp_full, left_on="PP_key", right_on="PP_key")
    print(f"Joining Equity full and Power Plant full, all Equity entries are preserved: {eq_pp.shape[0] == equity_full.shape[0]}")

    return pp_full, db_pp, eq_pp



####################### FUNCTIONS THAT DEAL WITH JOINING THE DATASETS BASED ON A RANGE FOR "commissioning_year" #######################

def is_close_overlap(ll1, ul1, ll2, ul2, tolerance=2):
    # Function to check if two ranges overlap within a tolerance
    return max(ll1, ll2) <= min(ul1, ul2) + tolerance

def prepare_for_joining(db):
    #@TODO check "-" corner case 
    # creates the columns in db for the joining using a range for joining the commissioning_year
    db['year_range'] = db['year_range'].fillna("")
    print(db[db["year_range"] == ""].shape[0], " missing year ranges")
    print(db["commissioning_year"].isna().sum(), " missing commissioning years")
    print(db[db["year_range"].str.contains("missing")].shape[0], " missing explict word commissioning years")
    print(db[db["commissioning_year"] == ""].shape)
   
    db['year_range'] = db.apply(lambda row: f"{row['commissioning_year']}-{row['commissioning_year']}" if row['year_range'] == "" else row['year_range'], axis=1)
    #print(db["year_range"].unique())

    db['lower_limit'] = db['year_range'].apply(lambda x: float(x.split("-")[0]) if "missing" not in x else np.nan)
    db['upper_limit'] = db['year_range'].apply(lambda x: float(x.split("-")[1]) if "missing" not in x else np.nan)

    return db