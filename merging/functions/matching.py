import pandas as pd
import numpy as np
from thefuzz import process
from thefuzz import fuzz
from functions.cleaning import fill_column_in_rows, my_reset_index

####################### FUNCTIONS THAT SET UP THE MATCHING #######################

def get_merging_info_dict(merging_info, db_name, debt_name):
    # get the merging info
    info = merging_info.loc[(merging_info['Datasource'] == db_name) & (merging_info['Table'] == debt_name)]

    variables_merging_dict = {}
    for i, row in info.iterrows():
        variables_merging_dict[row['Variable']] = row['Method']

    return variables_merging_dict



def get_function(methods_df, db_name: str, description: str):
    # TODO: this is taken from Step 5 and is also in Step 6, so remove it from there
    method_name = methods_df.loc[(methods_df['Data'] == db_name) & (methods_df['Situation'] == description)]['Methods'].values[0]

    if method_name == "fuzz.partial_ratio":
        scorer = fuzz.partial_ratio
    elif method_name == "fuzz.token_set_ratio":
        scorer = fuzz.token_set_ratio
    else:
        # defualt method
        scorer = fuzz.partial_ratio

    return scorer

####################### SUPPORT FUNCTIONS FOR THE MATCHING #######################


def get_fuzzy_matches(potential, row, col_row, col_potential, score_cutoff, scorer, score_col):
    # computes the similarity scores in potential and removes the rows that are not above the threshold
    # The difference between this function and compute_fuzzy_matches_for_all_row is that here we remove the rows
    # that are below the threshold, whereas in the other function we keep all rows

    potential = potential.copy(deep=True)

    # exctract the information of the rows that meet the cutoff
    # (res contains also the indexes of these rows)
    res = process.extractBests(row[col_row], potential[col_potential], score_cutoff = score_cutoff, scorer=scorer, limit=potential.shape[0])
    # if there are no rows above the cutoff then we just return None
    if len(res) == 0:
        return None

    # keep the rows that are above the cutoff
    potential = potential.iloc[[key for match, score, key in res]]
    # save the scoring info, used later to rank answers
    for match, score, key in res:
        potential.at[key, "score_" + score_col] = score

    return potential

def compute_fuzzy_matches_for_all_row(potential, row, col_row, col_potential, scorer, score_col):
    # computes the similarity score for each row and returns the potential table with the score
    # the difference between this function and get_fuzzy_matches is that here all the rows in potential are kept
    # whereas in get_fuzzy_matches we do remove the results below the threshold

    potential = potential.copy(deep=True)

    # computes the score for each row and save it in the row
    potential["score_" + score_col] = potential[col_potential].apply(lambda x: scorer(row[col_row], x) if x != "" else 0)

    return potential

def get_parameter_default_or_not(dict_to_use, col_name, default):
    # checks if the col_name is in the dictionary otherwise ti returns the default
    if col_name in dict_to_use:
        return dict_to_use[col_name]
    return default

def year_within_range(year, lower_limit, upper_limit, tolerance):
    # Function to check if a year with tolerance falls within a range
    return lower_limit <= year + tolerance and upper_limit >= year - tolerance

def prepare_pp_full_for_matching_commissioning_year(pp_full):
    # prepares pp_full for a commissioning_year-based power plant matching
    pp_full['year_range'] = pp_full['year_range'].fillna("")
    pp_full['year_range'] = pp_full.apply(lambda row: f"{row['commissioning_year']}-{row['commissioning_year']}" if row['year_range'] == "" and not np.isnan(row['commissioning_year']) else row['year_range'], axis=1)
    pp_full['year_range'] = pp_full.apply(lambda row: f"missing" if np.isnan(row['commissioning_year']) and row['year_range'] == "" else row['year_range'], axis=1)

    # TODO: we need to update this part in case there is a missing commissionig year to begining with
    pp_full['lower_limit'] = pp_full['year_range'].apply(lambda x: float(x.split("-")[0]) if "missing" not in x else np.nan)
    pp_full['upper_limit'] = pp_full['year_range'].apply(lambda x: float(x.split("-")[1]) if "missing" not in x else np.nan)

    # check: the rows that have "missing" in the year_range also don't have a commissioning_year nor the limits (and they are the only ones that missing data on these colums
    check_1 = pp_full.loc[pp_full['year_range'].str.contains("missing")].shape[0] == pp_full.loc[pp_full['commissioning_year'].isna()].shape[0]
    check_2 = pp_full.loc[pp_full['year_range'].str.contains("missing")].shape[0] == pp_full.loc[pp_full['upper_limit'].isna()].shape[0]
    check_3 = pp_full.loc[pp_full['year_range'].str.contains("missing")].shape[0] == pp_full.loc[pp_full['lower_limit'].isna()].shape[0]
    print(f"Check: the new columns are correct for the data that don't have commissioning_year: {check_1 and check_2 and check_3}")

    return pp_full

### CHECKING FUNCTIONS: check if there is enough information to proceed with the matching ###

def check_if_row_needs_power_plant_matching(row, data_columns):
    # this function implements the logic behind determining if row has enough information to run the analysis of the matching for
    # the power plant and determines whether we need to increase the similarity score for the power plant name
    # because we want to still use the power plant name when we lack information regarding the fuel type
    # if there is enough information, then it returns True as the first returned value
    # if there is not enough information, then it returns False as the first returned value
    # if we need to increase the threshold for power plant name, it returns True as the second returned value, False otherwise

    higher_name_threshold = False

    # if there is "primary_fuel" as column
    if "primary_fuel" in data_columns:

        # if there is country, primary_fuel and one between capacity and power plant name
        if row['country'] != "" and row['primary_fuel'] != "" and ("power_plant_name" in data_columns and row['power_plant_name'] != ""):
            # if the capacity was lacking (but power_plant_name wasn't) then we need to have an higher threshold for power_plant_name
            if "installed_capacity" in data_columns and np.isnan(row['installed_capacity']):
                higher_name_threshold = True
            return True, higher_name_threshold
        
        # if there is country and primary_fuel is missing, then we check if there is the power plant name
        if row['country'] != "" and row['primary_fuel'] == "" and ("power_plant_name" in data_columns and row['power_plant_name'] != ""):
            # we need to have an higher threshold for power_plant_name because the fuel type is lacking
            higher_name_threshold = True
            return True, higher_name_threshold
    else:
        # if there is no primary fuel then we need to make sure that the power_plant_name is valied
        if row['country'] != "" and ("power_plant_name" in data_columns and row['power_plant_name'] != ""):
            # we need to have an higher threshold for power_plant_name because the fuel type is lacking
            higher_name_threshold = True
            return True, higher_name_threshold

    # in all the other cases there is no need to do a match
    return False, higher_name_threshold

def check_if_row_needs_investment_matching(row, investment_year_col, data_columns):
    """
    this function does three things:
        1. need_matching: it implements the logic behind determining if row has enough information to run the analysis of the matching for
            the investment (debt or equity)
        2. higher_threshold_fuzzy_string_needed: if there is enough information to do the matching, it determines whether we need to increase the similarity score
            for the most relevant column because while we do have enough information this is indeed just the very bare mininum so 
            we need to be extra careful (as of now, this happens if there is no fuel type information)
        3. most_important_column: it returns the most relevant string-based column. We need to use one string-based to further restrinct the potential results
            (we already restrict based on "country", "primary_fuel", and "equity_investment_year" but this is too little) and the potentail candidates
            are "power_plant_name", "city", and "province" with "power_plant_name" > "city" > "province" in terms of imporance.
            So this functions determines the most relevant column based on this order of imporantce and the availability of information
    It returns the answer to this three points in order: (need_matching = True or False), (higher_threshold_fuzzy_string_needed = True or False),
        (most_important_column = name of the column)
    
    """

    # we do NOT need to have an increased threshold for the most relevant column (higher_threshold_fuzzy_string_needed == False) because there is primary_fuel

    # enough information to have a match (need_matching == True): country, fuel, year, and one between city, province, and power_plant_name
    if row['country'] != "" and ("primary_fuel" in data_columns and row['primary_fuel'] != "") and not np.isnan(row[investment_year_col]):
        # deterime the most_relevant_column based on the order of importance (that is, first check the most imporant columns and go to lower importance columns)
        # and availability of information
        if ("power_plant_name" in data_columns and row['power_plant_name'] != ""):
            return True, False, "power_plant_name"
        
        if ("city" in data_columns and row['city'] != ""):
            return True, False, "city"

        if ("province" in data_columns and row['province'] != ""):
            return True, False, "province"
        
    # we DO need to have an increased threshold for the most relevant column (higher_threshold_fuzzy_string_needed == True) because there is NO primary_fuel
    if row['country'] != "" and (("primary_fuel" not in data_columns) or ("primary_fuel" in data_columns and row['primary_fuel'] == "")) and not np.isnan(row[investment_year_col]):
        # determine the most_relevant_column as above
        if ("power_plant_name" in data_columns and row['power_plant_name'] != ""):
            return True, True, "power_plant_name"
        
        if ("city" in data_columns and row['city'] != ""):
            return True, True, "city"

        if ("province" in data_columns and row['province'] != ""):
            return True, True, "province"

    # if we are here then there is not enough information to proceed
    return False, False, "None"

def check_if_row_needs_matching_fdi(row, data_columns):

    # we do NOT need to have an increased threshold for the most relevant column (higher_threshold_fuzzy_string_needed == False) because there is primary_fuel

    # enough information to have a match (need_matching == True): country, fuel, year, and one between city, province, and power_plant_name
    if row['country'] != "" and ("primary_fuel" in data_columns and row['primary_fuel'] != "") and row['equity_investor_name'] != "":
        # deterime the most_relevant_column based on the order of importance (that is, first check the most imporant columns and go to lower importance columns)
        # and availability of information
        
        if ("city" in data_columns and row['city'] != ""):
            return True, False, "city"

        if ("province" in data_columns and row['province'] != ""):
            return True, False, "province"
        
    # we DO need to have an increased threshold for the most relevant column (higher_threshold_fuzzy_string_needed == True) because there is NO primary_fuel
    if row['country'] != "" and (("primary_fuel" not in data_columns) or ("primary_fuel" in data_columns and row['primary_fuel'] == "")) and row['equity_investor_name'] != "":
        # determine the most_relevant_column as above
        
        if ("city" in data_columns and row['city'] != ""):
            return True, True, "city"

        if ("province" in data_columns and row['province'] != ""):
            return True, True, "province"

    # if we are here then there is not enough information to proceed
    return False, False, "None"



####################### FUNCTIONS THAT DO THE MATCHING #######################

### POWER PLANT ###

def find_matches_power_plant(data, pp_full,  
                            # the thresholds and methods are specified can be specified by the user and passed as a dictionary
                            thresholds_dict = {}, methods_dict = {}, 
                            # other parameters    
                            equity_or_debt=("Equity", "Debt"), parameters_dict = {},
                            verbose=True):
    # this functions finds the matches in the already exisint power plants (pp_full) and returns the original data with the PP_key
    # for those rows that had a match
    # for the rows that don't have a match the PP_key is put to an empty string
    # increase_threshold_power_plant_name: from 0 to 1, it is the percentage increase we want to see. Therefore, the threshold will be computed
        # like this: threshold_final = threshold

    data = data.copy(deep = True)

    # set up the column names based on whether we are working on debt or equity data
    # note: we also set up the default value for the investemnt vs commissioning year threhsold
    if equity_or_debt == "Equity":
        investment_year_col = 'equity_investment_year'
        threshold_commissioning_year_default = 5
    elif equity_or_debt == "Debt":
        investment_year_col = 'debt_investment_year'
        threshold_commissioning_year_default = 2
    else: # the user is not giving a correct value
        print("Error: choose between \"Equity\" or \"Debt\" for \"equity_or_debt\" parameter")
        return None

    ###### LOAD THRESHOLDS AND METHODS AND OTHER PARAMETERS
    threshold_installed_capacity = get_parameter_default_or_not(thresholds_dict, "installed_capacity", 100)
    threshold_power_plant_base = get_parameter_default_or_not(thresholds_dict, "power_plant_name", 80) 
    threshold_commissioning_year = get_parameter_default_or_not(thresholds_dict, investment_year_col, threshold_commissioning_year_default)  
    
    if verbose == True:
        print("TRHESHOLDS used:")
        print(f"threshold_installed_capacity: {threshold_installed_capacity}")
        print(f"threshold_power_plant_base: {threshold_power_plant_base}")
        print(f"threshold_commissioning_year: {threshold_commissioning_year}")


    scorer_power_plant = get_parameter_default_or_not(methods_dict, "power_plant_name", fuzz.token_set_ratio)
    scorer_city=get_parameter_default_or_not(methods_dict, "city", fuzz.token_set_ratio)
    scorer_province=get_parameter_default_or_not(methods_dict, "province", fuzz.token_set_ratio)  
   
    if verbose == True:
        print("\nSCORERS used:")
        print(f"scorer_power_plant: {scorer_power_plant}")
        print(f"scorer_city: {scorer_city}")
        print(f"scorer_province: {scorer_province}")

    increase_threshold_power_plant_name = get_parameter_default_or_not(parameters_dict, "increase_threshold_power_plant_name", 0.1)
    use_multiplier=get_parameter_default_or_not(parameters_dict, "use_multiplier", False)
    multipliers=get_parameter_default_or_not(parameters_dict,"multipliers", [2, 3, 4])
    use_commissioning_year_filtering=get_parameter_default_or_not(parameters_dict, "use_commissioning_year_filtering", True)
    use_year_range=get_parameter_default_or_not(parameters_dict, "use_year_range", True)
    
    if verbose == True:
        print("\nOTHER PARAMETERS used:")
        print(f"increase_threshold_power_plant_name: {increase_threshold_power_plant_name}")
        print(f"use_multiplier: {use_multiplier}")
        print(f"multipliers: {multipliers}")
        print(f"use_commissioning_year_filtering: {use_commissioning_year_filtering}")
        print(f"use_year_range: {use_year_range}")
        

    if verbose == True:
        print(f"\nFiltering based on the commissioning_year: {use_commissioning_year_filtering}")
        print(f"commissioning_year is in pp_full: {'commissioning_year' in pp_full.columns}")
        print(f"\nFiltering based on the year_range: {use_year_range}")
        print(f"year_range is in pp_full: {'year_range' in pp_full.columns}")

    data['PP_key'] = ""
    print("\nStarting matching....")

    # TODO: we will need a function that given the the method (in this case the scorer) set up the right overall method: fuzzy matching vs bert vs etc
    counter = 0
    for index, row in data.iterrows():
        ######################## PRE-FILTERING ########################
        # we use a function that returns True (as the first returned value) if the row has enough information to be matched, False otherwise
        # the second returned value of this function tells whether we need to increase the threshold for power_plant_name because we lack
        # other relevant information (capacity or fuel)
        # the logic of determining what has enough information and what not is implemented there

        # runs the function
        need_matching, higher_name_threshold = check_if_row_needs_power_plant_matching(row, data.columns)
        # if we don't have enough information for the matching, then we do not continue with the matching
        # (we just go to the next row)
        if not need_matching:
            continue

        # update already the power_plant_name threshold if needed
        # TODO: this snippet of code assumes that the threshold goes from 0 to 100 which is true for the fuzz scorers
        # but what if we change the scorer and they have another scale? This code should include that case (so
        # no hardcoding of the highest threshold - 100 in this case)
        if higher_name_threshold == True:
            # increase
            threshold_power_plant = threshold_power_plant_base * (1 + increase_threshold_power_plant_name)
            # cap the threshold at the highest possible value accessible
            if threshold_power_plant > 100:
                threshold_power_plant = 100
        else:
            threshold_power_plant = threshold_power_plant_base

        ######################## EXACT MATCHING ########################

        # do equal join on country and fuel
        potential = pp_full.loc[pp_full['country'] == row['country']] 

        # if there is primary_fuel, then also limit the results to those that have the same fuel type
        if "primary_fuel" in data.columns and row['primary_fuel'] != "":
            potential = potential.loc[potential['primary_fuel'] == row['primary_fuel']]

        ######################## FUZZY MATCHING ########################

        # if there is a capacity, also get the rows with installed_capacity in a range
        if "installed_capacity" in data.columns and np.isnan(row['installed_capacity']) == False:
            potentials = []

            # try the multiplier relations
            if use_multiplier == True:
                for multiplier in multipliers:
                    potentials.append(potential.loc[potential['installed_capacity'] == row['installed_capacity'] * multiplier])

            # get the ones with the range
            potentials.append(potential.loc[(potential['installed_capacity'] <= row['installed_capacity'] + threshold_installed_capacity) & (potential['installed_capacity'] >= row['installed_capacity'] - threshold_installed_capacity)])

            # concat everything together
            potential = pd.concat(potentials)
            potential = potential.drop_duplicates()
            potential = my_reset_index(potential)

            if potential.shape[0] == 0:
                continue

            # # get the ones in the range
            # potential = potential.loc[(potential['installed_capacity'] <= row['installed_capacity'] + threshold_installed_capacity) & (potential['installed_capacity'] >= row['installed_capacity'] - threshold_installed_capacity)]
            
        # these are the scores we can use to pick the best match in order of importance (see following sections)
        scores_relevance = ["score_power_plant_name",'score_city', "score_province"] # this is the default option

        # COMMISSIONING_YEAR
        # we compare the investment_year with the commissioning_year: we keep the values of the commisisoning_year that are within a range of the investment
        # we also keep those potential matches that don't have a commissioning_year
        # only filter based on the commissioning_year if required by the user
        if use_commissioning_year_filtering == True:
            if "commissioning_year" in pp_full.columns and np.isnan(row[investment_year_col]) == False:
                potentials = []

                # get the rows that don't have a commissioning years, so we can keep them after the filtering
                no_comm_year = potential.loc[potential['commissioning_year'].isna()]
                
                # get the ones with the range
                if use_year_range == True and "year_range" in pp_full.columns:
                    in_range = potential.loc[~potential['commissioning_year'].isna()].copy(deep=True)
                    in_range['year_in_range'] = in_range.apply(lambda row_p: year_within_range(row[investment_year_col], row_p['lower_limit'], row_p['upper_limit'], threshold_commissioning_year), axis=1)
                    in_range = in_range.loc[in_range['year_in_range'] == True]
                else:
                    in_range = potential.loc[(potential['commissioning_year'] <= row[investment_year_col] + threshold_commissioning_year) & (potential['commissioning_year'] >= row[investment_year_col] - threshold_commissioning_year)]
                


                # concat everything together
                potential = pd.concat([in_range, no_comm_year])

                # update the scores_relevance so that we can use it to order the results
                scores_relevance += ['commissioning_year']

        # POWER_PLANT_NAME
        # we filter out those matches that don't have name that is not enough similar
        if "power_plant_name" in data.columns and row['power_plant_name'] != "":
            # get the most likely similar power plant, 
            potential = my_reset_index(potential)
            
            potential = get_fuzzy_matches(potential, row, "power_plant_name" , "power_plant_name" , threshold_power_plant, scorer_power_plant, "power_plant_name")
            # if there are is no match then there we do not continue
            
            if potential is None:
                continue
        else:
            # remove the score for the power_plant_name from scores_relevance
            scores_relevance.remove("score_power_plant_name")

        ######################## COMPUTE SIMILARITY SCORES ########################
        # here we compute the similarity scores for the city, and province columns which along with the similarity score for
        # power plant name will help decide the best match

        # CITY
        # here, for each row we just get the similarity score without looking if it is above the threshold
        # because if we are here then there are already good matches for the power_plant_name which is the most relevant field
        if "city" in data.columns and row['city'] != "":
            potential = my_reset_index(potential)
            potential = compute_fuzzy_matches_for_all_row(potential, row, "city", "city", scorer_city, "city")
        else:
            # remove the score for the power_plant_name from scores_relevance
            scores_relevance.remove("score_city")

        # PROVINCE
        if "province" in data.columns and row['province'] != "":
            potential = my_reset_index(potential)
            potential = compute_fuzzy_matches_for_all_row(potential, row, "province", "province", scorer_province, "province")
        else:
            # remove the score for the power_plant_name from scores_relevance
            scores_relevance.remove("score_province")

        ######################## PICK BEST AND SAVE RESULTS ########################
        # if we got here then there is at least one match
        # we can pick the best match based on the scores computed above
        # then get the corresponding PP_key

        # last check
        if potential.shape[0] == 0:
            continue

        # do the ranking accordingly
        potential = potential.sort_values(by=scores_relevance, ascending=False)

        # get the one that is ranked the first
        potential = potential.head(1)

        # save the results in data
        data.at[index, "PP_key"] = potential['PP_key'].values[0]
        # take track that we found a match
        counter += 1

    print(f"Matches found: {counter}")
            
    return data


def find_matches_power_plant_or_fill_PP_key(data, pp_full, placeholder_unique_string, 
                            equity_or_debt=("Equity", "Debt"), 
                            # the thresholds and methods are specified can be specified by the user and passed as a dictionary
                            thresholds_dict = {}, methods_dict = {}, 
                            # other parameters    
                            parameters_dict = {},
                            verbose=True):
    # runs the matching for the power plant
    # it returns the original data with the "PP_key" column already filled: with the PP_key of the matched power plant
    # if a match was found, otherwise with a unique ID

    # find the matches with the already existing power plants
    data = find_matches_power_plant(data, pp_full,  
                            thresholds_dict, methods_dict,  
                            equity_or_debt, parameters_dict,
                            verbose)
    
    # create a unique PP_key for those rows that didn't have a match
    print("Filling in with new PP_key....")
    data = my_reset_index(data)
    data = fill_column_in_rows(data, "PP_key", placeholder_unique_string)

    print("Process completed.")

    return data

### INVESTMENT ###

def find_matches_equity_and_debt_complete(data, to_match, equity_or_debt=("Equity", "Debt"), 
                            # the thresholds and methods are specified can be specified by the user and passed as a dictionary
                            thresholds_dict = {}, methods_dict = {}, 
                            # other parameters
                            parameters_dict = {},
                            verbose=True):

    # Main idea: we filter out based on country and fuel and year but that is not enough, we need at least one stronger filtering
    # we do this when doing the power plant matches: we also filter out based on the power plant name!
    # so, there are three variables we can use and here I list them in their oder of importance: power_plant_name, city, and province
    # if there is no power_plant_name, then we use city as the most relevant column and so on
    # so in addition to country, fuel and year we also filter out the results based on this relevant column
    # this function is written to encompass all possible cases: so we alwasy check whether the columns are present in data and if the
    # row has valid values (invalid values: np.nan for numerical columns, empty strings for string-based columns)
    # moreover, we can also do an exact match on the equity_investment_type (by using tweaking the use_equity_investment_type boolean
    # parameter)
    
    
    counter = 0

    
    # set up the column names based on whether we are working on debt or equity data
    if equity_or_debt == "Equity":
        investor_name_col = 'equity_investor_name'
        investment_year_col = 'equity_investment_year'
        investment_id_col = 'equity_id'
    elif equity_or_debt == "Debt":
        investor_name_col = 'debt_investor_name'
        investment_year_col = 'debt_investment_year'
        investment_id_col = 'debt_id'
    else: # the user is not giving a correct value
        print("Error: choose between \"Equity\" or \"Debt\" for \"equity_or_debt\" parameter")
        return None
    
    ###### LOAD THRESHOLDS AND METHODS AND OTHER PARAMETERS
    threshold_year = get_parameter_default_or_not(thresholds_dict, investment_year_col, 3)
    threshold_city=get_parameter_default_or_not(thresholds_dict, "city", 80)
    threshold_province = get_parameter_default_or_not(thresholds_dict, "province", 80)
    threshold_power_plant = get_parameter_default_or_not(thresholds_dict, "power_plant_name", 80)
    threshold_installed_capacity = get_parameter_default_or_not(thresholds_dict, "installed_capacity", 100)

    if verbose == True:
        print("TRHESHOLDS used:")
        print(f"threshold_year: {threshold_year}")
        print(f"threshold_city: {threshold_city}")
        print(f"threshold_province: {threshold_province}")
        print(f"threshold_power_plant: {threshold_power_plant}")
        print(f"threshold_installed_capacity: {threshold_installed_capacity}")

    scorer_investor=get_parameter_default_or_not(methods_dict, investor_name_col, fuzz.token_set_ratio)
    scorer_city=get_parameter_default_or_not(methods_dict, "city", fuzz.token_set_ratio)
    scorer_province=get_parameter_default_or_not(methods_dict, "province", fuzz.token_set_ratio)
    scorer_power_plant=get_parameter_default_or_not(methods_dict, "power_plant_name", fuzz.token_set_ratio)

    if verbose == True:
        print("\nSCORERS used:")
        print(f"scorer_investor: {scorer_investor}")
        print(f"scorer_city: {scorer_city}")
        print(f"scorer_province: {scorer_province}")
        print(f"scorer_power_plant: {scorer_power_plant}")


    increase_threshold_string_based_column = get_parameter_default_or_not(parameters_dict, "increase_threshold_string_based_column", 0.1)
    use_multiplier=get_parameter_default_or_not(parameters_dict, "use_multiplier", False)
    multipliers=get_parameter_default_or_not(parameters_dict,"multipliers", [2, 3, 4])
    use_equity_investment_type = get_parameter_default_or_not(parameters_dict,"use_equity_investment_type", False)

    if verbose == True:
        print("\nOTHER PARAMETERS used:")
        print(f"increase_threshold_string_based_column: {increase_threshold_string_based_column}")
        print(f"use_multiplier: {use_multiplier}")
        print(f"multipliers: {multipliers}")
        print(f"use_equity_investment_type: {use_equity_investment_type}")

    # useful dictionary
    scorers_string_columns = {
        "city": scorer_city,
        "province": scorer_province,
        "power_plant_name": scorer_power_plant
    }

    print("\nStarting matching....")

    # where we will save the matches with this structure: matches[index of data] = equity_id of the match found in equity
    final_matches = {}
    for index, row in data.iterrows():

        ################ PRE-FILTER ###############################
        # we determine if there is enough information to do the matching (need_matching), if we need higher levels of thresholds for the string because there is no
        # primary_fuel information (higher_threshold_fuzzy_string_needed), and the string-based column that we will use to still filter out results (most_important_column which
        # can either be "power_plant_name", "city", or "province")
        need_matching, higher_threshold_fuzzy_string_needed, most_important_column = check_if_row_needs_investment_matching(row, investment_year_col, data.columns)

        if not need_matching:
            continue
        

        # get the current threshold for the most important column based on the input given by the user
        if most_important_column == "power_plant_name":
            threshold_most_important_column = threshold_power_plant
        elif most_important_column == "city":
            threshold_most_important_column = threshold_city
        elif most_important_column == "province":
            threshold_most_important_column = threshold_province
        else:
            print("ERROR! Problem in determining the most important column!")
            return None

        # update the threshold for the most important column if needed
        if higher_threshold_fuzzy_string_needed == True:
            # increase
            threshold_most_important_column = threshold_most_important_column * (1 + increase_threshold_string_based_column)
        
            # cap the threshold at the highest possible value accessible
            if threshold_most_important_column > 100:
                threshold_most_important_column = 100


        ################ EQUAL JOIN ###############################

        # do equal join on country and fuel
        potential = to_match.loc[to_match['country'] == row['country']] 
        # if we already have no match then go to the next row
        if potential.shape[0] == 0:
            continue

        # if there is primary_fuel, then also limit the results to those that have the same fuel type
        if "primary_fuel" in data.columns and row['primary_fuel'] != "":
            potential = potential.loc[potential['primary_fuel'] == row['primary_fuel']]
            # if we already have no match then go to the next row
            if potential.shape[0] == 0:
                continue

        # if the user says to use the equity_investment_type, we can do an equal join
        if use_equity_investment_type == True:
            if "equity_investment_type" in data.columns and row['equity_investment_type'] != "":
                potential = potential.loc[potential['equity_investment_type'] == row['equity_investment_type']]
                # if we already have no match then go to the next row
                if potential.shape[0] == 0:
                    continue

        ################ FUZZY JOIN ###############################

        # these are the scores we can use to pick the best match in order of importance (see following sections)
        scores_relevance = ["score_investor_name", "score_power_plant_name",'score_city', "score_province"] # this is the default option

        #### Numerical Fuzzy Match ####

        # INSTALLED_CAPACITY
        # if there is a capacity, also get the rows with installed_capacity in a range
        if "installed_capacity" in data.columns and np.isnan(row['installed_capacity']) == False:
            potentials = []

            # try the multiplier relations
            if use_multiplier == True:
                for multiplier in multipliers:
                    potentials.append(potential.loc[potential['installed_capacity'] == row['installed_capacity'] * multiplier])

            # get the ones with the range
            potentials.append(potential.loc[(potential['installed_capacity'] <= row['installed_capacity'] + threshold_installed_capacity) & (potential['installed_capacity'] >= row['installed_capacity'] - threshold_installed_capacity)])

            # concat everything together
            potential = pd.concat(potentials)
            potential = potential.drop_duplicates()
            potential = my_reset_index(potential)
            # if we have no match then go to the next row
            if potential.shape[0] == 0:
                continue

            # # get the ones in the range
            # potential = potential.loc[(potential['installed_capacity'] <= row['installed_capacity'] + threshold_installed_capacity) & (potential['installed_capacity'] >= row['installed_capacity'] - threshold_installed_capacity)]
            
        # print(f"{row[investment_year_col]}: {potential[investment_year_col].values}")
        

        # YEAR
        # only keep the rows that are a within a certain range
        potential = potential.loc[(potential[investment_year_col] <= row[investment_year_col] + threshold_year) & (potential[investment_year_col] >= row[investment_year_col] - threshold_year)]
        # save the difference in the year, used later to rank answers
        potential['score_year'] = potential[investment_year_col].apply(lambda x: abs(row[investment_year_col] - x) if np.isnan(x) == False else row[investment_year_col])
        # if we have no match then go to the next row
        if potential.shape[0] == 0:
            continue

        #### String Fuzzy Match ####

        # MOST_RELEVANT_COLUMN
        # we use the most relevant column to filter out results (whereas for the other string-based columns we just
        # use them to compute the scores and rank the results accordingly, we do not filter out based on these scores)
        if most_important_column in data.columns and row[most_important_column] != "":
            potential = my_reset_index(potential)
            potential = get_fuzzy_matches(potential, row, most_important_column,most_important_column , threshold_most_important_column, scorers_string_columns[most_important_column], most_important_column)
            # if there are is no match then there we do not continue
            if potential is None:
                continue
        else:
            # remove the score for the most_important_column from scores_relevance
            scores_relevance.remove("score_" + most_important_column) 

        ######################## COMPUTE SIMILARITY SCORES ########################
        # here we compute the similarity scores for the investor_name and for the two other string-based columns that are not the most relevant one
        # (we already used the most_relevant_column to filter out the results)
        # the three string-based columns are: power_plant_name, city, and province. If there is no column in data or the row doesn't have
        # information for that column, then we do not compute the score

        # # INVESTOR NAME
        if row[investor_name_col] != "":
            potential = my_reset_index(potential)
            potential = compute_fuzzy_matches_for_all_row(potential, row, investor_name_col, 'investor_name', scorer_investor, "investor_name")
        else:
            # remove the score for the investor_name from scores_relevance since in this case
            # we didn't compute the score and therefore we can't use it to rank the results
            scores_relevance.remove("score_investor_name")


        # LESS RELEVANT STRING-BASED COLUMNS
        # get the other columns that are not the most relevant one
        other_string_columns = ['power_plant_name', "city", "province"] # all the string-based columns
        other_string_columns.remove(most_important_column) # dynamically get the other columns that were not used

        for other_column in other_string_columns:
            # always check if the column is present and if the row has valid results
            if other_column in data.columns and row[other_column] != "":
                potential = my_reset_index(potential)
                potential = compute_fuzzy_matches_for_all_row(potential, row, other_column, other_column, scorers_string_columns[other_column], other_column)
            else:
                # remove the score for the other_column from scores_relevance as above
                scores_relevance.remove("score_" + other_column)

        ######################## PICK BEST AND SAVE RESULTS ########################
        # if we got here then there is at least one match
        # we can pick the best match based on the scores computed above
        # then get the corresponding PP_key

        # last check
        if potential is None or potential.shape[0] == 0:
            continue

        # do the ranking accordingly
        potential = potential.sort_values(by=scores_relevance, ascending=False)

        # get the one that is ranked the first
        potential = potential.head(1)

        # save the match
        final_matches[index] = potential[investment_id_col].values[0]
        counter += 1
        
    print(f"Matches no: {counter}")

    return final_matches


### FDI_MARKETS: special function that combines the power plant and investment matching ###

def find_matches_fdi(data, to_match, equity_or_debt=("Equity", "Debt"), 
                         # the thresholds and methods are specified can be specified by the user and passed as a dictionary
                        thresholds_dict = {}, methods_dict = {}, 
                        # other parameters
                        parameters_dict = {}, 
                        verbose=True):
    
    """
    This function is a custom function to find the power plant matches in FDI_Markets by also using information
    from its equity data. Therefore, for each row in fdi we proceed as follows:
    * first, it determines whether we have enough information to run the matching algorithm (at least country, investor name, and one between city and province.
        See check_if_row_needs_matching_fdi for more information on how to assess this).
    * we do an exact match on the country, primary_fuel, and equity_investment_type (this latter can be avoided by using the use_equity_investment_type value)
    * we then filter out the potential rows based on the degree of similarity is for the investor name (that is, we keep those power plants where there has
        been an investor that has invested there and this investor is enough similar to the investor in FDI_Markets) and the city or province (if the city
        information is available we use this one, otherwise we go to the province information)
    * we then compute the similarity score for all the rows using the province information (if we haven't already used to filter out the matches)
    * we then rank the results based on the similarity score of the investor name, city, and province (in that order of importance) and
        pick the best: we assign its PP_key to FDI_markets.
    This function takes into account the availabiltiy of the columns values each time it assesses a new row.
    
    """
    
    
    counter = 0 # to count how many matches there are


    # set up the column names based on whether we are working on debt or equity data
    if equity_or_debt == "Equity":
        investor_name_col = 'equity_investor_name'
        investment_year_col = 'equity_investment_year'
        investment_id_col = 'equity_id'
    elif equity_or_debt == "Debt":
        investor_name_col = 'debt_investor_name'
        investment_year_col = 'debt_investment_year'
        investment_id_col = 'debt_id'
    else: # the user is not giving a correct value
        print("Error: choose between \"Equity\" or \"Debt\" for \"equity_or_debt\" parameter")
        return None

    ###### LOAD THRESHOLDS AND METHODS AND OTHER PARAMETERS
    threshold_investor=get_parameter_default_or_not(thresholds_dict, investor_name_col, 80)
    threshold_city=get_parameter_default_or_not(thresholds_dict, "city", 80)
    threshold_province = get_parameter_default_or_not(thresholds_dict, "province", 80)

    if verbose == True:
        print("TRHESHOLDS used:")
        print(f"threshold_investor: {threshold_investor}")
        print(f"threshold_city: {threshold_city}")
        print(f"threshold_province: {threshold_province}")

    scorer_investor=get_parameter_default_or_not(methods_dict, investor_name_col, fuzz.token_set_ratio)
    scorer_city=get_parameter_default_or_not(methods_dict, "city", fuzz.token_set_ratio)
    scorer_province=get_parameter_default_or_not(methods_dict, "province", fuzz.token_set_ratio)

    if verbose == True:
        print("\nSCORERS used:")
        print(f"scorer_investor: {scorer_investor}")
        print(f"scorer_city: {scorer_city}")
        print(f"scorer_province: {scorer_province}")
    
    increase_threshold_string_based =get_parameter_default_or_not(parameters_dict, "increase_threshold_string_based", 0.1)
    threshold_installed_capacity = get_parameter_default_or_not(parameters_dict,"threshold_installed_capacity", 100)
    use_multiplier=get_parameter_default_or_not(parameters_dict,"use_multiplier",False)
    multipliers=get_parameter_default_or_not(parameters_dict,"multipliers", [2, 3, 4])
    use_equity_investment_type = get_parameter_default_or_not(parameters_dict,"use_equity_investment_type",False)

    if verbose == True:
        print("\nOTHER PARAMETERS used:")
        print(f"increase_threshold_string_based: {increase_threshold_string_based}")
        print(f"threshold_installed_capacity: {threshold_installed_capacity}")
        print(f"use_multiplier: {use_multiplier}")
        print(f"multipliers: {multipliers}")
        print(f"use_equity_investment_type: {use_equity_investment_type}")

    # useful dictionary
    scorers_string_columns = {
        "city": scorer_city,
        "province": scorer_province
    }

    print("\nStarting matching....")

    # where we will save the matches with this structure: matches[index of data] = equity_id of the match found in equity
    final_matches = {}
    for index, row in data.iterrows():

        ################ PRE-FILTER ###############################
        # we determine if there is enough information to do the matching (need_matching), if we need higher levels of thresholds for the string because there is no
        # primary_fuel information (higher_threshold_fuzzy_string_needed), and the string-based column that we will use to still filter out results (either "city" or
        # "province")
        need_matching, higher_threshold_fuzzy_string_needed, most_important_column = check_if_row_needs_matching_fdi(row, data.columns)

        # if there is not enough information, then we go to the next FDI row
        if not need_matching:
            continue
        
        # get the current threshold for the most important column based on the input given by the user
        if most_important_column == "city":
            threshold_most_important_column = threshold_city
        elif most_important_column == "province":
            threshold_most_important_column = threshold_province
        else:
            print("ERROR! Problem in determining the most important column!")
            return None

        # update the threshold for the most important column if needed
        if higher_threshold_fuzzy_string_needed == True:
            # increase
            threshold_most_important_column = threshold_most_important_column * (1 + increase_threshold_string_based)
        
            # cap the threshold at the highest possible value accessible
            if threshold_most_important_column > 100:
                threshold_most_important_column = 100

        ################ EQUAL JOIN ###############################

        # do equal join on country and fuel
        potential = to_match.loc[to_match['country'] == row['country']] 
        # if we already have no match then go to the next row
        if potential.shape[0] == 0:
            continue

        # if there is primary_fuel, then also limit the results to those that have the same fuel type
        if "primary_fuel" in data.columns and row['primary_fuel'] != "":
            potential = potential.loc[potential['primary_fuel'] == row['primary_fuel']]
            # if we already have no match then go to the next row
            if potential.shape[0] == 0:
                continue

        # if the user says to use the equity_investment_type, we can do an equal join
        if use_equity_investment_type == True:
            if "equity_investment_type" in data.columns and row['equity_investment_type'] != "":
                potential = potential.loc[potential['equity_investment_type'] == row['equity_investment_type']]
                # if we already have no match then go to the next row
                if potential.shape[0] == 0:
                    continue

        ################ FUZZY JOIN ###############################

        # these are the scores we can use to pick the best match in order of importance (see following sections)
        scores_relevance = ['score_city', "score_province", "score_investor_name"] # this is the default option

        #### Numerical Fuzzy Match ####

        # INSTALLED_CAPACITY
        # if there is a capacity, also get the rows with installed_capacity in a range
        if "installed_capacity" in data.columns and np.isnan(row['installed_capacity']) == False:
            potentials = []

            # try the multiplier relations
            if use_multiplier == True:
                for multiplier in multipliers:
                    potentials.append(potential.loc[potential['installed_capacity'] == row['installed_capacity'] * multiplier])

            # get the ones with the range
            potentials.append(potential.loc[(potential['installed_capacity'] <= row['installed_capacity'] + threshold_installed_capacity) & (potential['installed_capacity'] >= row['installed_capacity'] - threshold_installed_capacity)])

            # concat everything together
            potential = pd.concat(potentials)
            potential = potential.drop_duplicates()
            potential = my_reset_index(potential)
            # if we have no match then go to the next row
            if potential.shape[0] == 0:
                return

            # # get the ones in the range
            # potential = potential.loc[(potential['installed_capacity'] <= row['installed_capacity'] + threshold_installed_capacity) & (potential['installed_capacity'] >= row['installed_capacity'] - threshold_installed_capacity)]
            
        # print(f"{row[investment_year_col]}: {potential[investment_year_col].values}")

        #### String Fuzzy Match ####

        # INVESTOR_NAME
        # we do an hard filtering on the investors
        if investor_name_col in data.columns and row[investor_name_col] != "":
            potential = my_reset_index(potential)
            potential = get_fuzzy_matches(potential, row, investor_name_col, 'investor_name', threshold_investor, scorer_investor, "investor_name")
            # if there are is no match then there we do not continue
            if potential is None:
                continue
        else:
            # remove the score for the most_important_column from scores_relevance
            scores_relevance.remove("score_investor_name") 

        # MOST_RELEVANT_COLUMN
        # we use the most relevant column to filter out results (whereas for the other string-based column we just
        # use it to compute the scores and rank the results accordingly, we do not filter out based on these scores)

        # but first we check if potential does have information regarding the most important column.
        # if such information is lacking, then we use the other column as the most relevant (note: this only happens if the 
        # original most_important_column was "city" because we can still use "province" as a backup. If "province"
        # was the most_important_column that was because the FDI row didn't have information for "city" so we don't have a 
        # backup. In this case we do not proceed with the matching)

        other_string_columns = ["city", "province"] # all the string-based columns
        other_string_columns.remove(most_important_column) # dynamically get the other columns that were not used
        # other_string_columns contain the column that we will compute all the similarity scores but we won't
        # filter out based on these values

        # check if potential doesn't have enough information for the most_important_column: if all the
        # rows in potential have empty strings then there is no useful information
        if potential.loc[potential[most_important_column] == ""].shape[0] == potential.shape[0]:
            # if the most_important_column was "city" then we have "province" as a backup
            if most_important_column == "city":
                # update the variables needed to run the analysis
                most_important_column = "province"
                other_string_columns.remove(most_important_column) # "province" is now used to filter out the results
                scores_relevance.remove("score_city") # we are not computing the scores for "city" anymore
                threshold_most_important_column = threshold_province # change the threshold to match the "province" one given by the user
                if higher_threshold_fuzzy_string_needed == True: # update it if we need an higher threshold
                    # increase
                    threshold_most_important_column = threshold_most_important_column * (1 + increase_threshold_string_based)
                    # cap the threshold at the highest possible value accessible
                    if threshold_most_important_column > 100:
                        threshold_most_important_column = 100
            # if the most_important_column was "province" then we don't have a backup
            elif most_important_column == "province":
                # then there is no information and then we just continue
                continue
        
        # Do the filtering based on the most_important_column
        if most_important_column in data.columns and row[most_important_column] != "":
            potential = my_reset_index(potential)
            potential = get_fuzzy_matches(potential, row, most_important_column, most_important_column , threshold_most_important_column, scorers_string_columns[most_important_column], most_important_column)
            # if there are is no match then there we do not continue
            if potential is None:
                continue
        else:
            # remove the score for the most_important_column from scores_relevance
            scores_relevance.remove("score_" + most_important_column) 

        ######################## COMPUTE SIMILARITY SCORES ########################
        # here we compute the similarity scores for the other string-based column that is not the most relevant one
        # (we already used the most_relevant_column to filter out the results)
        # Note: If there is no column in data or the row doesn't have information for that column, then we do not compute the score


        # LESS RELEVANT STRING-BASED COLUMN
        # the less relevant column is in other_strings_column (note: this can be empty in case "city" used to be the most_relevant_column
        # but the we updated the most_relevant_column to be "province")
        for other_column in other_string_columns:
            # always check if the column is present and if the row has valid results (this is needed if the most_relevant_column is "province" because
            # "city" doesn't have valid information)
            if other_column in data.columns and row[other_column] != "":
                potential = my_reset_index(potential)
                potential = compute_fuzzy_matches_for_all_row(potential, row, other_column, other_column, scorers_string_columns[other_column], other_column)
            else:
                # remove the score for the other_column from scores_relevance as above
                scores_relevance.remove("score_" + other_column)

        ######################## PICK BEST AND SAVE RESULTS ########################
        # if we got here then there is at least one match
        # we can pick the best match based on the scores computed above
        # then get the corresponding PP_key

        # last check
        if potential is None or potential.shape[0] == 0:
            continue

        # if there is no score_city and no score_province then we do not return anything
        if "score_city" not in scores_relevance and "score_province" not in scores_relevance:
            # this happens if the most_relevant_column used to "city" and then we updated it to be "province"
            # but row["province"] doesn't have enough information
            continue

        # do the ranking accordingly
        potential = potential.sort_values(by=scores_relevance, ascending=False)

        # get the one that is ranked the first
        potential = potential.head(1)

        # save the match
        final_matches[index] = [potential[investment_id_col].values[0], potential["PP_key"].values[0]]
        # save the results in data
        data.at[index, "PP_key"] = potential['PP_key'].values[0]
        # take track that we found a match
        counter += 1
        
    print(f"Matches no: {counter}")

    return data, final_matches
