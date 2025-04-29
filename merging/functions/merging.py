import numpy as np
import statistics
import pandas as pd


def get_merged_value(var, to_update_entry, matched_row, merging_dict):
    # function that determines the merged value for a variable based
    # on the information in the merging_dict

    # check for NaN values: if there is a NaN value (in the old and new entry), 
    # return the other
    if np.isnan(to_update_entry[var]):
        val = matched_row[var]
    elif np.isnan(matched_row[var]):
        val = to_update_entry[var]
    # otherwise do the merge as requested
    elif merging_dict[var] == "max":
        val = max(to_update_entry[var], matched_row[var])
    elif merging_dict[var] == "average":
        val = statistics.mean([to_update_entry[var], matched_row[var]])

    return val

# def merge_debt_equity_tables_ij_global(data, matches, equity_or_debt, original, merging_dict):
#     # This function comes out of step 6. It was extended in Step 7 by the one below

#     # where we will contain new info
#     transaction_rows = []
#     investor_rows = []

#     if equity_or_debt == "Equity":
#         id_name = 'equity_id'
#         investor_name = "equity_investor_name"
#         amount_name = "equity_investment_amount"
#         parent_name = "parent_company_of_investor"
#     elif equity_or_debt == "Debt":
#         id_name = 'debt_id'
#         investor_name = "debt_investor_name"
#         amount_name = "debt_investment_amount"
#         parent_name = "parent_company_of_investor"
#     else:
#         print("ERROR")
#         return None

#     # remove from the data we are studying the rows that have been matched
#     rows_no = data.shape[0]
#     new_data = data.drop(list(matches.keys()))
#     print(f"First check after dropping: {new_data.shape[0] == rows_no - len(matches)}")

#     # add these columns if neeeded (they will contain new info coming from data)
#     if "fdi_id" not in original.columns:
#         original["fdi_id"] = ""
#     if "rma_id" not in original.columns:
#         original["rma_id"] = ""
#     if "j_id" not in original.columns:
#         original["j_id"] = ""

#     for matched_index in matches:
#         # useful values
#         # these are taken from original
#         to_update_index = original.loc[original[id_name] == matches[matched_index]].index.values[0]
#         to_update_entry = original.iloc[to_update_index]
#         # these are taken from data
#         matched_row = data.iloc[matched_index]

#         # add the new ids to original
#         if "fdi_id" in data.columns:
#             original.at[to_update_index, "fdi_id"] = matched_row['fdi_id']
#         if "rma_id" in data.columns:
#             original.at[to_update_index, "rma_id"] = matched_row['rma_id']
#         if "j_id" in data.columns:
#             original.at[to_update_index, "j_id"] = matched_row['j_id']

#         # merge the columns that can be merged
#         for var in merging_dict:
#             val = get_merged_value(var, to_update_entry, matched_row, merging_dict)
#             original.at[to_update_index, var] = val

#         # create the entries for Transaction and Investor table
#         transaction_rows.append([to_update_entry[id_name], matched_row[investor_name], matched_row[amount_name], "N", "N"])
#         investor_rows.append([matched_row[investor_name], matched_row[parent_name]])

#     return new_data, original, transaction_rows, investor_rows

def merge_debt_equity_tables_ij_global(data, matches, equity_or_debt, original, merging_dict):
    # TODO: this function extendes the one used in Step 6 but it was not tested on Step 6 (but the changes should not
    # impact the Step 6). This function compared to the original one has: adds "r_id" if present and then creates
    # the investor row based on whethtr there is the parent investor company info

    # this function works both for merging equity or debt data
    # TODO: the original function was tested on merging equity data in Step 6, this function was tested on Step 7
    # data: the dataset being study such as IJ Global's equity or debt data
    # matches: the matches found already
    # original: the Equity or Debt table where we want to merge the new info
    # merging_dict: how to merge those columns that have the same name

    # where we will contain new info
    transaction_rows = []
    investor_rows = []

    if equity_or_debt == "Equity":
        id_name = 'equity_id'
        investor_name = "equity_investor_name"
        amount_name = "equity_investment_amount"
        parent_name = "parent_company_of_investor"
    elif equity_or_debt == "Debt":
        id_name = 'debt_id'
        investor_name = "debt_investor_name"
        amount_name = "debt_investment_amount"
        parent_name = "parent_company_of_investor"
    else:
        print("ERROR")
        return None

    # remove from the data we are studying the rows that have been matched
    rows_no = data.shape[0]
    new_data = data.drop(list(matches.keys()))
    print(f"First check after dropping: {new_data.shape[0] == rows_no - len(matches)}")

    # add these columns if neeeded (they will contain new info coming from data)
    if "fdi_id" not in original.columns:
        original["fdi_id"] = ""
    if "rma_id" not in original.columns:
        original["rma_id"] = ""
    if "j_id" not in original.columns:
        original["j_id"] = ""
    if "r_id" not in original.columns:
        original["r_id"] = ""

    counter = 0 # to count how many times we go throught the changes
    for matched_index in matches:
        # useful values
        # these are taken from original
        to_update_index = original.loc[original[id_name] == matches[matched_index]].index.values[0]
        to_update_entry = original.iloc[to_update_index]
        # these are taken from data
        matched_row = data.iloc[matched_index]

        # add the new ids to original
        if "fdi_id" in data.columns:
            original.at[to_update_index, "fdi_id"] = "; ".join([original.iloc[to_update_index]['fdi_id'], str(matched_row['fdi_id'])])
        if "rma_id" in data.columns:
            original.at[to_update_index, "rma_id"] = "; ".join([original.iloc[to_update_index]['rma_id'], str(matched_row['rma_id'])])
        if "j_id" in data.columns:
            original.at[to_update_index, "j_id"] ="; ".join([original.iloc[to_update_index]['j_id'], str(matched_row['j_id'])])
        if "r_id" in data.columns:
            original.at[to_update_index, "r_id"] = "; ".join([original.iloc[to_update_index]['r_id'], str(matched_row['r_id'])])

        # merge the columns that can be merged
        for var in merging_dict:
            val = get_merged_value(var, to_update_entry, matched_row, merging_dict)
            original.at[to_update_index, var] = val

        # create the entries for Transaction and Investor table
        if "r_id" in data.columns:
            transaction_rows.append([to_update_entry[id_name], matched_row[investor_name], matched_row[amount_name], "N", "N", matched_row["r_id"]])
        else:
            transaction_rows.append([to_update_entry[id_name], matched_row[investor_name], matched_row[amount_name], "N", "N", np.nan])
        if parent_name in data.columns:
            investor_rows.append([matched_row[investor_name], matched_row[parent_name]])
        else:
            investor_rows.append([matched_row[investor_name], ""])
        counter += 1

    print(f"Matches elaborated: {counter}") 

    return new_data, original, transaction_rows, investor_rows