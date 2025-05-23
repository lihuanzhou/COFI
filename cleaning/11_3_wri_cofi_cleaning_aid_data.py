 # -*- coding: utf-8 -*-
"""4 - extracting info - compute statistics.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1tQrQ0H-udas3DNhsnxal3JWcMDKXCWRe
"""

import pandas as pd

#---------THIRD PART MERGING ENERGY AND POWER PLANT AID DATA---------#

"""#### 1. Load data"""

new_rows= pd.read_excel("AidData - New Plants with extracted info - all FINAL_v2.xlsx")
original_rows = pd.read_excel("AidData - with extracted info and record ID v4_generated.xlsx")

new_rows = new_rows.loc[:, ~new_rows.columns.str.contains('^Unnamed')]
original_rows = original_rows.loc[:, ~original_rows.columns.str.contains('^Unnamed')]

new_rows.head(2)

original_rows.columns

print(f"New: {new_rows.shape[0]}")
print(f"Original: {original_rows.shape[0]}")

"""##### Add the nominal values to the original rows"""

db = pd.read_excel("AidDatasGlobalChineseDevelopmentFinanceDataset_v3.0.xlsx", sheet_name = "GCDF_3.0")

counter = 0
for i, row in original_rows.iterrows():
    aid_full = db.loc[db['Title'] == row['Title']]
    if aid_full.shape[0] > 1:
        print("lol")
        counter += 1
counter

for i, row in original_rows.iterrows():
    aid_full = db.loc[db['Title'] == row['Title']]
    original_rows.at[i, "Amount (Nominal USD)"] = aid_full['Amount (Nominal USD)'].values[0]

# check
i = -1
# print(f"Aid: {original_rows.iloc[i]['Amount (Nominal USD)']}")
# print(f"original: {db.loc[db['Title'] == original_rows.iloc[i]['Title']]["Amount (Nominal USD)"].values[0]}")

original_rows.head()

"""##### Concat"""

aid = pd.concat([new_rows, original_rows])
aid.shape[0] == new_rows.shape[0] + original_rows.shape[0]

"""Are there duplicates? That is are there rows that are shared by both the new and original rows?"""

print(f"No. of rows: {aid.shape[0]}")
for col in ["AidData Record ID", "Title"]:
    print(f"{col} no. unique: {aid[col].nunique()}")

"""So, it would seem so. But if we look at the original columns and look if there are duplicates in that subset."""

# original columns: these stems from the original AidData
original = list(aid.columns[0: 12])
original

# on the original information there is no duplicates
print(f"No. of rows: {aid.shape[0]}")
print(f"No. of unique values in original columns subset: {aid[original].drop_duplicates().shape[0]}")

aid = aid.drop_duplicates(subset=original)

aid = aid.reset_index()
aid = aid.drop(columns=["index"])

"""So, it would seem so."""

extracted_cols = ["primary_fuel", "installed_capacity", "additional_info", "city", "province"]

aid[extracted_cols].isna().sum()

for col in extracted_cols:
    aid[col] = aid[col].fillna("")

"""#### 2. Compute statistics"""

print(f"Rows extracted: {aid.shape[0]}")

rows = {}
for col in extracted_cols:
    missing_no = aid.loc[aid[col] == ""].shape[0]
    print(f"{col} missing: {missing_no}")
    print(f"{col} missing %: {missing_no / aid.shape[0] * 100}")
    print("")
    if col != "additional_info":
        rows[col] = [missing_no, missing_no / aid.shape[0] * 100]
    else:
        rows["name"] = [missing_no, missing_no / aid.shape[0] * 100]

pd.DataFrame(rows).transpose().rename(columns={0: "Missing #", 1: "Missing %"})

from itertools import combinations

combs = combinations(extracted_cols[0:3], 2)

rows_comb = []
for comb in combs:
    missing_no = aid.loc[(aid[comb[0]] == "") & (aid[comb[1]] == "")].shape[0]
    print(f"{comb[0]} & {comb[1]} missing: {missing_no}")
    print(f"{comb[0]} & {comb[1]} missing %: {missing_no / aid.shape[0] * 100}")
    print("")
    rows_comb.append([comb[0], comb[1], missing_no, missing_no / aid.shape[0] * 100])

combs = combinations(extracted_cols[3:], 2)

for comb in combs:
    missing_no = aid.loc[(aid[comb[0]] == "") & (aid[comb[1]] == "")].shape[0]
    print(f"{comb[0]} & {comb[1]} missing: {missing_no}")
    print(f"{comb[0]} & {comb[1]} missing %: {missing_no / aid.shape[0] * 100}")
    print("")
    rows_comb.append([comb[0], comb[1], missing_no, missing_no / aid.shape[0] * 100])

t = pd.DataFrame(rows_comb, columns=['Column 1', "Column 2", "Missing #", "Missing %"])
t['Column 2'] = t['Column 2'].apply(lambda x: x.replace("additional_info", "name"))
t

missing_no = aid.loc[(aid[extracted_cols[0]] == "") & (aid[extracted_cols[1]] == "") & (aid[extracted_cols[2]] == "")].shape[0]
print(f"name, installed_capacity, primary_fuel missing:")
print(f"#: {missing_no}")
print(f"%: {missing_no / aid.shape[0] * 100}\n")

missing_no = aid.loc[(aid[extracted_cols[0]] == "") & (aid[extracted_cols[1]] == "") & (aid[extracted_cols[2]] == "") & (aid[extracted_cols[-2]] == "") & (aid[extracted_cols[-1]] == "") ].shape[0]
print(f"All missing:")
print(f"#: {missing_no}")
print(f"%: {missing_no / aid.shape[0] * 100}")

"""## Save info"""

aid.shape[0]

aid.sort_values(by="AidData Record ID").to_excel("./Cleaned Data/AidData.xlsx", index=False)

#@TODO column renaming! power_plant_name and ID column mainly