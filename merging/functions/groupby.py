import pandas as pd
from sklearn.cluster import DBSCAN
from functions.cleaning import my_reset_index

def read_agg_info(source_name, source_columns, agg_info):
    # reads how the variables in the given dataframe need to be aggregated when doing a group by
    # returns a dictionary
    # source_name: the database of interst
    # source_columns: the columns of the database of interest. The columns that don't have a specified aggreagtion method decleared in the db_info then will get the defualt "max" method
    # agg_info: where the aggregation information is contained

    # select the data that refers to this source
    db_for_source = agg_info.loc[agg_info["Database"] == source_name]

    result = {}

    # get the aggregation method for those variables that have been specified by the user
    for i, row in db_for_source.iterrows():
        if row['Method'].strip() != "concat":
            result[row['Variable']] = row["Method"].strip()
        else: # we used to put a custom function for concat but we don't do that anymore
            # if we put the custom function, when in do_groupby we check if a column needs to be put to string because we need
            # to concat its values won't work. Therefore we put the concat function later in the do_groupby
            result[row['Variable']] = "concat"
            # result[row['Variable']] = '; '.join
    
    # for all the other variables put the defaul aggregation method
    for column in source_columns:
        if column not in result:
            result[column] = "max"

    return result

def do_groupby(groupby_db, name, db, agg_info):
    # function that runs the groupby of db according to the information stored in the groupby_db DataFrame and the agg_info dictionary
    # groupby_db: it contains the variables that needs to be used to groupby this db
    # name: the name of the db, it is used to fiter the groupby_db
    # db: the db to groupby
    # agg_info: it contains how to aggregate the other variables, it is in a dictionary format
    # returns the grouped db


    # then do the groupby accoding to the information provided
    groupby_columns = [x.strip() for x in groupby_db.loc[groupby_db["Database"] == name]['Variables'].values[0].split(",")]
    print("Groupby: " + str(groupby_columns))
    
    # remove the groupby_columns from the columns that need to be aggregated
    columns = db.columns.to_list()
    for col in groupby_columns:
        if col in columns:
                columns.remove(col)
    
    # create the aggregation info
    agg_operations = read_agg_info(name, columns, agg_info)

    # if we want to do "concat" we do need to make sure that the column to concat is full of strings!
    for col in agg_operations:
        if agg_operations[col] == 'concat':
            db[col] = db[col].astype("string")

    # add the specific concat function
    for col in agg_operations:
        if agg_operations[col] == "concat":
            agg_operations[col] = '; '.join

    print("Agg operations: " + str(agg_operations))

    # do groupby
    db_agg = db.groupby(groupby_columns).agg(agg_operations)
    db_agg = db_agg.reset_index()
    
    return db_agg


####################### FUNCTIONS FOR AGGREGATING WITH RANGES ON COMMISSIONING_YEAR ####################### 


def get_pairs_to_fix(db, loc_id_col_name):
    """ Returns in a dataframe the pairs of location_id and primary_fuel that need to be grouped with the special method (their rows have at least 2 different commissioning years).

    db: the db to study
    loc_id_col_name: the columns name that represents the location_id (e.g., "location_id", "wepp_id").

    returns: the pairs of location_id and primary_fuel that needs the special grouping.
    """
    db_testing =  db.loc[~db['commissioning_year'].str.contains("missing")].copy(deep=True)
    db_testing["commissioning_year"] = db_testing["commissioning_year"].astype("float")
    db_testing = db_testing.groupby([loc_id_col_name, 'primary_fuel']).agg({'commissioning_year': 'nunique'}).reset_index()
    pairs_to_fix = db_testing.loc[db_testing['commissioning_year'] > 1][[loc_id_col_name, "primary_fuel"]]
    return pairs_to_fix

def seperate_rows(db, pairs_to_fix, loc_id_col_name):
    # 0. create a column that combines the loc_id and the fuel so to faciliate getting the rows that need fixing
    db['full_index'] = db.apply(lambda row: row[loc_id_col_name] + " " + row['primary_fuel'], axis=1)
    pairs_to_fix['full_index'] = pairs_to_fix.apply(lambda row: row[loc_id_col_name] + " " + row['primary_fuel'], axis=1)

    # 1. get the rows that need fixing using the newly created columns
    db_special = db.loc[(db['full_index'].isin(pairs_to_fix['full_index']))].copy(deep=True)
    db_special = my_reset_index(db_special)
    print(f"Check: we got the right rows that need special fixing: {db_special['full_index'].nunique() == len(pairs_to_fix)}")

    # 1.2 remove the rows that still have a missing commissioning year
    # Note: these rows are here because they are paired with other rows that need the specail grouping (remember: when we checked if rows needed the grouping we removed first
    # the rows that had missing years)
    rows_no_fixing_needed = db_special.loc[db_special['commissioning_year'].str.contains("missing")].copy(deep=True)
    db_special = db_special.drop(rows_no_fixing_needed.index.to_list())

    # 2. get the rows that need the normal fixing. Also get the rows that we got from the previous step
    db_normal = db.loc[(~db['full_index'].isin(pairs_to_fix['full_index']))].copy(deep=True)
    db_normal = pd.concat([db_normal, rows_no_fixing_needed])
    db_normal = my_reset_index(db_normal)
    print(f"Check: all the rows have been taken: {db_special.shape[0] + db_normal.shape[0] == db.shape[0]}")
    
    return db_special, db_normal, pairs_to_fix


def do_groupby_new( db, groupby_columns, name,agg_info):
    # function that runs the groupby of db according to the information stored in the groupby_db DataFrame and the agg_info dictionary
    # compared to "do_groupby" it doesn't read the groupby columns and it doesn't use rules that are in agg_info for those
    # columns that are used to do the groupby
    # db: the db to groupby
    # groupby_columns: the columns to do the groupby on
    # name: the name of the db, it is used to fiter the groupby_db
    # agg_info: it contains how to aggregate the other variables, it is in a dictionary format
    # returns the grouped db
    
    
    columns = db.columns.to_list()
    # # remove the groupby_columns from the columns that need to be aggregated
    # for col in groupby_columns:
    #     if col in columns:
    #             columns.remove(col)

    # print(columns)
    
    # create the aggregation info
    agg_operations = read_agg_info(name, columns, agg_info)

    # if we want to do "concat" we do need to make sure that the column to concat is full of strings!
    for col in agg_operations:
        if agg_operations[col] == 'concat':
            db[col] = db[col].astype("string")

    # add the specific concat function
    for col in agg_operations:
        if agg_operations[col] == "concat":
            agg_operations[col] = '; '.join

    
    # remove the groupby_columns from agg_operations
    for col in groupby_columns:
        if col in agg_operations:
                agg_operations.pop(col)

    print("Agg operations: " + str(agg_operations))


    # do groupby
    db_agg = db.groupby(groupby_columns).agg(agg_operations)
    db_agg = db_agg.reset_index()
    
    return db_agg


def apply_dbscan_full(group, distance_years=2):
    # this function works for wepp and gppd since it uses "commissioning_year" as column
    group = group.copy(deep=True)
    if len(group) > 1:  # Only apply DBSCAN if the group has more than one element
        clustering = DBSCAN(eps=distance_years, min_samples=1)
        cluster_labels = clustering.fit_predict(group[['commissioning_year']])
        # Create a human-readable year range for each cluster
        group['year_cluster'] = cluster_labels
        group['year_range'] = group.groupby('year_cluster')['commissioning_year'].transform(lambda x: f"{x.min()}-{x.max()}")
        #group['year_range'] = group.groupby('year_cluster')['year'].transform(lambda x: f"{x.min()}")
    else:
        # not really needed
        group['year_range'] = f"{group['commissioning_year'].iloc[0]}-{group['commissioning_year'].iloc[0]}"
    return group


def do_clustering(db_special, pairs_to_fix, distance_years=2):
    # runs the clustering on db_special
    clustered_rows = []
    for i, row in pairs_to_fix.iterrows():
        fixed = apply_dbscan_full(db_special.loc[db_special['full_index'] == row['full_index']], distance_years)
        clustered_rows.append(fixed)

    db_special_clustered = pd.concat(clustered_rows)
    # check: there are the same number of rows and the same pairs to fix
    check = db_special_clustered.shape[0] == db_special_clustered.shape[0] and len(set(db_special_clustered['full_index'].unique()) & set(db_special_clustered['full_index'].unique())) == pairs_to_fix.shape[0]
    print(f"Check: all special rows were processed: { check }")

    return db_special_clustered