import pandas as pd

def get_tables_to_create_empty(db_structure):
    # turn this information in a more reliable format
    tables = {}
    for entity in db_structure['Name of new table'].unique():
        tables[entity] = [x.strip() for x in list(db_structure.loc[db_structure['Name of new table'] == entity]['Variables'].values)]

    # create empty dataframes
    tables_df = {}
    for table in tables:
        tables_df[table] = pd.DataFrame(columns=tables[table])

    return tables_df