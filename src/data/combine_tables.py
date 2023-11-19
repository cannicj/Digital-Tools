import pandas as pd
#Merges an array of tables into one table, matching by ID
def combine_tables(tables):
    combined_table = tables[0]
    for i in range(1,len(tables)) :
        combined_table = pd.merge(combined_table, tables[i], on=['DATE'], suffixes=('', '_df' + str(i)))
   
    # Remove duplicate columns with identical values
    for col in combined_table.columns:
        if '_df' in col:
            prefix, suffix = col.split('_df')
            if combined_table[prefix].equals(combined_table[col]):
                combined_table = combined_table.drop(col, axis=1)
    return combined_table