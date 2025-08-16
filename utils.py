import pandas as pd
import numpy as np


def remove_subsequences(df):
    sequences = df['Sequence'].tolist()
    indices_to_drop = []
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            if sequences[i] in sequences[j]:
                indices_to_drop.append(i)
                indices_to_drop.append(j)
                break  # No need to check against other sequences if it's a substring of one
            elif sequences[j] in sequences[i]:
                indices_to_drop.append(j)
                indices_to_drop.append(i)
                break  # No need to check against other sequences if it's a substring of one

    # remove duplicates from indices_to_drop
    indices_to_drop = list(set(indices_to_drop))
    # check that max indices_to_drop is less than length of df
    print(max(indices_to_drop) >= len(df))
    df = df.drop(indices_to_drop).reset_index(drop=True)
    return df

def merge_and_process(df1, df2, how='outer', clean=False):
    merged = pd.merge(df1, df2, on='Sequence', how=how)
    merged['Count_y'] = merged['Count_y'].fillna(0)
    merged['Count_x'] = merged['Count_x'].fillna(0)

    if clean:
        # REMOVE ZOMBIES
        merged = merged[merged['Count_x'] > 0]
        merged.reset_index(drop=True, inplace=True)

        # REMOVE FRANKENSTEINS
        merged = remove_subsequences(merged)

    merged['Enrichment'] = np.log2((merged['Count_y'] + 1) / (merged['Count_x'] + 1))
    merged['CPM'] = np.log2((merged['Count_y'] + 1) )# / sum(merged['Count_y']) * 1e6)

    return merged, sum(merged['Count_y']) * 1e6

# This merges simple dataframes without features like Neighborhood_vecs or Boltz_Distrib
def get_libs(df_9th, df_12th, df_13th, df_16th, clean=False):

    lib1_merged, count1 = merge_and_process(df_9th, df_13th, clean=clean)
    lib2_merged, count2 = merge_and_process(df_12th, df_16th, clean=clean)


    if clean:
        print(f"lib1_merged Count_x == 0: {lib1_merged['Count_x'].eq(0).sum()}")
        print(f"lib2_merged Count_x == 0: {lib2_merged['Count_x'].eq(0).sum()}")

    # Determine the overlap between the two merged DataFrames
    lib1_overlap = lib1_merged['Sequence'].isin(lib2_merged['Sequence'])
    lib2_overlap = lib2_merged['Sequence'].isin(lib1_merged['Sequence'])

    sorted_9_13_overlap = lib1_merged[lib1_overlap].sort_values('Sequence').reset_index(drop=False)
    sorted_12_16_overlap = lib2_merged[lib2_overlap].sort_values('Sequence').reset_index(drop=False)

    # Calculate the average enrichment for overlapping sequences
    new_enrichment = (count1 * sorted_9_13_overlap['Enrichment'] + count2 * sorted_12_16_overlap['Enrichment']) / (count1 + count2)
    new_cpm = (count1 * sorted_9_13_overlap['CPM'] + count2 * sorted_12_16_overlap['CPM']) / (count1 + count2)

    sorted_9_13_overlap['Enrichment'] = new_enrichment
    sorted_12_16_overlap['Enrichment'] = new_enrichment
    sorted_9_13_overlap['CPM'] = new_cpm
    sorted_12_16_overlap['CPM'] = new_cpm

    # Put the average enrichment back into the original DataFrames
    sorted_9_13_overlap = sorted_9_13_overlap.set_index('index')
    sorted_12_16_overlap = sorted_12_16_overlap.set_index('index')

    lib1_merged.loc[lib1_overlap,'Enrichment'] = sorted_9_13_overlap['Enrichment']
    lib2_merged.loc[lib2_overlap,'Enrichment'] = sorted_12_16_overlap['Enrichment']
    lib1_merged.loc[lib1_overlap,'CPM'] = sorted_9_13_overlap['CPM']
    lib2_merged.loc[lib2_overlap,'CPM'] = sorted_12_16_overlap['CPM']


    df_all = pd.concat([lib1_merged, lib2_merged[~lib2_overlap]], ignore_index=True)
    df_all = df_all.groupby("Sequence", as_index=False)[['CPM', 'Enrichment']].agg('max')

    # are there duplicates
    print(f"df_all duplicate sequences: {df_all['Sequence'].duplicated().sum()}")

    return lib1_merged, lib2_merged, df_all, lib1_overlap, lib2_overlap




def parse_data_frame(df):
  s = "defaultdict(<class 'int'>, {('A','C','G','X','X'): 1, …})"
  safe_globals = {
      '__builtins__': None,
  }
  safe_locals = {
      'defaultdict': defaultdict,
      'int': int,
      'np': np,            # if you have numpy types in there
  }
  def parse_defaultdict_repr(s: str) -> defaultdict:
      # 1. If it’s wrapped in [ … ] (a list), strip that off:
      if s.startswith('[') and s.endswith(']'):
          s = s[1:-1]
      # 2. Replace `<class 'int'>` → `int`
      s = re.sub(r"<class\s+'int'>", "int", s)
      # 3. Now safely eval with only the names you allow:
      safe_globals = {'__builtins__': None}
      safe_locals  = {'defaultdict': defaultdict, 'int': int}
      return eval(s, safe_globals, safe_locals)

  # Now eval the whole thing
  safe_globals = {"__builtins__": None, "np": np}

  for i in tqdm(range(len(df))):
      df.at[i,'Neighborhood_vecs'] = parse_defaultdict_repr(df['Neighborhood_vecs'][i])
      df.at[i,'Boltz_Distrib'] = eval(df['Boltz_Distrib'][i], safe_globals, {})
  return df

