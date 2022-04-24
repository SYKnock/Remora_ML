import pandas as pd
import sys


# +
def main(files):

    main_df = pd.DataFrame(index=range(0, 0), columns=['Category', 'Article'])
    for file in files:
        data_tsv = pd.read_csv(file, encoding='utf-8', sep='\t')
        dataframe_tsv = pd.DataFrame(data_tsv)
        main_df = pd.concat([main_df, dataframe_tsv], ignore_index=True)
    
    delete_list = []
    for idx in main_df.index:
        check = main_df.loc[idx, 'Category']
        if check != 0 and check != 1 and check != 2 and check != 3 and check != 4 and check != 5:
            delete_list.append(idx)
    
    main_df = main_df.drop(delete_list)

    main_df = main_df.sample(frac=1).reset_index(drop=True)

    output_name = "KOR_NEWS_DATA.tsv"
    main_df.to_csv(output_name, sep='\t', encoding="utf-8", index=False)


if __name__ == "__main__":
    main(sys.argv)
