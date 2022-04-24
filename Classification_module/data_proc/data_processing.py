import pandas as pd
import re
import argparse


def csv_to_tsv(path, category, save_path, mode):
    data_raw_csv = pd.read_csv(path, encoding='ms949')
    data_raw_csv = data_raw_csv.drop(data_raw_csv.columns[[0, 2, 3, 5]], axis=1)

    dataframe = pd.DataFrame(data_raw_csv)
    dataframe = dataframe.sample(frac=1).reset_index(drop=True)

    if mode == 1:
        target_symbol = re.compile('[\{\}\[\]\/?,;:|\)*~`!^\-_+<>@\#$&▲▶◆◀■【】\\\=\(\'\"]')
        target_pattern = re.compile('본문 내용|TV플레이어| 동영상 뉴스|flash 오류를 우회하기 위한 함수 추가function  flash removeCallback|tt|앵커 멘트|xa0')

        for idx in dataframe.index:
            article = dataframe.loc[idx, 'Article']
            newline_removed_article = article.replace('\\n', '').replace('\\t', '').replace('\\r', '')
            symbol_removed_article = re.sub(target_symbol, ' ', newline_removed_article)
            trash_sentence_removed_article = re.sub(target_pattern, '', symbol_removed_article)
            new_article = re.sub(' +', ' ', trash_sentence_removed_article).lstrip()
            check = new_article.find("기사제공")
            new_article = new_article[:check - 1]
            end_point = new_article.rfind("다.")
            new_article = new_article[:end_point + 2]
    
            dataframe.loc[idx, 'Article'] = new_article
    
    delete_list = []
    for idx in dataframe.index:
        if dataframe.loc[idx, 'Article'].count(' ') < 30:
            delete_list.append(idx)

    dataframe = dataframe.drop(delete_list)

    target_symbol = re.compile('[◇※☆★○●◎◇◆□■△▲▽▼✓✔☑ⓒ]')

    for idx in dataframe.index:
        if dataframe.loc[idx, 'Article'].find("img tag s") != -1:
            dataframe.loc[idx, 'Article'] = dataframe.loc[idx, 'Article'].replace("img tag s", " ")
        if dataframe.loc[idx, 'Article'].find("img tag e") != -1:
            dataframe.loc[idx, 'Article'] = dataframe.loc[idx, 'Article'].replace("img tag e", " ")
        dataframe.loc[idx, 'Article'] = re.sub(target_symbol, ' ', dataframe.loc[idx, 'Article'])
    
    for idx in dataframe.index:
        dataframe.loc[idx, 'Category'] = category
    
    dataframe.to_csv(save_path, sep='\t', encoding='utf-8', index=False)


def main():
    parser = argparse.ArgumentParser(description='PATH, MODE, SAVE_PATH, Category')
    parser.add_argument('--PATH', required=True, help='Input file path')
    parser.add_argument('--MODE', type=int, required=True, help='Normal:0 Sports:1')
    parser.add_argument('--SAVE_PATH', required=True, help='Output file path')
    parser.add_argument('--CATEGORY', type=int, required=True, help='Cateogry number')

    args = parser.parse_args()

    csv_to_tsv(args.PATH, args.CATEGORY, args.SAVE_PATH, args.MODE)


if __name__ == "__main__":
    main()