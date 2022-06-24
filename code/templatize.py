import pandas as pd
from templatization import Templatizer


def templatize(df, cols, tags, out_path, append_sentence=True, separator="[SEP]"):
    """
    :param df: dataframe to templatize
    :param cols: list of columns to templatize
    :param out_path: path to output file

    :return: final templatized dataframe
    """
    # Create a templatizer object
    temp = Templatizer()
    data = []
    out_cols = [f"Template{i}" for i in range(len(cols))]

    for idx, row in df.iterrows():
        row_data = []
        main_col = cols[0]
        main_sent = row[main_col]

        mapper= {"person": {}, "place": {}, "pos": {}, "num": {}}
        main_template, mapper = temp.templatize(main_sent, mapper=mapper, tags=tags)

        if append_sentence:
            main_template = '{} {} {}'.format(main_template, separator, main_sent)
        row_data.append(main_template)

        for col in cols[1:]:
            sent = row[col]
            template, _ = temp.templatize(sent, mapper, tags)
            row_data.append(template)

        data.append(row_data)
        print(f"[INFO] {idx} SAMPLES COMPLETED!")

        if (idx+1)%100 == 0:
            df_out = pd.DataFrame(data, columns=out_cols)
            df_out.to_csv(out_path, index=False)

    df_out = pd.DataFrame(data, columns=out_cols)
    df_out.to_csv(out_path, index=False)

    return df_out


if __name__ == "__main__":
    filename = "data/QQP_raw_test.csv"
    df = pd.read_csv(filename)
    cols = ["question1", "question2"]
    out_path = "data/QQP_templatized_test.csv"
    tags = {"pos": ["ALL"], "neg": []}

    templatize(df, cols, tags, out_path, append_sentence=True)

