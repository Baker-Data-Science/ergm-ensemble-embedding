mkdir -p ./data

# Preprocessed Data Files
wget https://raw.githubusercontent.com/Baker-Data-Science/GMFold/main/data/published_clean_files/Published_Preprocessed_N48%20after%209th.csv -O ./data/Published_Preprocessed_N48_after_9th.csv
wget https://raw.githubusercontent.com/Baker-Data-Science/GMFold/main/data/published_clean_files/Published_Preprocessed_N48%20after%2013th.csv -O ./data/Published_Preprocessed_N48_after_13th.csv
wget https://raw.githubusercontent.com/Baker-Data-Science/GMFold/main/data/published_clean_files/Published_Preprocessed_N58%20after%2012th.csv -O ./data/Published_Preprocessed_N58_after_12th.csv
wget https://raw.githubusercontent.com/Baker-Data-Science/GMFold/main/data/published_clean_files/Published_Preprocessed_N58%20after%2016th.csv -O ./data/Published_Preprocessed_N58_after_16th.csv

# Combined data Files
#
wget https://raw.githubusercontent.com/Baker-Data-Science/GMFold/main/data/fold_published.csv -O ./data/fold_published.csv

# Featureized Data Files
aws s3 cp s3://ucla-ds/data/fold-data/df_9th.csv ./data/ --no-sign-request
aws s3 cp s3://ucla-ds/data/fold-data/df_12th.csv ./data/ --no-sign-request
aws s3 cp s3://ucla-ds/data/fold-data/df_13th.csv ./data/ --no-sign-request
aws s3 cp s3://ucla-ds/data/fold-data/df_16th.csv ./data/ --no-sign-request
