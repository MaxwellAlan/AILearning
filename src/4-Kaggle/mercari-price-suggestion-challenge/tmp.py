#!/usr/bin/python3
# encoding: utf-8

import numpy as np
import pandas as pd


# # 本次比赛的评估指标是均方根对数误差（Root Mean Squared Logarithmic Error）
# def rmsle(y, y0):
#     assert len(y) == len(y0)
#     '''
#     log: 自然对数（基于e）
#     log10: 基于10的对数
#     log2:  基于2的对数
#     log1p: 基于log(1+ x)
#     '''
#     return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))


# def main():
#     train = pd.read_csv('data/4-Kaggle/mercari-price-suggestion-challenge/input/train.tsv', sep='\t')
#     test = pd.read_csv('data/4-Kaggle/mercari-price-suggestion-challenge/input/test.tsv', sep='\t')

#     # size of training and dataset
#     print(train.shape)
#     print(test.shape)

#     # different data types in the dataset: categorical (strings) and numeric
#     print(train.dtypes)

#     print(train.head())


# if __name__ == "__main__":
#     main()


#导入所需模块
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from subprocess import check_output


print(check_output(["ls", "data/4-Kaggle/mercari-price-suggestion-challenge/input/"]).decode("utf8"))

# Any results you write to the current directory are saved as output.
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.linear_model import Ridge
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mean_squared_log_error

#读文件
train = pd.read_csv('data/4-Kaggle/mercari-price-suggestion-challenge/input/train.tsv', sep='\t')
test = pd.read_csv('data/4-Kaggle/mercari-price-suggestion-challenge/input/test.tsv', sep='\t')
submiss = pd.read_csv('data/4-Kaggle/mercari-price-suggestion-challenge/input/sample_submission.csv', sep='\t')
#训练数据和测试数据一起处理
df = pd.concat([train, test], 0)
#训练数据的行数
nrow_train = train.shape[0]
#对价格进行处理
y_train = np.log1p(train['price'])
#准备测试数据的id
y_test=test['test_id']
#删除不需要处理的数据
df=df.drop(['price','test_id','train_id'],axis=1)
#对缺失值进行处理
df['category_name'] = df['category_name'].fillna('MISS').astype(str)
df['brand_name'] = df['brand_name'].fillna('missing').astype(str)
df['item_description'] = df['item_description'].fillna('No')
#数据类型处理
df['shipping'] = df['shipping'].astype(str)
df['item_condition_id'] = df['item_condition_id'].astype(str)
#文本处理
default_preprocessor = CountVectorizer().build_preprocessor()
def build_preprocessor(field):
    field_idx = list(df.columns).index(field)
    return lambda x: default_preprocessor(x[field_idx])

vectorizer = FeatureUnion([
    ('name', CountVectorizer(
        ngram_range=(1, 2),
        max_features=50000,
        preprocessor=build_preprocessor('name'))),
    ('category_name', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('category_name'))),
    ('brand_name', CountVectorizer(
        token_pattern='.+',
        preprocessor=build_preprocessor('brand_name'))),
    ('shipping', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('shipping'))),
    ('item_condition_id', CountVectorizer(
        token_pattern='\d+',
        preprocessor=build_preprocessor('item_condition_id'))),
    ('item_description', TfidfVectorizer(
        ngram_range=(1, 3),
        max_features=100000,
        preprocessor=build_preprocessor('item_description'))),
])
#传入数据集进行处理
X = vectorizer.fit_transform(df.values)
#处理后的训练数据
X_train = X[:nrow_train]
#处理后的测试数据
X_test = X[nrow_train:]

#模型
model = Ridge(
        solver='auto',
        fit_intercept=True,
        alpha=0.5,
        max_iter=100,
        normalize=False,
        tol=0.05)
#训练
model.fit(X_train, y_train)
#测试
preds = model.predict(X_test)
#保存结果
test["price"] = np.expm1(preds)
test[["test_id", "price"]].to_csv("data/4-Kaggle/mercari-price-suggestion-challenge/output/submission_ridge.csv", index = False)
