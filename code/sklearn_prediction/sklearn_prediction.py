# 1. load package 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

import warnings
warnings.filterwarnings('ignore')

def num_cat_splitor(X):
    s = (X.dtypes == 'object')
    object_cols = list(s[s].index)
    # object_cols # ['package', 'division', 'salary']
    num_cols = list(set(X.columns) - set(object_cols))
    # num_cols
    # ['Work_accident', 'time_spend_company', 'promotion_last_5years', 'id',
    #  'average_monthly_hours',  'last_evaluation',  'number_project']
    return num_cols, object_cols

# 特征数值筛选器
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values

# 选择前 k 个最重要的特征
def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]  

# 2. read_data
data = pd.read_csv("data/Employee_Satisfaction/train.csv")
test = pd.read_csv("data/Employee_Satisfaction/test.csv")

y = data['satisfaction_level']
X = data.drop(['satisfaction_level'], axis=1)

# 3. data_preprocess
# 数字特征、文字特征分离
num_cols, object_cols = num_cat_splitor(X)

# 4. 数据处理Pipeline
# 缺失值填充：SimpleImputer: https://blog.csdn.net/qq_43965708/article/details/115625768
# 数字特征
num_pipeline = Pipeline([
        ('selector', DataFrameSelector(num_cols)),
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
# 文字特征
cat_pipeline = Pipeline([
        ('selector', DataFrameSelector(object_cols)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])
# 组合数字和文字特征
full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])
X_prepared = full_pipeline.fit_transform(X)

# 5. 尝试不同的模型
from sklearn.ensemble import RandomForestRegressor
# forest_reg = RandomForestRegressor()
# forest_scores = cross_val_score(forest_reg,X_prepared,y,
#                                scoring='neg_mean_squared_error',cv=3)
# forest_rmse_scores = np.sqrt(-forest_scores)
# print(forest_rmse_scores)
# print(forest_rmse_scores.mean())
# print(forest_rmse_scores.std())

# 6. 参数搜索
param_grid = [
    {'n_estimators' : [3,10,30,50,80],'max_features':[2,4,6,8]},
    {'bootstrap':[False], 'n_estimators' : [3,10],'max_features':[2,3,4]},
]
forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                          scoring='neg_mean_squared_error')
grid_search.fit(X_prepared,y)

# # 最佳参数
# grid_search.best_params_
# # 最优模型
# grid_search.best_estimator_
# # 搜索结果
# cv_result = grid_search.cv_results_
# for mean_score, params in zip(cv_result['mean_test_score'], cv_result['params']):
#     print(np.sqrt(-mean_score), params)

# 7. 特征重要性筛选
# 8. 最终完整Pipeline
k = 3
feature_importances = grid_search.best_estimator_.feature_importances_

prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k)),
    ('forst_reg', RandomForestRegressor())
])

# 参数搜索
param_grid = [{
    'preparation__num_pipeline__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'feature_selection__k': list(range(5, len(feature_importances) + 1, 5)),
    'forst_reg__n_estimators' : [3,30,80],
    'forst_reg__max_features':[2,4,6,8]
}]

grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline, param_grid, cv=10,
                                scoring='neg_mean_squared_error', verbose=2, n_jobs=-1)

# 9. 训练
grid_search_prep.fit(X,y)
# grid_search_prep.best_params_
final_model = grid_search_prep.best_estimator_

# 10. 预测
y_pred_test = final_model.predict(test)
result = pd.DataFrame()
result['id'] = test['id']
result['satisfaction_level'] = y_pred_test
result.to_csv("data/Employee_Satisfaction/rf_ML_pipeline.csv",index=False)