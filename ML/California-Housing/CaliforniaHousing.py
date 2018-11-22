import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from zlib import crc32
from sklearn.model_selection import StratifiedShuffleSplit
import datetime

HOUSING_CSV_FILE = "./housing.csv"


def load_housing_data():
    return pd.read_csv(HOUSING_CSV_FILE)


def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32


def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]


housing = load_housing_data()
#print(housing.head(5))
housing.info()
print(housing.describe())
#housing.hist(bins=50, figsize=(20,15))
#plt.show()
housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing["income_cat"].where(housing["median_income"] < 5, 5.0, inplace=True)
#housing["income_cat"].hist()
#plt.show()

#train_set, test_set = split_train_test(housing, 0.2) # this will eventually result in usage of all test_set in training set as new data gets added
#housing_with_id = housing.reset_index() # adds an 'index' column
#train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

#print("train_set: " + str(len(train_set)))
#print("test_set: " + str(len(test_set)))

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

print("train_set: " + str(len(strat_train_set)))
print("test_set: " + str(len(strat_test_set)))
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))
print(strat_test_set["income_cat"].value_counts())

# now we can remove
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)


housing = strat_train_set.copy()
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10, 7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
plt.legend()
#plt.show()



## Experimenting with Attribute Combinations
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
print(corr_matrix)



## prepare the data for ML algorithm
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()



## Data Cleaning
#from sklearn.preprocessing import Imputer
from sklearn.impute import SimpleImputer as Imputer

# housing.dropna(subset=["total_bedrooms"])     # option 1
# housing.drop("total_bedrooms", axis=1)        # option 2
median = housing["total_bedrooms"].median()     # option 3
housing["total_bedrooms"].fillna(median, inplace=True)

imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1) # dropping because of text attribute
imputer.fit(housing_num)

X=imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)



## Handling Text and Categorical Attributes
from sklearn.preprocessing import CategoricalEncoder, OrdinalEncoder

#cat_encoder = CategoricalEncoder()
cat_encoder = OrdinalEncoder()
housing_cat = housing["ocean_proximity"]
housing_cat_reshaped = housing_cat.values.reshape(-1, 1)


## Custom Transformers
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)



## Transformation Pipelines
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.base import BaseEstimator, TransformerMixin

class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return  self
    def transform(self, X):
        return X[self.attribute_names].values


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]


num_pipeline = Pipeline([
    ('selector', DataFrameSelector(num_attribs)),
    ('imputer', Imputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])
housing_num_tr = num_pipeline.fit_transform(housing_num)

cat_pipeline = Pipeline([
    ('selector', DataFrameSelector(cat_attribs)),
    ('cat_encoder', OrdinalEncoder()) #CategoricalEcocder is deprecated
])


from sklearn.pipeline import FeatureUnion

full_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipeline),
    ("cat_pipeline", cat_pipeline)
])

housing_prepared = full_pipeline.fit_transform(housing)



### Select and Train a Model
## Training and Evaluation on the Training Set
from sklearn.linear_model import LinearRegression

print("\n\nLinear Regression: ")
lin_reg = LinearRegression()
t1 = datetime.datetime.now()
lin_reg.fit(housing_prepared, housing_labels)
t2 = datetime.datetime.now()
print("time-diff = ", t2 - t1)

# Testing
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions:", lin_reg.predict(some_data_prepared))
print("Labels:", list(some_labels))

from sklearn.metrics import mean_squared_error
housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
print("Error:", lin_rmse)


# Linear regression had errors, switch to DecisionTreeRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

print("\n\nDecision Tree Regression: ")
tree_reg = DecisionTreeRegressor()
t1 = datetime.datetime.now()
tree_reg.fit(housing_prepared, housing_labels)
t2 = datetime.datetime.now()
print("time-diff = ", t2 - t1)
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)

# Testing
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
print("Error:", tree_rmse)

## Better Evaluation Using Cross-Validation
scores = cross_val_score(tree_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
display_scores(tree_rmse_scores)


# Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor

print("\n\nRandom Forest Regression: ")
forest_reg = RandomForestRegressor()
t1 = datetime.datetime.now()
forest_reg.fit(housing_prepared, housing_labels)
t2 = datetime.datetime.now()
print("time-diff = ", t2 - t1)
forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
