import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from zlib import crc32
from sklearn.model_selection import StratifiedShuffleSplit

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
from sklearn.preprocessing import Imputer

# housing.dropna(subset=["total_bedrooms"])     # option 1
# housing.drop("total_bedrooms", axis=1)        # option 2
median = housing["total_bedrooms"].median()     # option 3
housing["total_bedrooms"].fillna(median, inplace=True)

imputer = Imputer(strategy="median")
housing_num = housing.drop("ocean_proximity", axis=1) # dropping because of text attribute
imputer.fit(housing_num)

X=imputer.transform(housing_num)
housing_tr = pd.DataFrame(X, columns=housing_num.columns)



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

