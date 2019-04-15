import pandas as pd
import numpy as np
import scipy.stats as stats
from sklearn.utils import shuffle
import math

#manually select columns might be useful
cols = list(pd.read_csv("cleansed_listings_dec18.csv", nrows =1))
# print(cols)
category_features = ['room_type','cancellation_policy']
bool_features = ['host_is_superhost', 'host_identity_verified','is_location_exact','has_availability','requires_license','instant_bookable',
                 'require_guest_profile_picture','require_guest_phone_verification']
numerical_features = ['accommodates', 'bathrooms', 'bedrooms', 'beds','guests_included', 'extra_people','minimum_nights', 'maximum_nights',
                      'availability_30', 'availability_60', 'availability_90','availability_365','number_of_reviews','calculated_host_listings_count',
                      'security_deposit', 'cleaning_fee','review_scores_rating','review_scores_accuracy','review_scores_cleanliness','review_scores_checkin',
                      'review_scores_communication','review_scores_location','review_scores_value','reviews_per_month']
special_features = ['latitude', 'longitude']
target = ['price']

category_data = pd.read_csv('cleansed_listings_dec18.csv', usecols = category_features, dtype = str)
boolean_data = pd.read_csv('cleansed_listings_dec18.csv', usecols = bool_features, dtype = str)
boolean_data = boolean_data.replace({'t': True, 'f': False})
numerical_data = pd.read_csv('cleansed_listings_dec18.csv', usecols = numerical_features)
special_data = pd.read_csv('cleansed_listings_dec18.csv', usecols = special_features)
target_data = pd.read_csv('cleansed_listings_dec18.csv', usecols = target)

#only keep columns with less than 20% missing values
# threshold = 0.2
# missing_val_percentage = (category_data.isnull().sum()) / category_data.shape[0]
# category_features = missing_val_percentage[missing_val_percentage < threshold].keys().values
# category_data = category_data.loc[:,category_features]
#
# missing_val_percentage = (boolean_data.isnull().sum()) / boolean_data.shape[0]
# bool_features = missing_val_percentage[missing_val_percentage < threshold].keys().values
# boolean_data = boolean_data.loc[:,bool_features]
#
# missing_val_percentage = (numerical_data.isnull().sum()) / numerical_data.shape[0]
# numerical_features = missing_val_percentage[missing_val_percentage < threshold].keys().values
# numerical_data = numerical_data.loc[:,numerical_features]
#
# missing_val_percentage = (special_data.isnull().sum()) / special_data.shape[0]
# special_features = missing_val_percentage[missing_val_percentage < threshold].keys().values
# special_data = special_data.loc[:,special_features]

data = pd.concat([target_data,category_data,boolean_data,numerical_data,special_data], axis=1, sort=False)
# data = data.dropna(axis=0).reset_index(drop=True) #drop rows with missing values
data.to_csv('data.csv')
data.price.describe(percentiles=[0.8, 0.9, 0.95])
data = data[data['price'] <= 350].reset_index(drop=True)
data = shuffle(data).reset_index(drop=True)
print(data.shape)
########################################################################################################################
#boolean variables: apply ANOVA to find important contributers to target variable
drop_bool = []
for c in bool_features:
    F, p_value = stats.f_oneway(data['price'][data[c] == False], data['price'][data[c] == True])
    print(c,F, p_value)
    if p_value > 0.05 or np.isnan(p_value): drop_bool.append(c)

data = data.drop(columns = drop_bool)
# data = data.drop(columns = ['host_is_superhost','require_guest_phone_verification'])
# print(data.shape)
########################################################################################################################
#encode catogory data into one hot vectors and apply ANOVA to find important variables
# a = ['Melbourne' , 'Port Phillip','Yarra','Stonnington','Moreland','Yarra Ranges','Glen Eira','Darebin','Boroondara','Maribyrnong']
# data = data[data['city'].isin(a)]
# a = ['Apartment','House','Townhouse','Condominium','Serviced apartment','Villa','Guesthouse','Guest suite','Bed and breakfast','Loft','Bungalow','Cottage']
# data = data[data['property_type'].isin(a)].reset_index(drop=True)
a = ['Entire home/apt','Private room']
data = data[data['room_type'].isin(a)].reset_index(drop=True)
a = ['strict_14_with_grace_period','moderate','flexible']
data = data[data['cancellation_policy'].isin(a)].reset_index(drop=True)
# a = ['Central Business District','Southbank','St Kilda','South Yarra','Docklands','Carlton','Richmond','Brunswick','Fitzroy','Collingwood','South Melbourne']
# data = data[data['neighborhood'].isin(a)].reset_index(drop=True)
# data = data.replace({'strict_14_with_grace_period': 'strict'})
# data = data.drop(columns = ['neighborhood','room_type','cancellation_policy'])

for c in category_features:
    one_hot = pd.get_dummies(data[c])
    # print(one_hot.shape[1])
    uniq = pd.unique(data[c])
    # print(c)
    # print(data[c].value_counts())
    # print(uniq)
    # print(len(uniq))
    F, p_value = stats.f_oneway(*(data[data[c] == u]['price'] for u in uniq))
    # print(c, F, p_value)
    data = data.drop(columns=c, axis=1)
    if p_value <= 0.05:
        # print(c)
        data = data.join(one_hot)
#######################################################################################################################
#calculate correlations between numerical varaibles and target variable, to find important ones
for c in numerical_features:
    corr = np.corrcoef(data['price'], data[c])
    print(c, corr[0,1])
    if abs(corr[0,1]) < 0.4:
        data = data.drop(columns=c, axis=1)
########################################################################################################################
#encode amenities data
# amenities = data['amenities'].tolist()
# # #remove special characters
# char_list = ['{', '}', '"']
# amenities = [re.sub("|".join(char_list), "", s) for s in amenities]
# # #remove space after spliting
# amenities_tokens = []
# for amenity in amenities:
#     tokens = amenity.strip().split(',')
#     amenities_tokens.append([token.strip() for token in tokens if token.strip()!= ""])
# # #encoding amenities into one-hot vectors
# test = pd.Series(amenities_tokens)
# mlb = MultiLabelBinarizer()
# res = pd.DataFrame(mlb.fit_transform(test), columns=mlb.classes_, index=test.index)
# unique_amenities = mlb.classes_
# res = pd.concat([data['price'],res], axis=1, sort=False)
# # #find the top 5 important amenities
# amenity_importance = {}
# for c in unique_amenities:
#     F, p_value = stats.f_oneway(res['price'][res[c] == 1], res['price'][res[c] == 0])
#     # print(c,F,p_value)
#     amenity_importance[c] = p_value
# from operator import itemgetter
# topN_amenities = list(dict(sorted(amenity_importance.items(), key=itemgetter(1))[0:5]).keys())
# print('top 5 importance amenities: ',topN_amenities)
# res = res[topN_amenities]
# data = pd.concat([data,res], axis=1, sort=False).drop(['amenities'],axis = 1)
# data = data.drop(columns='amenities', axis=1)
########################################################################################################################
#training / testing spliting
# data = data.drop(columns=['latitude','longitude', 'amenities'], axis=1)
train_percentage = 0.8
num_train = math.ceil(data.shape[0] * train_percentage)

train_x = data.drop(columns = ['price'])[:num_train]
train_y = data['price'][:num_train]
test_x = data.drop(columns = ['price'])[num_train:]
test_y = data['price'][num_train:]
########################################################################################################################
# from sklearn.preprocessing import MinMaxScaler
# scaler = MinMaxScaler(feature_range=(0, 1))
# train_x_scaled = scaler.fit_transform(train_x)
# train_x_scaled = pd.DataFrame(train_x_scaled).astype(float, copy=True, errors='ignore')
# test_x_scaled = scaler.fit_transform(test_x)
# test_x_scaled = pd.DataFrame(test_x_scaled).astype(float, copy=True, errors='ignore')
########################################################################################################################
from sklearn import neighbors
for k in [3,4,5,6,7,8,9,10]:
    model = neighbors.KNeighborsRegressor(n_neighbors = k)
    model.fit(train_x,train_y)
    pred = model.predict(test_x)
    print(np.mean(abs(test_y - pred) / test_y))
    # print(sum(abs(test_y - pred)) / len(pred))

# 27.879859783301477
# 27.036328871892927
# 26.685468451242805
# 26.958253664754594
# 27.03195848128926
# 27.104445506692162
# 26.99224559167196
# 27.426195028680713
########################################################################################################################
from sklearn.linear_model import LinearRegression
reg = LinearRegression().fit(train_x, train_y)
pred = reg.predict(train_x)
print(sum(abs(train_y - pred)) / pred.shape[0])
# 29.007892060495063

from sklearn.ensemble import RandomForestRegressor
for d in [5,6,7,8,9,10]:
    clf = RandomForestRegressor(n_estimators=100, max_depth=d, random_state=0)
    clf.fit(train_x, train_y)
    pred = clf.predict(train_x)
    # print('training error: ', np.mean(abs(train_y - pred) / train_y))
    print('training error: ', sum(abs(train_y - pred)) / pred.shape[0])
    pred = clf.predict(test_x)
    # print('testing error: ', np.mean(abs(test_y - pred) / test_y))
    print('testing error: ', sum(abs(test_y - pred)) / pred.shape[0])

# training error:  26.168252106701868
# testing error:  26.321350472646873
# training error:  25.32490369964387
# testing error:  26.06416299820079
# training error:  24.417053109534766
# testing error:  25.713002484096588
# training error:  23.31644752774143
# testing error:  25.513021310111306
# training error:  22.113961080775994
# testing error:  25.38421647174648
# training error:  20.875410531979128
# testing error:  25.199990399930563
