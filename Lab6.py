#!/usr/bin/env python
# coding: utf-8

# Лабораторная работа №6 Лисин РТ5-61Б


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set(style="ticks")


data = pd.read_csv('pokemon.csv', sep = ",")


data.head()


data = data.drop('Name', 1)
data = data.drop('Number', 1)
data = data.drop('Total', 1)


data = pd.get_dummies(data)



data.head()


columns = data.columns.tolist()
column = columns.pop(columns.index("HP"))
columns.append(column)

data = data[columns]


# ## Разделим выборку

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, KFold, train_test_split
y_column = "HP"
x_columns = data.columns.tolist()
x_columns.pop(x_columns.index(y_column))

data_x_train, data_x_test, data_y_train, data_y_test = train_test_split(data[x_columns], data[y_column], test_size = 0.95, random_state = 15)

data.isnull().sum()


data.dtypes


# ## Бэггинг

from sklearn.ensemble import BaggingClassifier, BaggingRegressor


# In[12]:


bagging = BaggingRegressor(n_estimators=15, oob_score=True, random_state=10)
bagging.fit(data_x_train, data_y_train)

bg_y_pred = bagging.predict(data_x_test)


from sklearn.metrics import mean_absolute_error, mean_squared_error,  median_absolute_error, r2_score

print('Средняя абсолютная ошибка:',   mean_absolute_error(data_y_test, bg_y_pred))
print('Медианная абсолютная ошибка:',   median_absolute_error(data_y_test, bg_y_pred))
print('Среднеквадратичная ошибка:',   mean_squared_error(data_y_test, bg_y_pred, squared = False))
print('Коэффициент детерминации:',   r2_score(data_y_test, bg_y_pred))


plt.scatter(data_x_test.Attack, data_y_test,    marker = 's', label = 'Тестовая выборка')
plt.scatter(data_x_test.Attack, bg_y_pred, marker = 'o', label = 'Предсказанные данные')
plt.legend (loc = 'lower right')
plt.xlabel ('Attack')
plt.ylabel ('HP')
plt.show()


# ## Случайный лес

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

forest = RandomForestRegressor(n_estimators=15, oob_score=True, random_state=10)
forest.fit(data_x_train, data_y_train)


rf_y_pred = forest.predict(data_x_test)

print('Средняя абсолютная ошибка:',   mean_absolute_error(data_y_test, rf_y_pred))
print('Медианная абсолютная ошибка:',   median_absolute_error(data_y_test, rf_y_pred))
print('Среднеквадратичная ошибка:',   mean_squared_error(data_y_test, rf_y_pred, squared = False))
print('Коэффициент детерминации:',   r2_score(data_y_test, rf_y_pred))

plt.scatter(data_x_test.Attack, data_y_test,    marker = 's', label = 'Тестовая выборка')
plt.scatter(data_x_test.Attack, rf_y_pred, marker = 'o', label = 'Предсказанные данные')
plt.legend (loc = 'lower right')
plt.xlabel ('Attack')
plt.ylabel ('HP')
plt.show()




