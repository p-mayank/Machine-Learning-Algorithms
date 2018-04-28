''' Implementation of Decision Trees '''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import export_graphviz
import graphviz

pd.options.mode.chained_assignment = None  # default='warn'

df = pd.read_csv('bikes.csv')
features=["season","year","month","holiday","weekday","workingday","weather","temp","humidity","windspeed"]
print("Features in dataset: "+str(features))

def get_x_y(df):
    try:
        df.drop("date", axis=1, inplace=True)
    except:
        pass
    train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
    y_train = train_set["count"]
    train_set.drop("count", axis=1, inplace=True)
    X_train = train_set
    y_test = test_set["count"]
    test_set.drop("count", axis=1, inplace=True)
    X_test = test_set
    return X_train, y_train, X_test, y_test

#Training using the data
X_train, y_train, X_test, y_test = get_x_y(df)
dtr = DecisionTreeRegressor()
dtr.fit(X_train, y_train)
print("Training Complete")

#Visualization
dot_data = export_graphviz(dtr, feature_names=features, out_file=None, filled=True, rounded=True, special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render("bikes") 
print("Graph saved as bikes.pdf")

#Important Features
print("Feature Importances: ")
print(dtr.tree_.compute_feature_importances())

#(Days grouping)
feature_indexes = dtr.tree_.feature
ct=0
for k in feature_indexes:
    if(k==4):
        ct+=1
print("Number of indexes grouped with days feature: "+str(ct))

#(Total Number of Nodes)
print("Total number of nodes in the tree: "+str(dtr.tree_.node_count))

#No. of leaf nodes
leaf_boolean = dtr.tree_.children_left == -1
ct=[k for k in leaf_boolean if k]
print("Number of leaf nodes: "+str(len(ct)))

#Mean Square Error
from sklearn.metrics import mean_squared_error
y_pred = dtr.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print("Mean Square Error: "+str(mse))

############GROUPING DATA IN TERMS OF MONTH#################################
def get_df_month_wise(df, months):
    df_row_merged = df[df["month"]==months[0]]
    for i in range(1,len(months)):
        df_temp = df[df["month"]==months[i]]
        df_row_merged = pd.concat([df_row_merged, df_temp], ignore_index=True)
    return df_row_merged

#January and February
df_jan_feb = get_df_month_wise(df, [1, 2])
df_may_to_oct = get_df_month_wise(df, [5, 6, 7, 8, 9, 10])
df_mar_apr_nov_dec = get_df_month_wise(df, [3, 4, 11, 12])

#Training using data
X_train_1, y_train_1, X_test_1, y_test_1 = get_x_y(df_jan_feb)
dtr_1 = DecisionTreeRegressor()
dtr_1.fit(X_train_1, y_train_1)
print("Training Complete Ch-1")

#Visualization
dot_data = export_graphviz(dtr_1, feature_names=features, out_file=None, filled=True, rounded=True, special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render("bikes_gp1")
print("Graph saved as bikes_gp1.pdf")

#Mean Square Error
y_pred_1 = dtr_1.predict(X_test_1)
mse = mean_squared_error(y_test_1, y_pred_1)
print("Mean Square Error Ch-1: "+str(mse))

#Similar to previous group
X_train_2, y_train_2, X_test_2, y_test_2 = get_x_y(df_may_to_oct)
dtr_2 = DecisionTreeRegressor()
dtr_2.fit(X_train_2, y_train_2)
print("Training Complete Ch-2")

dot_data = export_graphviz(dtr_2, feature_names=features, out_file=None, filled=True, rounded=True, special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render("bikes_gp2")
print("Graph saved as bikes_gp2.pdf")

y_pred_2 = dtr_2.predict(X_test_2)
mse = mean_squared_error(y_test_2, y_pred_2)
print("Mean Square Error Ch-2: "+str(mse))

X_train_3, y_train_3, X_test_3, y_test_3 = get_x_y(df_mar_apr_nov_dec)
dtr_3 = DecisionTreeRegressor()
dtr_3.fit(X_train_3, y_train_3)
print("Training Complete Ch-3")

dot_data = export_graphviz(dtr_3, feature_names=features, out_file=None, filled=True, rounded=True, special_characters=True) 
graph = graphviz.Source(dot_data) 
graph.render("bikes_gp3")
print("Graph saved as bikes_gp3.pdf")

y_pred_3 = dtr_3.predict(X_test_3)
mse = mean_squared_error(y_test_3, y_pred_3)
print("Mean Square Error Ch-3: "+str(mse))