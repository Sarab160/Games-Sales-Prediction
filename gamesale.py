import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.ensemble import BaggingRegressor,VotingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

db_user = "postgres"
db_pass = "cheema160"
db_host = "localhost"      # or your server IP
db_port = "5432"           # default PostgreSQL port
db_name = "vgsales"

# Create connection
engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}")

df=pd.read_sql("Select * from vgsales",engine)
print(df.head())
# print(df["platform"].unique())
# print(df["genre"].unique())
# print(df["publisher"].unique())
# print(df.info())

# sns.pairplot(data=df)
# plt.show()

x=df[["year","na_sales","eu_sales","jp_sales","other_sales"]]
y=df["global_sales"]

feature=df[["platform","genre","publisher"]]
od=OrdinalEncoder()
encode=od.fit_transform(feature)
col=od.get_feature_names_out()
encode_data=pd.DataFrame(encode,columns=col)

X=pd.concat([x,encode_data],axis=1)

x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

li=[("knr",KNeighborsRegressor(n_neighbors=5)),
    ("dtr",DecisionTreeRegressor(max_depth=10)),
    ("lr",LinearRegression())]

vc=VotingRegressor(li)
vc.fit(x_train,y_train)
print("Voting regressor score",vc.score(x_test,y_test))
print("Mean Absolute error",mean_absolute_error(y_test,vc.predict(x_test)))
print("Mean squared error",mean_squared_error(y_test,vc.predict(x_test)))

