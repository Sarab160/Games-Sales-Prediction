import pandas as pd
from sqlalchemy import create_engine
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OrdinalEncoder
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error


db_user = "postgres"
db_pass = "cheema160"
db_host = "localhost"      # or your server IP
db_port = "5432"           # default PostgreSQL port
db_name = "vgsales"

# Create connection
engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}")

df=pd.read_sql("Select * from vgsales",engine)
print(df.head())
print(df["platform"].unique())
print(df["genre"].unique())
print(df["publisher"].unique())
# print(df.info())

sns.pairplot(data=df)
plt.show()


