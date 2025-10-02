import pandas as pd
import streamlit as st
from sqlalchemy import create_engine
from sklearn.preprocessing import OrdinalEncoder
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import VotingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import plotly.express as px

db_user = "postgres"
db_pass = "cheema160"
db_host = "localhost"
db_port = "5432"
db_name = "vgsales"

engine = create_engine(f"postgresql+psycopg2://{db_user}:{db_pass}@{db_host}:{db_port}/{db_name}")
df=pd.read_sql("Select * from vgsales",engine)

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

st.set_page_config(page_title="Game Sales Predictor",layout="wide")
st.title("🎮 Game Sales Prediction App")
st.subheader("Powered by Voting Regressor")

col1,col2=st.columns(2)
with col1:
    year=st.number_input("Release Year",min_value=1980,max_value=2025,value=2000)
    platform=st.selectbox("Platform",df["platform"].unique())
    genre=st.selectbox("Genre",df["genre"].unique())
    publisher=st.selectbox("Publisher",df["publisher"].unique())
with col2:
    na=st.number_input("North America Sales",0.0,50.0,1.0)
    eu=st.number_input("Europe Sales",0.0,50.0,1.0)
    jp=st.number_input("Japan Sales",0.0,50.0,1.0)
    other=st.number_input("Other Sales",0.0,50.0,1.0)

if st.button("Predict Global Sales"):
    new_data=pd.DataFrame([[platform,genre,publisher]],columns=["platform","genre","publisher"])
    enc=od.transform(new_data)
    enc_df=pd.DataFrame(enc,columns=col)
    inp=pd.DataFrame([[year,na,eu,jp,other]],columns=["year","na_sales","eu_sales","jp_sales","other_sales"])
    final=pd.concat([inp,enc_df],axis=1)
    pred=vc.predict(final)[0]
    st.success(f"🌍 Predicted Global Sales: {pred:.2f} million units")

st.subheader("📊 Model Performance")
col3,col4,col5=st.columns(3)
with col3:
    st.metric("R² Score",f"{vc.score(x_test,y_test):.3f}")
with col4:
    st.metric("MAE",f"{mean_absolute_error(y_test,vc.predict(x_test)):.3f}")
with col5:
    st.metric("MSE",f"{mean_squared_error(y_test,vc.predict(x_test)):.3f}")

st.subheader("🔎 Data Exploration")
fig=px.scatter(df,x="year",y="global_sales",color="genre",size="na_sales",hover_data=["name","platform"])
st.plotly_chart(fig,use_container_width=True)
