import streamlit as st
import pandas as pd
import warnings
from etna.datasets.tsdataset import TSDataset
from etna.pipeline import Pipeline
from etna.transforms import LagTransform, LogTransform, DifferencingTransform
from etna.models import CatBoostPerSegmentModel

warnings.filterwarnings(action="ignore", message="Torchmetrics v0.9")


# Page configuration
st.set_page_config(
    page_title='Simple Prediction App',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Title of the App
st.title('Simple Prediction App')

# Loading dataset
original_df = pd.read_csv('https://raw.githubusercontent.com/demidovakatya/mashinnoye-obucheniye/master/5-data'
                          '-analysis-applications/1_1_time_series/monthly-australian-wine-sales.csv')
# original_df = pd.read_csv("data/monthly-australian-wine-sales.csv")

original_df["timestamp"] = pd.to_datetime(original_df["month"])
original_df["target"] = original_df["sales"]
original_df.drop(columns=["month", "sales"], inplace=True)
original_df["segment"] = "main"

df = TSDataset.to_dataset(original_df)
ts = TSDataset(df, freq="MS")
st.write(ts.describe())


# Input widjects
st.sidebar.subheader('Input features')


# Creating checkbox for transforms
transforms = []
HORIZON = 8
if st.sidebar.checkbox('LogTransform'):
    log = LogTransform(in_column="target")
    transforms.append(log)

if st.sidebar.checkbox('DifferencingTransform'):
    period = st.sidebar.slider('choose period', 1, 24, value=8)
    dif = DifferencingTransform(in_column="target", period=period)
    transforms.append(dif)

if st.sidebar.checkbox('select Horizon'):
    HORIZON = st.sidebar.slider('horizon of prediction', 1, 24, value=8)

start_lag, end_lag = st.sidebar.select_slider(
    'lags',
    options=list(range(1, 25, 1)),
    value=(1, 24))
log = LagTransform(in_column="target", lags=list(range(start_lag, end_lag, 1)))
transforms.append(log)


# Data split
last_month = pd.to_datetime(original_df['timestamp'].max())
train_end = (last_month - pd.DateOffset(months=HORIZON)).strftime('%Y-%m-%d')
test_start = (last_month - pd.DateOffset(months=HORIZON-1)).strftime('%Y-%m-%d')

train_ts, test_ts = ts.train_test_split(
    train_start="1980-01-01",
    train_end=train_end,
    test_start=test_start,
    test_end=last_month.strftime('%Y-%m-%d'),
)

# Model

model = Pipeline(
    model=CatBoostPerSegmentModel(),
    transforms=transforms,
    horizon=HORIZON,
)

model.fit(train_ts)
forecast_ts = model.forecast()


d1 = pd.DataFrame({
    'forecast': map(int, forecast_ts.df['main']['target']),
    'test': test_ts.df['main']['target']
})
d2 = pd.DataFrame({'train': train_ts.df['main']['target']})

d = d1.merge(d2, on='timestamp', how='outer')

st.line_chart(d, height=800)

