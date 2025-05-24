import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

st.set_page_config(page_title="UK Inflation Forecast", layout="centered")
st.title("UK CPIH Inflation Forecast (with GDP Growth)")
st.markdown("Upload CPIH inflation and GDP growth data to forecast future inflation using ARIMAX.")

cpih_file = st.file_uploader("Upload CPIH CSV file", type=["csv"])
gdp_file = st.file_uploader("Upload GDP Growth CSV file", type=["csv"])

if cpih_file and gdp_file:
    try:
        # --- Load CPIH ---
        df_cpih = pd.read_csv(cpih_file, skiprows=6)
        df_cpih.columns = ['Year', 'CPIH']
        df_cpih = df_cpih[pd.to_numeric(df_cpih['Year'], errors='coerce').notna()]
        df_cpih['Year'] = df_cpih['Year'].astype(int)
        df_cpih['CPIH'] = pd.to_numeric(df_cpih['CPIH'], errors='coerce')
        df_cpih = df_cpih.dropna().set_index('Year').sort_index()

        # --- Load GDP ---
        df_gdp = pd.read_csv(gdp_file)
        df_gdp.columns = ['Year', 'GDP_Growth']
        df_gdp['Year'] = df_gdp['Year'].astype(int)
        df_gdp['GDP_Growth'] = pd.to_numeric(df_gdp['GDP_Growth'], errors='coerce')
        df_gdp = df_gdp.dropna().set_index('Year').sort_index()

        # --- Merge datasets ---
        df = df_cpih.join(df_gdp, how='inner')

        st.subheader("Historical Data")
        st.line_chart(df)

        # Forecast horizon
        n_years = st.slider("Forecast horizon (years)", 1, 10, 5)

        # Manual ARIMA order for CPIH
        p = st.number_input("CPIH AR term (p)", min_value=0, max_value=5, value=1)
        d = st.number_input("CPIH Differencing (d)", min_value=0, max_value=2, value=1)
        q = st.number_input("CPIH MA term (q)", min_value=0, max_value=5, value=1)

        # Manual ARIMA order for GDP
        pg = st.number_input("GDP AR term (p)", min_value=0, max_value=5, value=1, key='gdp_p')
        dg = st.number_input("GDP Differencing (d)", min_value=0, max_value=2, value=1, key='gdp_d')
        qg = st.number_input("GDP MA term (q)", min_value=0, max_value=5, value=1, key='gdp_q')

        with st.spinner("Forecasting GDP growth..."):
            gdp_model = ARIMA(df['GDP_Growth'], order=(pg, dg, qg))
            gdp_fit = gdp_model.fit()
            gdp_forecast = gdp_fit.forecast(steps=n_years)
            future_exog = pd.DataFrame({'GDP_Growth': gdp_forecast})

        with st.spinner("Fitting ARIMAX model for CPIH..."):
            cpih_model = SARIMAX(df['CPIH'], exog=df[['GDP_Growth']], order=(p, d, q),
                                 enforce_stationarity=False, enforce_invertibility=False)
            cpih_fit = cpih_model.fit(disp=False)
            forecast_res = cpih_fit.get_forecast(steps=n_years, exog=future_exog)
            forecast = forecast_res.predicted_mean
            conf_int = forecast_res.conf_int()
            forecast_years = list(range(df.index[-1] + 1, df.index[-1] + 1 + n_years))

        # Plot forecast
        st.subheader("CPIH Forecast")
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df.index, df['CPIH'], label='Historical CPIH', marker='o')
        ax.plot(forecast_years, forecast, label='Forecast CPIH', linestyle='--', marker='x')
        ax.fill_between(forecast_years, conf_int.iloc[:, 0], conf_int.iloc[:, 1], alpha=0.3, label='95% CI')
        ax.set_title("CPIH Forecast with GDP Growth (ARIMAX)")
        ax.set_xlabel("Year")
        ax.set_ylabel("CPIH (%)")
        ax.legend()
        ax.grid(True)
        st.pyplot(fig)

        # Forecast table
        st.subheader("Forecast Table")
        forecast_df = pd.DataFrame({
            "Forecast (%)": forecast,
            "Lower 95% CI": conf_int.iloc[:, 0],
            "Upper 95% CI": conf_int.iloc[:, 1]
        }, index=forecast_years)
        st.dataframe(forecast_df.style.format("{:.2f}"))

    except Exception as e:
        st.error(f"Error: {e}")
