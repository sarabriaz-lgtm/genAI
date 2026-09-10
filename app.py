import streamlit as st
import pandas as pd
import numpy as np
import requests
import matplotlib.pyplot as plt
from io import BytesIO


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI Dam Water Availability",
    page_icon="💧",
    layout="wide"
)


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_scs_cn_runoff(rainfall_mm, cn):
    """
    Calculate direct runoff using the SCS Curve Number method.

    Q = (P - 0.2S)^2 / (P + 0.8S)

    for P > 0.2S

    Parameters
    ----------
    rainfall_mm : array-like
        Daily rainfall in mm
    cn : float
        Curve Number

    Returns
    -------
    runoff_mm : numpy array
        Daily direct runoff in mm
    """

    if cn <= 0 or cn > 100:
        raise ValueError("Curve Number must be between 1 and 100.")

    S = (25400 / cn) - 254
    Ia = 0.2 * S

    rainfall = np.asarray(rainfall_mm, dtype=float)

    runoff = np.where(
        rainfall > Ia,
        ((rainfall - Ia) ** 2) /
        (rainfall + 0.8 * S),
        0
    )

    return runoff


def get_daily_rainfall(
    latitude,
    longitude,
    start_date,
    end_date
):
    """
    Download daily precipitation from Open-Meteo.
    """

    url = "https://archive-api.open-meteo.com/v1/archive"

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "daily": "precipitation_sum",
        "timezone": "auto"
    }

    response = requests.get(
        url,
        params=params,
        timeout=120
    )

    if response.status_code != 200:
        raise Exception(
            f"Rainfall API error: {response.status_code}\n"
            f"{response.text}"
        )

    data = response.json()

    if "daily" not in data:
        raise Exception(
            "No daily rainfall data were returned."
        )

    df = pd.DataFrame({
        "date": data["daily"]["time"],
        "rainfall_mm": data["daily"]["precipitation_sum"]
    })

    df["date"] = pd.to_datetime(df["date"])

    df["rainfall_mm"] = pd.to_numeric(
        df["rainfall_mm"],
        errors="coerce"
    ).fillna(0)

    return df


def calculate_water_availability(
    rainfall_df,
    catchment_area_km2,
    cn
):

    df = rainfall_df.copy()

    # --------------------------------------------------------
    # SCS-CN
    # --------------------------------------------------------

    S = (25400 / cn) - 254
    Ia = 0.2 * S

    df["runoff_mm"] = calculate_scs_cn_runoff(
        df["rainfall_mm"].values,
        cn
    )

    # --------------------------------------------------------
    # Convert runoff depth to volume
    #
    # 1 mm over 1 km² = 1,000 m³
    # --------------------------------------------------------

    df["runoff_m3"] = (
        df["runoff_mm"]
        * catchment_area_km2
        * 1000
    )

    # Acre-feet
    df["runoff_acft"] = (
        df["runoff_m3"] / 1233.48184
    )

    # --------------------------------------------------------
    # Date information
    # --------------------------------------------------------

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # --------------------------------------------------------
    # Annual results
    # --------------------------------------------------------

    annual = df.groupby("year").agg(
        annual_rainfall_mm=("rainfall_mm", "sum"),
        annual_runoff_mm=("runoff_mm", "sum"),
        annual_runoff_m3=("runoff_m3", "sum"),
        annual_runoff_acft=("runoff_acft", "sum")
    ).reset_index()

    annual["annual_runoff_MAF"] = (
        annual["annual_runoff_acft"] / 1_000_000
    )

    # --------------------------------------------------------
    # Monthly results
    # --------------------------------------------------------

    monthly = df.groupby(
        ["year", "month"]
    ).agg(
        rainfall_mm=("rainfall_mm", "sum"),
        runoff_mm=("runoff_mm", "sum"),
        runoff_m3=("runoff_m3", "sum"),
        runoff_acft=("runoff_acft", "sum")
    ).reset_index()

    monthly_average = monthly.groupby("month").agg(
        average_rainfall_mm=("rainfall_mm", "mean"),
        average_runoff_mm=("runoff_mm", "mean"),
        average_runoff_m3=("runoff_m3", "mean"),
        average_runoff_acft=("runoff_acft", "mean")
    ).reset_index()

    month_names = {
        1: "January",
        2: "February",
        3: "March",
        4: "April",
        5: "May",
        6: "June",
        7: "July",
        8: "August",
        9: "September",
        10: "October",
        11: "November",
        12: "December"
    }

    monthly_average["month_name"] = (
        monthly_average["month"].map(month_names)
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    mean_annual_rainfall = (
        annual["annual_rainfall_mm"].mean()
    )

    mean_annual_runoff = (
        annual["annual_runoff_mm"].mean()
    )

    mean_annual_volume_m3 = (
        annual["annual_runoff_m3"].mean()
    )

    mean_annual_volume_acft = (
        annual["annual_runoff_acft"].mean()
    )

    mean_annual_volume_MAF = (
        annual["annual_runoff_MAF"].mean()
    )

    min_row = annual.loc[
        annual["annual_runoff_m3"].idxmin()
    ]

    max_row = annual.loc[
        annual["annual_runoff_m3"].idxmax()
    ]

    median_runoff = annual[
        "annual_runoff_m3"
    ].median()

    normal_row = annual.loc[
        (
            annual["annual_runoff_m3"]
            - median_runoff
        ).abs().idxmin()
    ]

    summary = {
        "S": S,
        "Ia": Ia,
        "mean_annual_rainfall": mean_annual_rainfall,
        "mean_annual_runoff": mean_annual_runoff,
        "mean_annual_volume_m3": mean_annual_volume_m3,
        "mean_annual_volume_acft": mean_annual_volume_acft,
        "mean_annual_volume_MAF": mean_annual_volume_MAF,
        "minimum_volume_m3": annual["annual_runoff_m3"].min(),
        "maximum_volume_m3": annual["annual_runoff_m3"].max(),
        "dry_year": int(min_row["year"]),
        "dry_year_volume": min_row["annual_runoff_m3"],
        "normal_year": int(normal_row["year"]),
        "normal_year_volume": normal_row["annual_runoff_m3"],
        "wet_year": int(max_row["year"]),
        "wet_year_volume": max_row["annual_runoff_m3"]
    }

    return (
        df,
        annual,
        monthly,
        monthly_average,
        summary
    )


def create_excel(
    daily,
    annual,
    monthly,
    monthly_average
):

    output = BytesIO()

    with pd.ExcelWriter(
        output,
        engine="openpyxl"
    ) as writer:

        daily.to_excel(
            writer,
            sheet_name="Daily Simulation",
            index=False
        )

        monthly.to_excel(
            writer,
            sheet_name="Monthly Results",
            index=False
        )

        annual.to_excel(
            writer,
            sheet_name="Annual Results",
            index=False
        )

        monthly_average.to_excel(
            writer,
            sheet_name="Average Monthly",
            index=False
        )

    output.seek(0)

    return output


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("💧 Dam Water Availability")

st.sidebar.markdown(
    """
### Hydrological Model

**Rainfall → SCS-CN → Runoff → Water Availability**
"""
)

st.sidebar.divider()

st.sidebar.subheader("Dam Site")

latitude = st.sidebar.number_input(
    "Latitude",
    min_value=-90.0,
    max_value=90.0,
    value=34.1500,
    format="%.6f"
)

longitude = st.sidebar.number_input(
    "Longitude",
    min_value=-180.0,
    max_value=180.0,
    value=72.4500,
    format="%.6f"
)

catchment_area = st.sidebar.number_input(
    "Catchment Area (km²)",
    min_value=0.01,
    value=85.0,
    step=1.0
)

cn = st.sidebar.number_input(
    "SCS Curve Number",
    min_value=1,
    max_value=100,
    value=78,
    step=1
)

st.sidebar.divider()

st.sidebar.subheader("Simulation Period")

start_date = st.sidebar.date_input(
    "Start Date",
    value=pd.Timestamp("2000-01-01")
)

end_date = st.sidebar.date_input(
    "End Date",
    value=pd.Timestamp("2025-12-31")
)

run_model = st.sidebar.button(
    "🚀 Run Hydrological Model",
    use_container_width=True
)


# ============================================================
# MAIN PAGE
# ============================================================

st.title("💧 Dam Water Availability Assessment")

st.markdown(
    """
### AI-Assisted Hydrological Assessment

This application estimates **catchment runoff and annual water
availability** using daily rainfall and the **SCS Curve Number
(SCS-CN) method**.
"""
)

st.info(
    "Model output represents estimated direct runoff from the "
    "catchment. Reservoir routing, evaporation, transmission "
    "losses, environmental flows and storage-yield analysis are "
    "not included in this Version 1 model."
)


# ============================================================
# INPUT SUMMARY
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Latitude",
        f"{latitude:.4f}°"
    )

with col2:
    st.metric(
        "Longitude",
        f"{longitude:.4f}°"
    )

with col3:
    st.metric(
        "Catchment Area",
        f"{catchment_area:,.1f} km²"
    )

with col4:
    st.metric(
        "Curve Number",
        f"{cn}"
    )


# ============================================================
# MODEL EXECUTION
# ============================================================

if run_model:

    if start_date >= end_date:

        st.error(
            "End date must be later than start date."
        )
        st.stop()

    with st.spinner(
        "Downloading daily rainfall and running "
        "hydrological simulation..."
    ):

        try:

            rainfall_df = get_daily_rainfall(
                latitude,
                longitude,
                start_date.strftime("%Y-%m-%d"),
                end_date.strftime("%Y-%m-%d")
            )

            (
                daily,
                annual,
                monthly,
                monthly_average,
                summary
            ) = calculate_water_availability(
                rainfall_df,
                catchment_area,
                cn
            )

            st.session_state["daily"] = daily
            st.session_state["annual"] = annual
            st.session_state["monthly"] = monthly
            st.session_state["monthly_average"] = monthly_average
            st.session_state["summary"] = summary

            st.success(
                "Hydrological simulation completed successfully."
            )

        except Exception as e:

            st.error(
                f"An error occurred:\n\n{str(e)}"
            )

            st.stop()


# ============================================================
# DISPLAY RESULTS
# ============================================================

if "summary" in st.session_state:

    summary = st.session_state["summary"]
    daily = st.session_state["daily"]
    annual = st.session_state["annual"]
    monthly_average = st.session_state["monthly_average"]

    st.divider()

    st.header("📊 Water Availability Results")

    # --------------------------------------------------------
    # MAIN METRICS
    # --------------------------------------------------------

    c1, c2, c3 = st.columns(3)

    with c1:
        st.metric(
            "Mean Annual Rainfall",
            f"{summary['mean_annual_rainfall']:,.1f} mm"
        )

    with c2:
        st.metric(
            "Mean Annual Runoff",
            f"{summary['mean_annual_runoff']:,.1f} mm"
        )

    with c3:
        st.metric(
            "Annual Water Availability",
            f"{summary['mean_annual_volume_m3']:,.0f} m³"
        )

    c4, c5, c6 = st.columns(3)

    with c4:
        st.metric(
            "Annual Availability",
            f"{summary['mean_annual_volume_acft']:,.0f} acre-ft"
        )

    with c5:
        st.metric(
            "Annual Availability",
            f"{summary['mean_annual_volume_MAF']:.4f} MAF"
        )

    with c6:
        st.metric(
            "SCS Retention (S)",
            f"{summary['S']:.1f} mm"
        )


    # ========================================================
    # YEAR CLASSIFICATION
    # ========================================================

    st.subheader("Hydrological Year Classification")

    y1, y2, y3 = st.columns(3)

    with y1:
        st.metric(
            "Dry Year",
            str(summary["dry_year"]),
            f"{summary['dry_year_volume']:,.0f} m³"
        )

    with y2:
        st.metric(
            "Normal Year",
            str(summary["normal_year"]),
            f"{summary['normal_year_volume']:,.0f} m³"
        )

    with y3:
        st.metric(
            "Wet Year",
            str(summary["wet_year"]),
            f"{summary['wet_year_volume']:,.0f} m³"
        )


    # ========================================================
    # ANNUAL RUNOFF CHART
    # ========================================================

    st.subheader("Annual Water Availability")

    fig1, ax1 = plt.subplots(
        figsize=(12, 5)
    )

    ax1.bar(
        annual["year"],
        annual["annual_runoff_m3"]
    )

    ax1.set_xlabel("Year")
    ax1.set_ylabel("Runoff Volume (m³)")
    ax1.set_title(
        "Annual Estimated Catchment Runoff"
    )

    ax1.grid(
        axis="y",
        alpha=0.3
    )

    plt.tight_layout()

    st.pyplot(fig1)

    plt.close(fig1)


    # ========================================================
    # ANNUAL RAINFALL
    # ========================================================

    st.subheader("Annual Rainfall")

    fig2, ax2 = plt.subplots(
        figsize=(12, 5)
    )

    ax2.plot(
        annual["year"],
        annual["annual_rainfall_mm"],
        marker="o"
    )

    ax2.set_xlabel("Year")
    ax2.set_ylabel("Rainfall (mm)")
    ax2.set_title(
        "Annual Rainfall"
    )

    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig2)

    plt.close(fig2)


    # ========================================================
    # MONTHLY WATER AVAILABILITY
    # ========================================================

    st.subheader(
        "Average Monthly Water Availability"
    )

    fig3, ax3 = plt.subplots(
        figsize=(12, 5)
    )

    ax3.plot(
        monthly_average["month"],
        monthly_average["average_runoff_m3"],
        marker="o"
    )

    ax3.set_xlabel("Month")
    ax3.set_ylabel(
        "Average Runoff Volume (m³)"
    )

    ax3.set_title(
        "Average Monthly Runoff"
    )

    ax3.set_xticks(range(1, 13))

    ax3.set_xticklabels([
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec"
    ])

    ax3.grid(True, alpha=0.3)

    plt.tight_layout()

    st.pyplot(fig3)

    plt.close(fig3)


    # ========================================================
    # ANNUAL TABLE
    # ========================================================

    st.subheader("Annual Results")

    display_annual = annual.copy()

    display_annual[
        "annual_rainfall_mm"
    ] = display_annual[
        "annual_rainfall_mm"
    ].round(2)

    display_annual[
        "annual_runoff_mm"
    ] = display_annual[
        "annual_runoff_mm"
    ].round(2)

    display_annual[
        "annual_runoff_m3"
    ] = display_annual[
        "annual_runoff_m3"
    ].round(0)

    display_annual[
        "annual_runoff_acft"
    ] = display_annual[
        "annual_runoff_acft"
    ].round(0)

    display_annual[
        "annual_runoff_MAF"
    ] = display_annual[
        "annual_runoff_MAF"
    ].round(4)

    st.dataframe(
        display_annual,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # DAILY SIMULATION
    # ========================================================

    with st.expander(
        "View Daily Rainfall and Runoff Simulation"
    ):

        st.dataframe(
            daily,
            use_container_width=True,
            hide_index=True
        )


    # ========================================================
    # DOWNLOAD EXCEL
    # ========================================================

    st.subheader("Download Results")

    excel_file = create_excel(
        daily,
        annual,
        st.session_state["monthly"],
        monthly_average
    )

    st.download_button(
        label="📥 Download Complete Excel Results",
        data=excel_file,
        file_name="Dam_Water_Availability_Assessment.xlsx",
        mime=(
            "application/vnd.openxmlformats-officedocument."
            "spreadsheetml.sheet"
        ),
        use_container_width=True
    )


    # ========================================================
    # AI ENGINEERING INTERPRETATION
    # ========================================================

    st.divider()

    st.header("🤖 AI Engineering Interpretation")

    st.markdown(
        f"""
### Hydrological Assessment

The assessed catchment has an area of approximately
**{catchment_area:,.2f} km²**, with a selected SCS Curve Number
of **{cn}**.

For the selected simulation period, the estimated mean annual
rainfall is **{summary['mean_annual_rainfall']:,.1f} mm**, while
the corresponding mean annual direct runoff is
**{summary['mean_annual_runoff']:,.1f} mm**.

Based on the SCS-CN simulation, the estimated average annual
catchment runoff is approximately:

**{summary['mean_annual_volume_m3']:,.0f} m³/year**

or approximately:

**{summary['mean_annual_volume_acft']:,.0f} acre-ft/year.**

The lowest simulated annual runoff occurred in
**{summary['dry_year']}**, while the highest occurred in
**{summary['wet_year']}**.

These results represent estimated direct runoff generated from
the catchment based on the selected rainfall dataset and Curve
Number. For detailed dam planning and firm water-supply
assessment, additional considerations such as baseflow,
reservoir evaporation, transmission losses, environmental
releases, reservoir storage characteristics and flow regulation
should be incorporated.
"""
    )

else:

    st.info(
        "Enter the dam-site parameters in the sidebar and click "
        "**Run Hydrological Model** to start the assessment."
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "AI-Assisted Dam Water Availability Assessment | "
    "SCS Curve Number Method | Hydrological Prototype"
)
