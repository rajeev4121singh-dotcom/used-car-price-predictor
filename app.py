
import datetime
import json
import os

import joblib
import numpy as np
import pandas as pd
import streamlit as st

MODEL = "models/used_car_price_model.joblib"
HISTORY = "data/historical_car_prices.csv"
ADJ = "config/adjustments.json"

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="Used Car Price Predictor India",
    page_icon="🚗",
    layout="wide"
)

st.title("🚗 Used Car Price Predictor – India")

st.caption(
    "AI resale estimate, historical purchase price, "
    "vehicle inspection and detailed price impact analysis."
)

# --------------------------------------------------
# LOAD TRAINED MODEL
# --------------------------------------------------

if not os.path.exists(MODEL):
    st.warning("Model not trained yet.")
    st.code("python -m src.train")
    st.stop()

a = joblib.load(MODEL)

model = a["model"]
features = a["features"]
nums = a["numeric_features"]
cats = a["categorical_features"]
defaults = a["defaults"]
brands = a["brands"]
bybrand = a["models_by_brand"]
fuels = a["fuels"]
year_now = a["current_year"]

# --------------------------------------------------
# PRICE ADJUSTMENT RULES
# --------------------------------------------------

BASE = {
    "excellent": 0.03,
    "good": 0,
    "average": -0.09,
    "poor": -0.15,
    "minor_accident": -0.10,
    "major_accident": -0.20,
    "flood_damage": -0.30,
    "repainted": -0.07,
    "full_service": 0.02,
    "no_service_records": -0.09,
    "insurance_expired": -0.065,
    "insurance_third_party": -0.055,
    "tyres_worn": -0.07,
    "modified": -0.07,
    "engine_minor": -0.10,
    "engine_major": -0.25,
    "gearbox_minor": -0.08,
    "gearbox_major": -0.20,
    "brakes_worn": -0.07,
    "battery_weak": -0.07,
    "rust": -0.12,
    "ac_fault": -0.08,
    "documents_pending": -0.10,
    "warranty": 0.03,
    "market_low": -0.10,
    "market_high": 0.05
}

try:
    with open(ADJ, encoding="utf-8") as f:
        BASE.update(json.load(f))
except FileNotFoundError:
    st.caption("Using default adjustment rules.")
except Exception as e:
    st.warning(f"Adjustment configuration error: {e}")


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def rupee(value):
    return f"₹{value:,.0f}"


def predict(values):
    data = pd.DataFrame([values])[features]
    return max(0.0, float(model.predict(data)[0]))


def load_history(upload):
    try:
        if upload is not None:
            return pd.read_csv(upload)

        if os.path.exists(HISTORY):
            return pd.read_csv(HISTORY)

    except Exception as e:
        st.error(f"Historical file error: {e}")

    return None


def lookup(df, brand, model_name, purchase_year, variant):

    if df is None:
        return None, "No historical price file supplied."

    d = df.copy()

    d.columns = [
        c.strip().lower() for c in d.columns
    ]

    required = {
        "brand",
        "model",
        "purchase_year"
    }

    if not required <= set(d.columns):
        return None, "Historical CSV columns are missing."

    d["by"] = (
        d.brand.astype(str).str.strip().str.casefold()
    )

    d["mo"] = (
        d.model.astype(str).str.strip().str.casefold()
    )

    d["py"] = pd.to_numeric(
        d.purchase_year,
        errors="coerce"
    )

    matches = d[
        (d.by == brand.casefold())
        & (d.mo == model_name.casefold())
        & (d.py == int(purchase_year))
    ]

    if variant and "variant" in d:

        exact = matches[
            matches.variant.astype(str)
            .str.strip()
            .str.casefold() == variant.casefold()
        ]

        if not exact.empty:
            matches = exact

    if matches.empty:
        return None, "No matching historical price."

    col = (
        "on_road_price_inr"
        if "on_road_price_inr" in matches
        else "ex_showroom_price_inr"
    )

    if col not in matches:
        return None, "Historical price column missing."

    value = pd.to_numeric(
        matches.iloc[0][col],
        errors="coerce"
    )

    if pd.isna(value):
        return None, "Invalid historical price."

    return float(value), col


# --------------------------------------------------
# 1. CAR SPECIFICATIONS
# --------------------------------------------------

st.header("1. Car Specifications")

c1, c2 = st.columns(2)

with c1:

    brand = st.selectbox("Brand", brands)

    model_name = st.selectbox(
        "Model",
        bybrand.get(brand, ["Other"])
    )

    mfg_year = st.number_input(
        "Manufacturing year",
        min_value=1990,
        max_value=year_now,
        value=max(1990, year_now - 5)
    )

    fuel = st.selectbox("Fuel type", fuels)

    km = st.number_input(
        "Kilometres driven",
        min_value=501,
        max_value=999999,
        value=50000,
        step=1000
    )

with c2:

    extra = {}

    categorical_inputs = [
        ("transmission", "Transmission"),
        ("owner", "Ownership"),
        ("seller_type", "Seller type")
    ]

    for col, label in categorical_inputs:

        if col in cats:

            extra[col] = st.selectbox(
                label,
                a["category_options"].get(
                    col, ["Unknown"]
                )
            )

    labels = {
        "mileage": "Mileage (km/l or km/kg for gas fuels)",
        "engine": "Engine capacity (CC)",
        "max_power": "Maximum power (BHP)",
        "seats": "Seating capacity"
    }

    for col in nums:

        if col not in ["age", "km_driven"]:

            default = float(
                defaults.get(col, 0) or 0
            )

            if col == "seats":

                extra[col] = st.number_input(
                    "Seating capacity",
                    min_value=2,
                    max_value=12,
                    value=int(
                        np.clip(
                            round(default or 5), 2, 12
                        )
                    )
                )

            else:

                extra[col] = st.number_input(
                    labels.get(col, col),
                    min_value=0.0,
                    value=max(0.0, default),
                    step=0.1
                )


# --------------------------------------------------
# 2. ORIGINAL PRICE AND SELLER PRICE
# --------------------------------------------------

st.header("2. Purchase and Selling Price")

p1, p2, p3 = st.columns(3)

with p1:

    purchase_year = st.number_input(
        "Year owner purchased",
        min_value=1990,
        max_value=year_now,
        value=int(mfg_year)
    )

with p2:

    original_manual = st.number_input(
        "Original purchase price (₹)",
        min_value=0,
        max_value=100000000,
        value=0,
        step=10000
    )

with p3:

    asking = st.number_input(
        "Seller asking price (₹)",
        min_value=0,
        max_value=100000000,
        value=0,
        step=10000
    )

variant = st.text_input(
    "Variant / Trim",
    placeholder="Example: VXi, ZXi+, SX"
)

upload = st.file_uploader(
    "Upload historical purchase-price CSV",
    type=["csv"]
)

history = load_history(upload)

historical, lookup_note = lookup(
    history,
    brand,
    model_name,
    purchase_year,
    variant
)

original = (
    original_manual
    if original_manual > 0
    else historical
)

if historical is not None:

    st.info(
        f"Historical purchase price: "
        f"{rupee(historical)}"
    )

else:

    st.caption(lookup_note)


# --------------------------------------------------
# 3. CAR CONDITION AND DAMAGE
# --------------------------------------------------

st.header("3. Vehicle Condition and Damage")

d1, d2 = st.columns(2)

with d1:

    condition = st.selectbox(
        "Overall car condition",
        ["excellent", "good", "average", "poor"],
        index=1,
        format_func=str.title
    )

    accident_options = {
        "No accident": "none",
        "Minor accident": "minor_accident",
        "Major accident": "major_accident"
    }

    accident = accident_options[
        st.selectbox(
            "Accident history",
            list(accident_options)
        )
    ]

    flood = st.checkbox("Flood / Water damage")

    repainted = st.checkbox(
        "Major body panels repainted"
    )

with d2:

    insurance_options = {
        "Comprehensive insurance": "active_comprehensive",
        "Third-party insurance": "insurance_third_party",
        "Expired insurance": "insurance_expired"
    }

    insurance = insurance_options[
        st.selectbox(
            "Insurance status",
            list(insurance_options)
        )
    ]

    service_options = {
        "Full service history": "full_service",
        "Partial service history": "partial_service",
        "No service records": "no_service_records"
    }

    service = service_options[
        st.selectbox(
            "Service history",
            list(service_options)
        )
    ]

    tyres = st.checkbox("Tyres need replacement")

    modified = st.checkbox(
        "Non-standard modifications"
    )


# --------------------------------------------------
# 4. MECHANICAL INSPECTION
# --------------------------------------------------

st.header("4. Mechanical Inspection")

inspection_options = {

    "Engine condition": {
        "Normal": "none",
        "Minor fault": "engine_minor",
        "Major fault": "engine_major"
    },

    "Gearbox condition": {
        "Normal": "none",
        "Minor fault": "gearbox_minor",
        "Major fault": "gearbox_major"
    },

    "Brake condition": {
        "Good": "none",
        "Replacement needed": "brakes_worn"
    },

    "Battery condition": {
        "Good": "none",
        "Weak battery": "battery_weak"
    },

    "Rust and corrosion": {
        "No significant rust": "none",
        "Significant rust": "rust"
    },

    "Air conditioning": {
        "Working": "none",
        "Faulty": "ac_fault"
    },

    "Legal documents": {
        "No known issues": "none",
        "Pending paperwork": "documents_pending"
    },

    "Warranty": {
        "No warranty": "none",
        "Valid transferable warranty": "warranty"
    },

    "Market demand": {
        "Normal demand": "none",
        "Low demand": "market_low",
        "High demand": "market_high"
    }
}

inspection = {}

columns = st.columns(3)

for i, (label, options) in enumerate(
    inspection_options.items()
):

    with columns[i % 3]:

        selected_option = st.selectbox(
            label,
            list(options)
        )

        inspection[label] = options[selected_option]


# --------------------------------------------------
# 5. CUSTOM PRICE ADJUSTMENT SETTINGS
# --------------------------------------------------

st.header("5. Price Adjustment Settings")

custom = st.checkbox(
    "Apply stricter ownership, fuel and mileage rules",
    value=False,
    help=(
        "Replaces the learned effects of these features "
        "with custom valuation assumptions."
    )
)

expected_mileage = st.number_input(
    "Expected mileage for this exact car variant",
    min_value=0.0,
    value=0.0,
    step=0.5,
    help=(
        "Use the same units as actual mileage. "
        "Enter 0 to skip the custom mileage adjustment."
    )
)

st.caption(
    "Custom adjustments are assumed valuation rules, "
    "not verified market depreciation rates."
)


# --------------------------------------------------
# 6. CALCULATE CAR PRICE
# --------------------------------------------------

if st.button(
    "Calculate Car Price",
    type="primary",
    use_container_width=True
):

    row = {
        c: defaults.get(c, 0)
        for c in features
    }

    row.update({
        "age": max(year_now - int(mfg_year), 1),
        "km_driven": float(km),
        "brand": brand,
        "model_name": model_name,
        "fuel": fuel
    })

    row.update(extra)

    # Original ML prediction

    pred = predict(row)

    selected = row.copy()

    # Reference inputs for feature comparison

    reference = {
        c: defaults.get(c, selected[c])
        for c in features
    }

    reference.update({
        "brand": brand,
        "model_name": model_name
    })

    if "owner" in features:

        owner_options = a["category_options"].get(
            "owner", []
        )

        first_owner = next(
            (
                x for x in owner_options
                if str(x).strip().lower() == "first owner"
            ),
            reference["owner"]
        )

        reference["owner"] = first_owner

    if "fuel" in features and "Petrol" in fuels:

        reference["fuel"] = "Petrol"

    if expected_mileage > 0 and "mileage" in features:

        reference["mileage"] = expected_mileage

    custom_keys = (
        {"owner", "fuel", "mileage"}
        if custom else set()
    )

    if expected_mileage <= 0:

        custom_keys.discard("mileage")

    # Keep only features supported by the trained model

    custom_keys.intersection_update(features)

    current = reference.copy()

    start = predict(current)

    previous = start

    impacts = []

    # --------------------------------------------------
    # ML FEATURE IMPACT ANALYSIS
    # --------------------------------------------------

    feature_names = {
        "age": "Vehicle age",
        "km_driven": "Kilometres driven",
        "fuel": "Fuel type",
        "owner": "Ownership",
        "transmission": "Transmission",
        "mileage": "Mileage",
        "engine": "Engine capacity",
        "max_power": "Maximum power",
        "seats": "Seating capacity",
        "seller_type": "Seller type",
        "brand": "Brand",
        "model_name": "Car model"
    }

    for c in features:

        if c in custom_keys:
            continue

        current[c] = selected[c]

        value = predict(current)

        difference = value - previous

        impacts.append({
            "Factor": feature_names.get(c, c),
            "Method": "ML comparison",
            "Change (₹)": difference,
            "Change (%)": (
                100 * difference / start
                if start else 0.0
            )
        })

        previous = value

    base = previous

    # --------------------------------------------------
    # CONDITION ADJUSTMENTS
    # --------------------------------------------------

    adjustments = [

        ("Overall condition", condition),

        ("Accident history", accident),

        ("Insurance", insurance),

        ("Service history", service),

        (
            "Flood damage",
            "flood_damage" if flood else "none"
        ),

        (
            "Repainting",
            "repainted" if repainted else "none"
        ),

        (
            "Tyres",
            "tyres_worn" if tyres else "none"
        ),

        (
            "Modifications",
            "modified" if modified else "none"
        )

    ] + list(inspection.items())

    rates = [
        (label, BASE.get(key, 0))
        for label, key in adjustments
    ]

    # --------------------------------------------------
    # OWNERSHIP PRICE DEDUCTIONS
    # --------------------------------------------------

    if custom and "owner" in custom_keys:

        owner_rates = {
            "first owner": 0.00,
            "second owner": -0.10,
            "third owner": -0.15,
            "fourth owner": -0.20,
            "fourth & above owner": -0.20,
            "fifth owner": -0.25,
            "fifth & above owner": -0.25
        }

        owner = str(
            selected["owner"]
        ).strip().lower()

        rates.append((
            "Ownership (custom)",
            owner_rates.get(owner, 0)
        ))

    # --------------------------------------------------
    # FUEL TYPE PRICE ADJUSTMENTS
    # --------------------------------------------------

    if custom and "fuel" in custom_keys:

        fuel_rates = {
            "petrol": 0.00,
            "diesel": 0.00,
            "cng": 0.02,
            "lpg": -0.15,
            "hybrid": 0.05
        }

        rates.append((
            "Fuel type (custom)",
            fuel_rates.get(
                str(fuel).strip().lower(), 0
            )
        ))

    # --------------------------------------------------
    # MILEAGE PRICE ADJUSTMENTS
    # --------------------------------------------------

    if custom and "mileage" in custom_keys:

        actual_mileage = float(
            selected["mileage"]
        )

        ratio = actual_mileage / expected_mileage

        if ratio < 0.45:
            mileage_rate = -0.20

        elif ratio < 0.56:
            mileage_rate = -0.15

        elif ratio < 0.67:
            mileage_rate = -0.12

        elif ratio < 0.83:
            mileage_rate = -0.09

        elif ratio < 0.94:
            mileage_rate = -0.07

        elif ratio < 1.06:
            mileage_rate = 0.00

        elif ratio < 1.22:
            mileage_rate = -0.02

        elif ratio < 1.39:
            mileage_rate = 0.00

        else:
            mileage_rate = 0.02

        rates.append((
            "Mileage (custom)",
            mileage_rate
        ))

    # --------------------------------------------------
    # INDIVIDUAL PRICE IMPACT
    # --------------------------------------------------

    for label, rate in rates:

        impacts.append({
            "Factor": label,
            "Method": "Rule-based adjustment",
            "Change (₹)": base * rate,
            "Change (%)": rate * 100
        })

    raw_factor = 1 + sum(
        rate for _, rate in rates
    )

    # Minimum valuation floor of 35%

    factor = max(0.35, raw_factor)

    if factor != raw_factor:

        impacts.append({
            "Factor": "Minimum valuation floor",
            "Method": "Policy adjustment",
            "Change (₹)": base * (
                factor - raw_factor
            ),
            "Change (%)": 100 * (
                factor - raw_factor
            )
        })

    adjusted = base * factor

    # Indicative prediction range

    low = max(
        0,
        float(
            np.expm1(
                np.log1p(base) + a["res_lo"]
            )
        )
    ) * factor

    high = max(
        low,
        max(
            0,
            float(
                np.expm1(
                    np.log1p(base) + a["res_hi"]
                )
            )
        ) * factor
    )

    df = pd.DataFrame(impacts)

    increases = df.loc[
        df["Change (₹)"] > 0,
        "Change (₹)"
    ].sum()

    decreases = df.loc[
        df["Change (₹)"] < 0,
        "Change (₹)"
    ].sum()

    # --------------------------------------------------
    # 7. FINAL PRICE RESULTS
    # --------------------------------------------------

    st.divider()

    st.header("6. Car Valuation Results")

    r1, r2, r3 = st.columns(3)

    r1.metric(
        "Original ML Prediction",
        rupee(pred)
    )

    r2.metric(
        "Final Estimated Price",
        rupee(adjusted),
        f"{(factor - 1) * 100:+.2f}% rule adjustment"
    )

    r3.metric(
        "Indicative Price Range",
        f"{rupee(low)} - {rupee(high)}"
    )

    # --------------------------------------------------
    # 8. PRICE INCREASE AND DECREASE SECTION
    # --------------------------------------------------

    st.header("7. Price Increase & Decrease Analysis")

    s1, s2, s3 = st.columns(3)

    s1.metric(
        "Reference Price",
        rupee(start)
    )

    s2.metric(
        "Total Price Increase",
        rupee(increases),
        (
            f"+{increases / start * 100:.2f}%"
            if start else None
        )
    )

    s3.metric(
        "Total Price Decrease",
        rupee(decreases),
        (
            f"{decreases / start * 100:.2f}%"
            if start else None
        )
    )

    positive = df[df["Change (₹)"] > 0]

    negative = df[df["Change (₹)"] < 0]

    neutral = df[df["Change (₹)"] == 0]

    # Positive price factors

    st.subheader("🟢 Factors Increasing Car Price")

    if positive.empty:

        st.info("No positive price adjustments.")

    else:

        st.dataframe(
            positive.style.format({
                "Change (₹)": "+₹{:,.0f}",
                "Change (%)": "+{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

    # Negative price factors

    st.subheader("🔴 Factors Decreasing Car Price")

    if negative.empty:

        st.info("No negative price adjustments.")

    else:

        st.dataframe(
            negative.style.format({
                "Change (₹)": "₹{:,.0f}",
                "Change (%)": "{:.2f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

    # Neutral price factors

    with st.expander("Factors with no price change"):

        st.dataframe(
            neutral,
            use_container_width=True,
            hide_index=True
        )

    # --------------------------------------------------
    # 9. PRICE IMPACT CHART
    # --------------------------------------------------

    st.subheader("Individual Factor Impact Chart")

    chart_data = df[
        ["Factor", "Change (₹)"]
    ].copy()

    chart_data["Impact"] = np.where(
        chart_data["Change (₹)"] >= 0,
        "Increase",
        "Decrease"
    )

    st.bar_chart(
        chart_data,
        x="Factor",
        y="Change (₹)",
        color="Impact"
    )

    # --------------------------------------------------
    # 10. FINAL PRICE BREAKDOWN
    # --------------------------------------------------

    st.subheader("Final Price Calculation")

    summary = pd.DataFrame({

        "Description": [
            "Reference Price",
            "Total Price Increase",
            "Total Price Decrease",
            "Final Estimated Price"
        ],

        "Amount (₹)": [
            start,
            increases,
            decreases,
            adjusted
        ]
    })

    st.dataframe(
        summary.style.format({
            "Amount (₹)": "₹{:,.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.success(
        f"Final Estimated Car Price: {rupee(adjusted)}"
    )

    with st.expander("Reference inputs used by ML"):

        st.write(reference)

    st.caption(
        "ML feature impacts depend on the reference "
        "inputs and calculation order. Brand and model "
        "are held fixed. Rule-based percentages use "
        "the scenario base price."
    )

    # --------------------------------------------------
    # 11. DEPRECIATION ANALYSIS
    # --------------------------------------------------

    comparison = {
        "Original ML estimate": pred,
        "Final adjusted estimate": adjusted
    }

    if original and original > 0:

        st.header("8. Car Depreciation Analysis")

        depreciation = original - adjusted

        depreciation_percent = (
            depreciation / original
        ) * 100

        q1, q2, q3 = st.columns(3)

        q1.metric(
            "Original Purchase Price",
            rupee(original)
        )

        q2.metric(
            "Price Lost",
            rupee(depreciation)
        )

        q3.metric(
            "Depreciation",
            f"{depreciation_percent:.2f}%"
        )

        comparison["Original purchase price"] = float(
            original
        )

    # --------------------------------------------------
    # 12. SELLER ASKING PRICE COMPARISON
    # --------------------------------------------------

    if asking > 0:

        st.header("9. Seller Price Comparison")

        difference = asking - adjusted

        difference_percent = (
            difference / adjusted * 100
            if adjusted else 0
        )

        b1, b2, b3 = st.columns(3)

        b1.metric(
            "Seller Asking Price",
            rupee(asking)
        )

        b2.metric(
            "Estimated Fair Price",
            rupee(adjusted)
        )

        b3.metric(
            "Price Difference",
            rupee(abs(difference))
        )

        if asking < low:

            st.success(
                "GOOD DEAL: Asking price is below "
                "the indicative valuation range. "
                "Verify vehicle condition before buying."
            )

        elif asking <= high:

            st.info(
                "FAIR DEAL: Asking price is within "
                "the indicative valuation range."
            )

        else:

            st.error(
                "OVERPRICED: Asking price is above "
                "the indicative valuation range."
            )

        st.write(
            f"Asking price vs estimate: "
            f"{difference_percent:+.2f}%"
        )

        comparison["Seller asking price"] = float(
            asking
        )

    st.subheader("Car Price Comparison Chart")

    st.bar_chart(
        pd.Series(
            comparison,
            name="Price (₹)"
        )
    )

    # --------------------------------------------------
    # 13. DOWNLOAD VALUATION REPORT
    # --------------------------------------------------

    st.subheader("Download Price Impact Report")

    report = df.to_csv(index=False).encode("utf-8-sig")

    st.download_button(
        label="Download Price Impact Report (CSV)",
        data=report,
        file_name="car_price_impact_report.csv",
        mime="text/csv"
    )

    st.warning(
        "This valuation is an estimate. Inspection "
        "adjustments are configurable assumptions. "
        "Always verify the car's condition, documents "
        "and current comparable market prices."
    )


# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------

with st.expander("Machine Learning Model Performance"):

    st.write(
        "Best Model:",
        a.get("best_model_name")
    )

    st.write(
        "Training Rows:",
        a.get("training_rows")
    )

    st.json(a.get("metrics", {}))


# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.divider()

st.caption(
    "Used Car Price Predictor India | "
    "Machine Learning and Vehicle Valuation System"
)
