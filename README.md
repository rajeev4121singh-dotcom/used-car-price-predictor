# Used Car Price Predictor – India (Enhanced)

Features: ML resale estimate; extra specs when present (mileage, engine, power, seats); historical purchase-price lookup; manual invoice price; seller asking price comparison; insurance, accident/flood damage, condition, service, tyre, repaint and modification scenario adjustments.

## Important limitation
The standard CarDekho dataset does not contain actual purchase invoices, insurance policy history, accident/flood records or service history. Historical lookup therefore uses a separate verified CSV, and condition adjustments are transparent configurable assumptions, not ML-learned effects.

## Setup (Windows PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```
Put a CarDekho CSV in `data/` with columns `name, year, selling_price, km_driven, fuel`. Prefer the richer `Car details v3.csv` if available. Then run:
```powershell
python -m src.train
streamlit run app.py
```
Open the local URL printed by Streamlit (usually http://localhost:8501).

## Historical purchase-price CSV
Copy `data/historical_car_prices_template.csv` to `data/historical_car_prices.csv` and fill verified records, or upload a CSV in the app. Required columns: `brand, model, purchase_year` and either `on_road_price_inr` or `ex_showroom_price_inr`. Optional: `variant, source, city`. The lookup matches brand/model/year (variant where available). Ex-showroom price is not the same as on-road invoice price.

Edit `config/adjustments.json` to change the visible scenario assumptions. See `UPDATES.md` for additions.
