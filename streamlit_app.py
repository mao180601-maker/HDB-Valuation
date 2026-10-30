import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# ReportLab imports for PDF generation
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors

# ==========================================
# PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="HDB Property Valuation Dashboard",
    page_icon="🏠",
    layout="wide"
)

# ==========================================
# 1. LOAD MODEL & EMBEDDED DATA (No CSV Needed)
# ==========================================
@st.cache_resource
def load_assets():
    model_package = joblib.load('hdb_random_forest_model.pkl')
    return model_package

model_package = load_assets()

rf_pipeline = model_package['pipeline']
max_year = int(model_package['max_year'])
global_mean_price = model_package['global_mean_price']
r2_score_val = model_package.get('r2_score', 0.898)
rmse_val = model_package.get('rmse', 55303.0)
df = model_package['raw_data']

# ==========================================
# 2. SIDEBAR / INPUT CONTROLS
# ==========================================
st.sidebar.header("🏠 Property Parameters")

town_options = sorted(df['town'].unique().tolist())
u_town = st.sidebar.selectbox("Town", options=town_options, index=town_options.index('BUKIT BATOK') if 'BUKIT BATOK' in town_options else 0)

streets = sorted(df[df['town'] == u_town]['street_name'].unique().tolist())
u_street = st.sidebar.selectbox("Street", options=streets)

flat_options = sorted(df['flat_type'].unique().tolist())
u_flat = st.sidebar.selectbox("Flat type", options=flat_options, index=flat_options.index('4 ROOM') if '4 ROOM' in flat_options else 0)

u_size = st.sidebar.number_input("Size sqm", min_value=30, max_value=200, value=110)

storey_options = sorted(df['storey_range'].unique().tolist())
u_storey = st.sidebar.selectbox("Floor", options=storey_options, index=storey_options.index('04 TO 06') if '04 TO 06' in storey_options else 0)

u_year = st.sidebar.number_input("Valuation Yr", min_value=2010, max_value=2100, value=max_year)

u_growth_pct = st.sidebar.number_input("Growth Rate (%)", min_value=-50.0, max_value=50.0, value=2.5, step=0.1)
u_growth = u_growth_pct / 100.0

# ==========================================
# MAIN DASHBOARD TABS
# ==========================================
st.title("Singapore HDB Resale Valuation Dashboard")
tab1, tab2 = st.tabs(["Property Valuation Predictor", "Comparable Transactions Analysis"])

# ==========================================
# 3. PDF GENERATION LOGIC
# ==========================================
def generate_pdf_report(u_town, u_street, u_flat, u_size, u_storey, u_year, growth_pct, predicted_lease, pred_price, pred_interval_low, pred_interval_high, r2_score_val, rmse_val, chart_filename, comps_df):
    pdf_path = "hdb_valuation_report.pdf"
    doc = SimpleDocTemplate(pdf_path, pagesize=letter, rightMargin=36, leftMargin=36, topMargin=36, bottomMargin=36)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle('TitleStyle', parent=styles['Heading1'], fontSize=22, leading=26, textColor=colors.HexColor('#1A365D'), spaceAfter=15)
    header_style = ParagraphStyle('HeaderStyle', parent=styles['Heading2'], fontSize=16, leading=20, textColor=colors.HexColor('#2B6CB0'), spaceBefore=10, spaceAfter=8)
    body_style = ParagraphStyle('BodyStyle', parent=styles['Normal'], fontSize=12, leading=16, textColor=colors.HexColor('#2D3748'))
    large_stat_style = ParagraphStyle('LargeStat', parent=styles['Normal'], fontSize=18, leading=22, fontName='Helvetica-Bold', textColor=colors.HexColor('#2C5282'))

    elements = []
    elements.append(Paragraph("HDB Property Valuation Report", title_style))
    elements.append(Paragraph(f"<b>Property:</b> {u_flat} | {u_street}, {u_town}", body_style))
    elements.append(Paragraph(f"<b>Specifications:</b> {u_size} sqm | Floor: {u_storey} | Valuation Year: <b>{u_year}</b>", body_style))
    elements.append(Paragraph(f"<b>Annual Growth Rate:</b> <b>{growth_pct:.1f}% per year</b> | <b>Est. Lease Remain:</b> {predicted_lease:.1f} yrs", body_style))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"Predicted valuation: S${pred_price:,.0f}", large_stat_style))
    elements.append(Paragraph(f"<b>Prediction Intervals:</b> S${pred_interval_low:,.0f} to S${pred_interval_high:,.0f}", body_style))
    elements.append(Paragraph(f"<b>Test Set R² Score:</b> {r2_score_val:.3f}", body_style))
    elements.append(Paragraph(f"<b>RMSE:</b> S${rmse_val:,.0f}", body_style))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph("<b>Indicator Explanations & Examples:</b>", header_style))
    elements.append(Paragraph("• <b>Prediction Intervals:</b> Shows lower and upper price range (e.g., 95% certainty) based on tree-to-tree variations.<br/><i>Example:</i> Narrower is better ($720K–$740K indicates high certainty vs $600K–$850K).", body_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph("• <b>Test Set R² Score:</b> Measures overall model accuracy on unseen data.<br/><i>Example:</i> Higher is better (0.85+ is excellent, below 0.50 is poor).", body_style))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph("• <b>RMSE:</b> Represents average dollar error across predictions.<br/><i>Example:</i> Lower is better ($25,000 deviation is strong).", body_style))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("Historical Area Price Distribution (Adjusted)", header_style))
    elements.append(RLImage(chart_filename, width=450, height=240))
    elements.append(Spacer(1, 15))

    elements.append(Paragraph("15 Most Comparable Transactions (Nearest to Valuation Year)", header_style))
    table_data = [["Year", "Month", "Block", "Street", "Type", "Size", "Storey", "Price"]]
    for _, row in comps_df.head(15).iterrows():
        table_data.append([
            str(row['resale_year']), str(row['resale_month']), str(row['block']),
            str(row['street_name']), str(row['flat_type']), str(row['floor_area_sqm']),
            str(row['storey_range']), f"${row['resale_price']:,.0f}"
        ])

    t = Table(table_data, colWidths=[40, 40, 45, 120, 50, 45, 60, 70])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#2B6CB0')),
        ('TEXTCOLOR', (0,0), (-1,0), colors.whitesmoke),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 9),
        ('BOTTOMPADDING', (0,0), (-1,0), 6),
        ('GRID', (0,0), (-1,-1), 0.5, colors.HexColor('#CBD5E0'))
    ]))
    elements.append(t)
    doc.build(elements)

# ==========================================
# 4. PREDICTION & CALCULATION LOGIC
# ==========================================
matching_records = df[(df['town'] == u_town) & (df['street_name'] == u_street)]
if len(matching_records) > 0:
    base_lease = matching_records['lease_remain'].median()
    base_record_year = matching_records['resale_year'].median()
else:
    town_records = df[df['town'] == u_town]
    base_lease = town_records['lease_remain'].median() if len(town_records) > 0 else 85.0
    base_record_year = town_records['resale_year'].median() if len(town_records) > 0 else max_year

year_diff = int(u_year - base_record_year)
predicted_lease = max(10.0, base_lease - year_diff)

input_data = pd.DataFrame([{
    'town': u_town,
    'flat_type': u_flat,
    'storey_range': u_storey,
    'floor_area_sqm': u_size,
    'flat_model': 'Model A',
    'resale_year': int(u_year),
    'lease_remain': predicted_lease
}])

base_pred_price = rf_pipeline.predict(input_data)[0]

historical_baseline_year = max_year
compounding_years = max(0, int(u_year - historical_baseline_year))
pred_price = base_pred_price * ((1.0 + u_growth) ** compounding_years)

comparables = df[(df['town'] == u_town) & (df['flat_type'] == u_flat)].copy()
if len(comparables) < 15:
    comparables = df[df['town'] == u_town].copy()

comparables['year_diff'] = (comparables['resale_year'] - int(u_year)).abs()
if 'resale_month' in comparables.columns:
    comparables = comparables.sort_values(by=['year_diff', 'resale_year', 'resale_month'], ascending=[True, False, False])
else:
    comparables = comparables.sort_values(by=['year_diff', 'resale_year'], ascending=[True, False])

prices = comparables['resale_price'].values * ((1.0 + u_growth) ** compounding_years)
median_p = np.median(prices)

preprocessor = rf_pipeline.named_steps['preprocessor']
regressor = rf_pipeline.named_steps['regressor']
processed_input = preprocessor.transform(input_data)
tree_preds = np.array([tree.predict(processed_input)[0] for tree in regressor.estimators_]) * ((1.0 + u_growth) ** compounding_years)

pred_interval_low, pred_interval_high = np.percentile(tree_preds, [2.5, 97.5])

plt.figure(figsize=(9, 5))
sns.set_theme(style="whitegrid")
sns.histplot(prices, bins=12, color='#00A3A6', edgecolor='white')
plt.axvline(median_p, color='#1A365D', linestyle='--', linewidth=2.5, label='Median')
plt.axvline(pred_price, color='#FF9933', linestyle='-', linewidth=2.5, label='Prediction')
plt.title(f'Historical area price distribution ({int(u_year)} @ {u_growth_pct:.1f}% p.a. growth)', fontsize=14, fontweight='bold')
plt.xlabel('Price (S$)', fontsize=12, fontweight='bold')
plt.ylabel('Transactions', fontsize=12, fontweight='bold')
plt.legend(loc='upper right', fontsize=11)
plt.tight_layout()

chart_path = 'valuation_histogram.png'
plt.savefig(chart_path, dpi=200)
plt.close()

generate_pdf_report(u_town, u_street, u_flat, u_size, u_storey, int(u_year), u_growth_pct, predicted_lease, pred_price, pred_interval_low, pred_interval_high, r2_score_val, rmse_val, chart_path, comparables)

display_cols = ['resale_year', 'resale_month', 'block', 'street_name', 'flat_type', 'floor_area_sqm', 'storey_range', 'resale_price']
top_comps = comparables.head(15)[display_cols].copy()
top_comps['resale_price'] = top_comps['resale_price'].map(lambda x: f"${x * ((1.0 + u_growth) ** compounding_years):,.2f}")

def render_indicators_box():
    st.info(
        "📊 **Indicator Explanations & Examples:**\n\n"
        "• **Prediction Intervals:** Shows lower and upper price range (e.g., 95% certainty) based on tree-to-tree variations.\n"
        "  *Example:* Narrower is better ($720K–$740K indicates high certainty vs $600K–$850K).\n\n"
        "• **Test Set R² Score:** Measures overall model accuracy on unseen data.\n"
        "  *Example:* Higher is better (0.85+ is excellent, below 0.50 is poor).\n\n"
        "• **RMSE:** Represents average dollar error across predictions.\n"
        "  *Example:* Lower is better ($25,000 deviation is strong)."
    )

# ==========================================
# RENDER TAB 1
# ==========================================
with tab1:
    st.image(chart_path)

    st.markdown(f"""
<div style="font-family: Arial; line-height: 1.5; margin-top: 15px;">
    <h2 style="font-size: 26px; color: #1A365D; margin-bottom: 5px;">Predicted valuation: S${pred_price:,.0f}</h2>
    <p style="font-size: 16px; font-weight: bold; color: #2B6CB0;">Valuation Year: {int(u_year)} | Growth Rate: {u_growth_pct:.1f}% p.a. | Lease Remain: {predicted_lease:.1f} yrs</p>
    <p style="font-size: 16px; font-weight: bold; color: #2C5282;">Prediction Intervals: S${pred_interval_low:,.0f} to S${pred_interval_high:,.0f}</p>
    <p style="font-size: 16px; font-weight: bold; color: #2D3748;">Test Set R² Score: {r2_score_val:.3f} | RMSE: S${rmse_val:,.0f}</p>
</div>
""", unsafe_allow_html=True)

    render_indicators_box()

    with open("hdb_valuation_report.pdf", "rb") as pdf_file:
        st.download_button(
            label="📄 Download PDF Valuation Report",
            data=pdf_file,
            file_name="hdb_valuation_report.pdf",
            mime="application/octet-stream"
        )

    st.markdown("<p style='font-family: monospace; font-size: 13px; color: #4A5568; background: #EDF2F7; padding: 6px; border-radius: 4px;'>📄 PDF Report successfully generated and saved as: hdb_valuation_report.pdf</p>", unsafe_allow_html=True)
    st.markdown("<h3>15 Most Comparable Transactions (Nearest to Valuation Year)</h3>", unsafe_allow_html=True)
    st.dataframe(top_comps.reset_index(drop=True), use_container_width=True)

# ==========================================
# RENDER TAB 2
# ==========================================
with tab2:
    st.markdown(f"""
<div style="font-family: Arial; line-height: 1.4; margin-top: 10px;">
    <h2 style="font-size: 24px; color: #1A365D; margin-bottom: 5px;">Predicted valuation: S${pred_price:,.0f}</h2>
    <p style="font-size: 15px; font-weight: bold; color: #2C5282; margin-bottom: 2px;">Prediction Intervals: S${pred_interval_low:,.0f} to S${pred_interval_high:,.0f}</p>
    <p style="font-size: 15px; font-weight: bold; color: #2D3748; margin-bottom: 10px;">Test Set R² Score: {r2_score_val:.3f} | RMSE: S${rmse_val:,.0f}</p>
</div>
""", unsafe_allow_html=True)

    render_indicators_box()
    st.markdown("<h3>15 Most Comparable Transactions (Nearest to Valuation Year)</h3>", unsafe_allow_html=True)
    st.dataframe(top_comps.reset_index(drop=True), use_container_width=True)