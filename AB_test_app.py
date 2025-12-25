import streamlit as st
import pandas as pd
import numpy as np
import scipy.stats as stats
import plotly.express as px

st.set_page_config(
    page_title="A/B Testing Simulation",
    page_icon="📊",
    layout="wide"
)

st.title("A/B testing simulation for pricing")

# Parameters
st.sidebar.header("Simulation Parameters")

with st.sidebar.expander("1.Baseline", expanded=True):
    n_visitors = st.slider("Visitors per Variant", 1000, 50000, 10000, step=1000)
    base_conv = st.slider("Base Conversion Rate (%)", 1.0, 15.0, 4.0, 0.5) / 100

with st.sidebar.expander("2. Pricing Strategy", expanded=False):
    p_basic = st.number_input("Price: Basic ($)", value=10)
    p_decoy = st.number_input("Price: Decoy ($)", value=90)
    p_prem = st.number_input("Price: Premium ($)", value=100)

with st.sidebar.expander("3. Decoy Mechanics", expanded=True):
    st.write("**1. Attraction Effect (Conversion):**")
    conv_lift = st.slider(
        "Conversion Lift (%)",
        0.0, 50.0, 15.0,
        help="Relative increase. If Base is 4% and Lift is 10%, Target is 4.4%."
    ) / 100

    st.write("**2. Compromise Effect (Plan Mix):**")
    natural_cheap_pref = st.slider(
        "Natural Basic Preference (%)",
        0.0, 100.0, 70.0,
        help="In Control, what % prefer the cheap plan?"
    ) / 100

    nudge_strength = st.slider(
        "Decoy Strength (%)",
        0.0, 50.0, 25.0,
        help="% of Basic users who switch to premium because of the Decoy."
    ) / 100

    decoy_selection = st.slider(
        "Decoy Selection Rate (%)",
        0.0, 20.0, 2.0,
        help="% of buyers who actually click the decoy."
    ) / 100


# Data generation
@st.cache_data
def generate_data(n, base, lift, p_b, p_d, p_p, nat_pref, nudge, decoy_rate):
    np.random.seed(42)

    # Control
    c_conv = np.random.binomial(1, base, n)
    c_probs = [nat_pref, 1 - nat_pref]

    c_plans_list = []
    for x in c_conv:
        if x == 0:
            c_plans_list.append("None")
        else:
            c_plans_list.append(np.random.choice(["Basic", "Premium"], p=c_probs))

    # Decoy
    v_target_rate = base * (1 + lift)
    v_conv = np.random.binomial(1, v_target_rate, n)

    # Decoy Logic
    prob_decoy = decoy_rate
    remaining_prob = 1 - prob_decoy
    adjusted_basic_pref = max(0, nat_pref - nudge)

    prob_basic = adjusted_basic_pref * remaining_prob
    prob_premium = (1 - adjusted_basic_pref) * remaining_prob

    total_p = prob_basic + prob_decoy + prob_premium
    v_probs = [prob_basic / total_p, prob_decoy / total_p, prob_premium / total_p]

    v_plans_list = []
    for x in v_conv:
        if x == 0:
            v_plans_list.append("None")
        else:
            v_plans_list.append(np.random.choice(["Basic", "Decoy", "Premium"], p=v_probs))

    df = pd.DataFrame({
        'Variant': ['Control'] * n + ['With Decoy'] * n,
        'Plan': c_plans_list + v_plans_list
    })

    price_map = {"None": 0, "Basic": p_b, "Decoy": p_d, "Premium": p_p}
    df['Revenue'] = df['Plan'].map(price_map)
    df['Converted'] = df['Revenue'].apply(lambda x: 1 if x > 0 else 0)

    return df, v_target_rate

df, target_v_rate = generate_data(n_visitors, base_conv, conv_lift, p_basic, p_decoy, p_prem,
                                  natural_cheap_pref, nudge_strength, decoy_selection)

# KPI Comparison
st.header("1. KPI Comparison")

summary = df.groupby('Variant').agg(
    Visitors=('Variant', 'count'),
    Conversions=('Converted', 'sum'),
    Total_Revenue=('Revenue', 'sum'),
    ARPU=('Revenue', 'mean'),
    Std_Dev=('Revenue', 'std')
).reset_index()

summary['Conversion Rate'] = summary['Conversions'] / summary['Visitors']
summary['AOV'] = summary['Total_Revenue'] / summary['Conversions']

ctrl = summary[summary['Variant'] == 'Control'].iloc[0]
var = summary[summary['Variant'] == 'With Decoy'].iloc[0]

lift_cr = (var['Conversion Rate'] - ctrl['Conversion Rate']) / ctrl['Conversion Rate']
lift_arpu = (var['ARPU'] - ctrl['ARPU']) / ctrl['ARPU']

col1, col2, col3, col4 = st.columns(4)
col1.metric("Conversion Rate", f"{var['Conversion Rate']:.2%}", f"{lift_cr:+.1%}")
col2.metric("AOV (Average Order Value)", f"${var['AOV']:.2f}", f"{var['AOV'] - ctrl['AOV']:+.2f}")
col3.metric("ARPU (Rev/User)", f"${var['ARPU']:.2f}", f"{lift_arpu:+.1%}")
col4.metric("Total Revenue", f"${var['Total_Revenue']:,.0f}",
            f"+${var['Total_Revenue'] - ctrl['Total_Revenue']:,.0f}")

st.divider()

# Plan Selection
st.header("2. Plan Selection")
converters = df[df['Converted'] == 1]
plan_mix = converters.groupby(['Variant', 'Plan']).size().reset_index(name='Count')
plan_mix['Total'] = plan_mix.groupby('Variant')['Count'].transform('sum')
plan_mix['Percentage'] = plan_mix['Count'] / plan_mix['Total']

color_map = {"Basic": "#636EFA", "Decoy": "#EF553B", "Premium": "#00CC96"}

fig = px.bar(
    plan_mix, x="Variant", y="Percentage", color="Plan",
    text=plan_mix['Percentage'].apply(lambda x: f"{x:.1%}"),
    color_discrete_map=color_map,
    category_orders={"Plan": ["Basic", "Decoy", "Premium"]}
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# Analysis
st.header("3. Analysis")
tab1, tab2 = st.tabs(["Statistical Significance", "Bayesian Analysis"])

# T-test
with tab1:
    st.subheader("Frequentist Method (Welch's T-Test)")
    st.markdown("Variable tested: **Average Revenue Per User (ARPU)**")

    rev_c = df[df['Variant'] == 'Control']['Revenue']
    rev_v = df[df['Variant'] == 'With Decoy']['Revenue']
    t_stat, p_val = stats.ttest_ind(rev_v, rev_c, equal_var=False)

    if p_val < 0.05:
        st.success(
            f"**Significant Result (P-value: {p_val:.4f})**: The difference is statistically significant.")
    else:
        st.warning(f"**Not Significant (P-value: {p_val:.4f})**: The difference is not significant.")

# Bayesian
with tab2:
    st.subheader("Bayesian Method")

    # Bayesian Statistics Calculation
    simulations = 100000
    mu_c = ctrl['ARPU']
    se_c = ctrl['Std_Dev'] / np.sqrt(ctrl['Visitors'])
    mu_v = var['ARPU']
    se_v = var['Std_Dev'] / np.sqrt(var['Visitors'])

    # Sample for probability calculation
    posterior_control = np.random.normal(mu_c, se_c, simulations)
    posterior_variant = np.random.normal(mu_v, se_v, simulations)
    prob_v_better = (posterior_variant > posterior_control).mean()

    st.metric("Probability Variant with Decoy is Better:", f"{prob_v_better:.1%}")

    st.markdown("#### Posterior Distributions (ARPU)")

    x_min = min(mu_c - 4 * se_c, mu_v - 4 * se_v)
    x_max = max(mu_c + 4 * se_c, mu_v + 4 * se_v)
    x_values = np.linspace(x_min, x_max, 500)

    y_c = stats.norm.pdf(x_values, mu_c, se_c)
    y_v = stats.norm.pdf(x_values, mu_v, se_v)

    df_bayes_plot = pd.DataFrame({
        'Revenue per Visitor': np.concatenate([x_values, x_values]),
        'Density': np.concatenate([y_c, y_v]),
        'Variant': ['Control'] * 500 + ['With Decoy'] * 500
    })

    fig_bayes = px.line(
        df_bayes_plot,
        x='Revenue per Visitor',
        y='Density',
        color='Variant',
        color_discrete_map={'Control': '#636EFA', 'With Decoy': '#EF553B'},
    )

    fig_bayes.update_traces(fill='tozeroy', opacity=0.4)
    fig_bayes.update_layout(yaxis_title="Probability Density")

    st.plotly_chart(fig_bayes, use_container_width=True)
