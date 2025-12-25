# A-B-Testing-for-Pricing
This project is a Streamlit-based simulation tool designed to demonstrate the impact of the decoy effect on product pricing strategies. Dan Ariely, professor at Duke University and author of 'Predictably Irrational' carried out an experiment on the pricing of a magazine. He observed that the ARPU (Average Revenue Per User) significantly jumped. This experiment makes the decoy effect an important pricing strategy to be considered.

## Key Features
1. Dynamic Data Generation: Simulates user behavior and conversion rates.

2. KPI Dashboard: Displays Conversion Rate, ARPU (Average Revenue Per User), AOV (Average Order Value) which is the average amount spent per transaction and Total Revenue.

3. Frequentist Analysis: Analyzes whether the difference between the Control and decoy variant is statistically significant through a T-Test on the ARPU variable.

4. Bayesian Analysis: Provides a continuous probability distribution (Posterior Density) to visualize the likelihood of one decoy variant outperforming the control.

## Hypothesis tested

Bayesian Hypothesis: The decoy variant results in a higher ARPU than the control.

Frequentist Hypothesis: 
1. Null Hypothesis($H_0$): There is no difference in the mean revenue between the control group and decoy group. Any observed difference is due to noise.
2. Alternative Hypothesis ($H_a$): There is a statistically significant difference in the mean revenue between the control group and the decoy group.

## How to run
1. Install the required libraries using pip:
pip install streamlit pandas numpy scipy plotly

2. Run the Application:
streamlit run AB_test_app.py

## How to interpret results
The simulation adds a decoy option to a pricing page to nudge users from a basic plan toward a higher-margin premium plan.

Control: Users choose between basic and premium.

Decoy Variant: Users choose between basic, decoy, and premium.

### Analysis Methods
Statistical Significance: Uses the p-value from a T-test on ARPU to demonstrate if the ARPU difference between variants is significant or due to chance.

Bayesian Probability: Calculates the specific probability that the decoy variant is better than the control.


