import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from scipy import stats
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.stats.diagnostic import het_breuschpagan, het_white
from statsmodels.stats.stattools import durbin_watson
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

# =====================================
# 1. GENERATE REALISTIC DATASET
# =====================================

def generate_website_performance_data(n_samples=150):
    """
    Generate realistic website performance dataset
    """
    # Base parameters
    np.random.seed(42)
    
    # Independent variables
    page_size = np.random.normal(800, 300, n_samples)  # KB
    page_size = np.clip(page_size, 100, 2000)  # Realistic range
    
    http_requests = np.random.poisson(25, n_samples) + 5  # 5-50 requests
    
    image_size = np.random.exponential(200, n_samples)  # KB
    image_size = np.clip(image_size, 10, 1000)
    
    css_js_files = np.random.poisson(8, n_samples) + 2  # 2-20 files
    
    # Add some correlation between variables (realistic scenario)
    page_size = page_size + 0.3 * image_size + np.random.normal(0, 50, n_samples)
    http_requests = http_requests + 0.01 * page_size + np.random.normal(0, 3, n_samples)
    
    # Dependent variable: Load Time (in milliseconds)
    # Realistic relationship with some noise
    load_time = (
        500 +  # Base load time
        0.8 * page_size +  # Page size impact
        15 * http_requests +  # HTTP requests impact  
        0.5 * image_size +  # Image size impact
        25 * css_js_files +  # CSS/JS files impact
        np.random.normal(0, 200, n_samples)  # Random noise
    )
    
    # Ensure positive values and realistic range
    load_time = np.clip(load_time, 200, 8000)
    
    # Create DataFrame
    data = pd.DataFrame({
        'page_size_kb': np.round(page_size, 1),
        'http_requests': np.round(http_requests).astype(int),
        'image_size_kb': np.round(image_size, 1), 
        'css_js_files': np.round(css_js_files).astype(int),
        'load_time_ms': np.round(load_time, 1)
    })
    
    return data

# Generate dataset
df = generate_website_performance_data(150)

# Save dataset
df.to_csv('website_performance_dataset.csv', index=False)

print("Dataset Website Performance Generated!")
print(f"Dataset shape: {df.shape}")
print("\nFirst 10 rows:")
print(df.head(10))

print("\nDataset Info:")
print(df.info())

print("\nDescriptive Statistics:")
print(df.describe())

# =====================================
# 2. EXPLORATORY DATA ANALYSIS
# =====================================

# Create visualizations
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Website Performance Dataset - Exploratory Data Analysis', fontsize=16)

# Distribution plots
df['load_time_ms'].hist(bins=20, ax=axes[0,0], alpha=0.7, color='skyblue')
axes[0,0].set_title('Distribution of Load Time')
axes[0,0].set_xlabel('Load Time (ms)')

df['page_size_kb'].hist(bins=20, ax=axes[0,1], alpha=0.7, color='lightgreen')
axes[0,1].set_title('Distribution of Page Size')
axes[0,1].set_xlabel('Page Size (KB)')

df['http_requests'].hist(bins=20, ax=axes[0,2], alpha=0.7, color='orange')
axes[0,2].set_title('Distribution of HTTP Requests')
axes[0,2].set_xlabel('Number of HTTP Requests')

# Scatter plots
axes[1,0].scatter(df['page_size_kb'], df['load_time_ms'], alpha=0.6, color='blue')
axes[1,0].set_title('Page Size vs Load Time')
axes[1,0].set_xlabel('Page Size (KB)')
axes[1,0].set_ylabel('Load Time (ms)')

axes[1,1].scatter(df['http_requests'], df['load_time_ms'], alpha=0.6, color='red')
axes[1,1].set_title('HTTP Requests vs Load Time')
axes[1,1].set_xlabel('HTTP Requests')
axes[1,1].set_ylabel('Load Time (ms)')

axes[1,2].scatter(df['image_size_kb'], df['load_time_ms'], alpha=0.6, color='green')
axes[1,2].set_title('Image Size vs Load Time')
axes[1,2].set_xlabel('Image Size (KB)') 
axes[1,2].set_ylabel('Load Time (ms)')

plt.tight_layout()
plt.show()

# Correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, 
            square=True, linewidths=0.5)
plt.title('Correlation Matrix - Website Performance Variables')
plt.show()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# =====================================
# 3. MULTIPLE LINEAR REGRESSION MODEL
# =====================================

# Prepare variables
X = df[['page_size_kb', 'http_requests', 'image_size_kb', 'css_js_files']]
y = df['load_time_ms']

# Add constant for statsmodels
X_with_const = sm.add_constant(X)

# Fit the model using statsmodels for detailed statistics
model = sm.OLS(y, X_with_const).fit()

print("\n" + "="*60)
print("MULTIPLE LINEAR REGRESSION RESULTS")
print("="*60)
print(model.summary())

# Extract coefficients and statistics
coefficients = model.params
p_values = model.pvalues
r_squared = model.rsquared
adj_r_squared = model.rsquared_adj
f_statistic = model.fvalue
f_pvalue = model.f_pvalue

print(f"\nModel Summary:")
print(f"R-squared: {r_squared:.4f}")
print(f"Adjusted R-squared: {adj_r_squared:.4f}")
print(f"F-statistic: {f_statistic:.4f}")
print(f"F-statistic p-value: {f_pvalue:.4f}")

print(f"\nRegression Equation:")
equation = f"Load Time = {coefficients[0]:.2f}"
for i, var in enumerate(['page_size_kb', 'http_requests', 'image_size_kb', 'css_js_files']):
    sign = "+" if coefficients[i+1] >= 0 else ""
    equation += f" {sign}{coefficients[i+1]:.2f}*{var}"
print(equation)

# =====================================
# 4. CLASSICAL ASSUMPTION TESTS
# =====================================

print("\n" + "="*60)
print("CLASSICAL ASSUMPTION TESTS")
print("="*60)

# Get residuals and fitted values
residuals = model.resid
fitted_values = model.fittedvalues

# 4.1 NORMALITY TEST
print("\n4.1 NORMALITY TEST")
print("-" * 30)

# Shapiro-Wilk test
shapiro_stat, shapiro_p = stats.shapiro(residuals)
print(f"Shapiro-Wilk Test:")
print(f"Statistic: {shapiro_stat:.4f}")
print(f"p-value: {shapiro_p:.4f}")
print(f"Interpretation: {'Residuals are normally distributed' if shapiro_p > 0.05 else 'Residuals are not normally distributed'}")

# Kolmogorov-Smirnov test
ks_stat, ks_p = stats.kstest(residuals, 'norm', args=(residuals.mean(), residuals.std()))
print(f"\nKolmogorov-Smirnov Test:")
print(f"Statistic: {ks_stat:.4f}")
print(f"p-value: {ks_p:.4f}")
print(f"Interpretation: {'Residuals are normally distributed' if ks_p > 0.05 else 'Residuals are not normally distributed'}")

# Q-Q plot
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Histogram of residuals
ax1.hist(residuals, bins=20, alpha=0.7, color='skyblue', density=True)
ax1.set_title('Distribution of Residuals')
ax1.set_xlabel('Residuals')
ax1.set_ylabel('Density')

# Add normal curve
x = np.linspace(residuals.min(), residuals.max(), 100)
ax1.plot(x, stats.norm.pdf(x, residuals.mean(), residuals.std()), 'r-', linewidth=2, label='Normal Curve')
ax1.legend()

# Q-Q plot
stats.probplot(residuals, dist="norm", plot=ax2)
ax2.set_title('Q-Q Plot of Residuals')

plt.tight_layout()
plt.show()

# 4.2 MULTICOLLINEARITY TEST (VIF)
print("\n4.2 MULTICOLLINEARITY TEST (VIF)")
print("-" * 35)

vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]

print(vif_data)
print("\nInterpretation:")
for i, row in vif_data.iterrows():
    if row['VIF'] < 5:
        print(f"- {row['Variable']}: No multicollinearity (VIF = {row['VIF']:.2f})")
    elif row['VIF'] < 10:
        print(f"- {row['Variable']}: Moderate multicollinearity (VIF = {row['VIF']:.2f})")
    else:
        print(f"- {row['Variable']}: High multicollinearity (VIF = {row['VIF']:.2f})")

# 4.3 HETEROSCEDASTICITY TEST
print("\n4.3 HETEROSCEDASTICITY TEST")
print("-" * 32)

# Breusch-Pagan test
bp_statistic, bp_pvalue, _, _ = het_breuschpagan(residuals, X_with_const)
print(f"Breusch-Pagan Test:")
print(f"Statistic: {bp_statistic:.4f}")
print(f"p-value: {bp_pvalue:.4f}")
print(f"Interpretation: {'Homoscedasticity (constant variance)' if bp_pvalue > 0.05 else 'Heteroscedasticity (non-constant variance)'}")

# White test
white_statistic, white_pvalue, _, _ = het_white(residuals, X_with_const)
print(f"\nWhite Test:")
print(f"Statistic: {white_statistic:.4f}")
print(f"p-value: {white_pvalue:.4f}")
print(f"Interpretation: {'Homoscedasticity (constant variance)' if white_pvalue > 0.05 else 'Heteroscedasticity (non-constant variance)'}")

# Plot residuals vs fitted values
plt.figure(figsize=(10, 6))
plt.scatter(fitted_values, residuals, alpha=0.6)
plt.axhline(y=0, color='red', linestyle='--')
plt.xlabel('Fitted Values')
plt.ylabel('Residuals')
plt.title('Residuals vs Fitted Values Plot')
plt.show()

# 4.4 AUTOCORRELATION TEST (if applicable)
print("\n4.4 AUTOCORRELATION TEST")
print("-" * 28)

dw_statistic = durbin_watson(residuals)
print(f"Durbin-Watson Test:")
print(f"Statistic: {dw_statistic:.4f}")
print(f"Interpretation:")
if 1.5 < dw_statistic < 2.5:
    print("- No significant autocorrelation")
elif dw_statistic < 1.5:
    print("- Positive autocorrelation detected")
else:
    print("- Negative autocorrelation detected")

# =====================================
# 5. MODEL INTERPRETATION
# =====================================

print("\n" + "="*60)
print("MODEL INTERPRETATION")
print("="*60)

print("\nCoefficient Interpretation:")
var_names = ['page_size_kb', 'http_requests', 'image_size_kb', 'css_js_files']
var_descriptions = [
    'Page Size (KB)', 
    'HTTP Requests', 
    'Image Size (KB)', 
    'CSS/JS Files'
]

for i, (var, desc) in enumerate(zip(var_names, var_descriptions)):
    coef = coefficients[i+1]
    p_val = p_values[i+1]
    significance = "significant" if p_val < 0.05 else "not significant"
    direction = "positive" if coef > 0 else "negative"
    
    print(f"\n{desc}:")
    print(f"  - Coefficient: {coef:.4f}")
    print(f"  - p-value: {p_val:.4f}")
    print(f"  - Effect: {direction} and {significance}")
    print(f"  - Interpretation: For each unit increase in {desc.lower()}, load time {'increases' if coef > 0 else 'decreases'} by {abs(coef):.2f} milliseconds, holding other variables constant.")

# =====================================
# 6. CONCLUSIONS
# =====================================

print("\n" + "="*60)
print("CONCLUSIONS")
print("="*60)

print(f"\n1. Model Goodness of Fit:")
print(f"   - The model explains {r_squared*100:.1f}% of the variance in website load time")
print(f"   - F-test p-value: {f_pvalue:.4f} - Model is {'significant' if f_pvalue < 0.05 else 'not significant'}")

print(f"\n2. Variable Significance:")
significant_vars = [var_names[i] for i in range(len(var_names)) if p_values[i+1] < 0.05]
print(f"   - Significant predictors: {', '.join(significant_vars) if significant_vars else 'None'}")

print(f"\n3. Classical Assumptions:")
print(f"   - Normality: {'Satisfied' if shapiro_p > 0.05 else 'Violated'}")
print(f"   - Multicollinearity: {'No issues' if all(vif_data['VIF'] < 5) else 'Some concerns'}")
print(f"   - Homoscedasticity: {'Satisfied' if bp_pvalue > 0.05 else 'Violated'}")
print(f"   - Autocorrelation: {'No issues' if 1.5 < dw_statistic < 2.5 else 'Present'}")

print(f"\n4. Practical Implications:")
print(f"   - Most impactful factor: {var_names[np.argmax(np.abs(coefficients[1:]))]}")
print(f"   - The model can be used to predict website load time based on technical characteristics")
print(f"   - Website optimization should focus on the most significant factors")

print("\n" + "="*60)
print("DATA EXPORT COMPLETED")
print("="*60)
print("Files created:")
print("- website_performance_dataset.csv (dataset)")
print("- All analysis results displayed above")