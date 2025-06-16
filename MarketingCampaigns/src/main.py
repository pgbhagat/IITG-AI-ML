import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import math
import numpy as np
import stats
from scipy import stats

def clean_columns(columns):
    new_columns = [col.replace('_', '') for col in columns]
    new_columns = [col.strip() for col in new_columns]
    return new_columns

def handle_null_income(marketing_data_df):
    marketing_data_df['Income'] = marketing_data_df['Income'].astype(str)
    marketing_data_df['Income'] = marketing_data_df['Income'].str.replace('$', '', regex=False)
    marketing_data_df['Income'] = marketing_data_df['Income'].str.replace(',', '', regex=False)
    marketing_data_df['Income'] = pd.to_numeric(marketing_data_df['Income'], errors='coerce')

    marketing_data_df['Income'] = marketing_data_df.groupby(['Education', 'MaritalStatus'])['Income'].transform(
        lambda x: x.fillna(x.mean()))
    return marketing_data_df

def visualize_distributions_and_outliers(df, numerical_cols):
    """
    Creates box plots and histograms for the specified numerical columns
    to visualize distributions and outliers.

    Args:
        df (pd.DataFrame): The input DataFrame.
        numerical_cols (list): A list of numerical column names to plot.
    """
    print("\n--- Visualizing Distributions and Outliers ---")

    num_plots_per_col = 2  # Box plot and Histogram
    valid_cols = [col for col in numerical_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
    num_valid_cols = len(valid_cols)

    # Calculate grid dimensions: aiming for roughly square layout
    total_plots_needed = num_valid_cols * num_plots_per_col
    num_cols_in_grid = 4  # You can adjust this for wider/narrower layout
    num_rows_in_grid = math.ceil(total_plots_needed / num_cols_in_grid)

    plt.figure(figsize=(num_cols_in_grid * 4, num_rows_in_grid * 4))  # Adjust figure size as needed
    plt.suptitle('Distributions and Outliers of Numerical Features', y=1.02, fontsize=2)

    plot_index = 1
    for col in valid_cols:
        # Box Plot
        plt.subplot(num_rows_in_grid, num_cols_in_grid, plot_index)
        sns.boxplot(y=df[col])
        plt.title(f'Box Plot: {col}')
        plt.ylabel('')  # Remove y-label for cleaner look in subplots
        plot_index += 1

        # Histogram
        plt.subplot(num_rows_in_grid, num_cols_in_grid, plot_index)
        sns.histplot(df[col].dropna(), kde=True)
        plt.title(f'Hist: {col}')
        plt.xlabel('')  # Remove x-label for cleaner look in subplots
        plt.ylabel('')  # Remove y-label for cleaner look in subplots
        plot_index += 1

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])  # Adjust rect to make space for suptitle
    plt.show()

    print("--- End Visualization ---")

def cap_outliers_iqr(df, column, whis=1.5):
    """
    Treat the outliers in a specified column using interquartile range (IQR)
    Value below Q1-whis*IRQ will be set to Q1-whis*IRQ
    Value above Q3+whis*IRQ will be set to Q3+whis*IRQ
    """
    if column in df.columns and pd.api.types.is_numeric_dtype(df[column]):
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - whis * IQR
        upper = Q3 + whis * IQR
        outliers = df[(df[column] < lower) | (df[column] > upper)].shape[0]
        print(f'Number of outliers found: {outliers} for column: {column}')
        print(f'Using lower bound: {lower} and upper bound: {upper} for column: {column}')

        df[column] = np.where(df[column] < lower, lower, df[column])
        df[column] = np.where(df[column] > upper, upper, df[column])
        print(f"  Outliers in '{column}' have been capped.")

    return df

def show_heatmap_of_columns(cols):
    if len(cols) > 1:
        correlation = df_encoded[cols].corr()
        plt.figure(figsize=(18, 12))
        sns.heatmap(correlation, annot=True, fmt='0.2f', linewidths=.5, cmap='coolwarm')
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.title('Correlation Heatmap', fontsize=10, fontweight='bold')
        plt.show()

marketing_data_df = pd.read_csv('../resources/marketing_data.csv')
marketing_data_df.columns = clean_columns(marketing_data_df.columns)
print('Null income column before filling with mean', marketing_data_df['Income'].isnull().sum())
marketing_data_df = handle_null_income(marketing_data_df)
print('Null income column after filling with mean', marketing_data_df['Income'].isnull().sum())

if 'Kidhome' in marketing_data_df.columns and 'Teenhome' in marketing_data_df.columns:
    marketing_data_df['TotalChildren'] = marketing_data_df['Kidhome'] + marketing_data_df['Teenhome']
else:
    print("Could not create 'TotalChildren': 'Kidhome' or 'Teenhome' column not found.")

if 'YearBirth' in marketing_data_df.columns:
    marketing_data_df['Age'] = datetime.now().year - marketing_data_df['YearBirth']
else:
    print("Could not create 'Age': 'YearBirth' column not found.")

spending_cols = [
    'MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts', 'MntGoldProds'
]
if all(col in marketing_data_df.columns for col in spending_cols):
    marketing_data_df['TotalSpending'] = marketing_data_df[spending_cols].sum(axis=1)
else:
    missing_spending_cols = [col for col in spending_cols if col not in marketing_data_df.columns]
    print(f"Could not create 'TotalSpending': Missing columns {missing_spending_cols}.")

#print(marketing_data_df[['ID', 'TotalSpending']])
print(marketing_data_df.columns)

#Exclude ID and YeadBirth as Age is calculated
#also exclude binary columns AcceptedCmp*, Response, Complain.
#exclude country
numerical_cols_for_analysis = [
    'Income', 'Recency', 'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds',
    'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'NumWebVisitsMonth', 'TotalChildren', 'Age', 'TotalSpending'
]
actual_numerical_cols = [col for col in numerical_cols_for_analysis
                         if col in marketing_data_df.columns and pd.api.types.is_numeric_dtype(marketing_data_df[col])]

print("VISUALIZING DISTRIBUTIONS AND OUTLIERS (BEFORE TREATMENT)")

#visualize_distributions_and_outliers(marketing_data_df, actual_numerical_cols)

cols_to_treat_outliers = [
    'Income', 'Age', 'MntWines', 'MntFruits', 'MntMeatProducts',
    'MntFishProducts', 'MntSweetProducts', 'MntGoldProds', 'TotalSpending',
    'Recency', 'NumDealsPurchases', 'NumWebPurchases', 'NumCatalogPurchases',
    'NumStorePurchases', 'NumWebVisitsMonth', 'TotalChildren'
]
# Filter columns to only include those that actually exist and are numeric in the DataFrame
actual_cols_to_treat = [col for col in cols_to_treat_outliers
                        if col in marketing_data_df.columns and pd.api.types.is_numeric_dtype(marketing_data_df[col])]

for col in actual_cols_to_treat:
    marketing_data_df = cap_outliers_iqr(marketing_data_df, col)

print("VISUALIZING DISTRIBUTIONS AND OUTLIERS (AFTER TREATMENT)")
#visualize_distributions_and_outliers(marketing_data_df, actual_numerical_cols)

df_encoded = marketing_data_df.copy()

education_order = ['Basic', '2n Cycle', 'Graduation', 'Master', 'PhD']
if 'Education' in df_encoded.columns:
    #print(f"EDUCATION unique values: {df_encoded.Education.unique()}")
    education_mapping = {level: i for i, level in enumerate(education_order)}
    #print(f"EDUCATION mapping: {education_mapping}")
    df_encoded['EducationEncoded'] = df_encoded['Education'].map(education_mapping)
    df_encoded = df_encoded.drop('Education', axis=1)
    print(f"New 'EducationEncoded' column created. First few values:\n{df_encoded['EducationEncoded'].head()}")

one_hot_cols = ['MaritalStatus', 'Country']
initial_cols = df_encoded.columns.tolist()
existing_one_hot_cols = [col for col in one_hot_cols if col in df_encoded.columns]

if existing_one_hot_cols:
    df_encoded = pd.get_dummies(df_encoded, columns=existing_one_hot_cols, drop_first=True, dtype=int)
    print("One hot encoding applied, New columns added:", [col for col in df_encoded.columns if col not in initial_cols])

#print(df_encoded.columns.tolist())
#print(df_encoded.head(5))

numerical_and_encoded_cols = [col for col in df_encoded.columns if pd.api.types.is_numeric_dtype(df_encoded[col]) and col != 'ID']
#print(numerical_and_encoded_cols)
#show_heatmap_of_columns(numerical_and_encoded_cols)

print("-"*30)
print('Hypothesis testing: Older people prefer shopping in-store')
print("-"*30)

correlation_age_store_purchases = df_encoded['Age'].corr(df_encoded['NumStorePurchases'])
correlation_age_web_purchases = df_encoded['Age'].corr(df_encoded['NumWebPurchases'])
correlation_age_web_visit = df_encoded['Age'].corr(df_encoded['NumWebVisitsMonth'])

print(f"Correlation between Age and NumStorePurchases: {correlation_age_store_purchases:.2f}")
print(f"Correlation between Age and NumWebPurchases: {correlation_age_web_purchases:.2f}")
print(f"Correlation between Age and NumWebVisitsMonth: {correlation_age_web_visit:.2f}")

plt.figure(figsize = (18, 12))
plt.subplot(1, 3, 1)
sns.scatterplot(x='Age', y='NumStorePurchases', data = df_encoded)
plt.title('Age vs. Store Purchases')
plt.xlabel('Age')
plt.ylabel('Number of Store Purchases')

plt.subplot(1, 3, 2)
sns.scatterplot(x='Age', y='NumWebPurchases', data = df_encoded)
plt.title('Age vs. Web Purchases')
plt.xlabel('Age')
plt.ylabel('Number of Web Purchases')

plt.subplot(1, 3, 3)
sns.scatterplot(x='Age', y='NumWebVisitsMonth', data=df_encoded)
plt.title('Age vs. NumWebVisitsMonth')
plt.xlabel('Age')
plt.ylabel('Number of Web Visited per month')
plt.tight_layout()
#plt.show()



print("-"*30)
print('Hypothesis testing: Customers having kids prefers web purchases')
print("-"*30)

correlation_total_children_store_purchases = df_encoded['TotalChildren'].corr(df_encoded['NumStorePurchases'])
correlation_total_children_web_purchases = df_encoded['TotalChildren'].corr(df_encoded['NumWebPurchases'])
correlation_total_children_web_visit = df_encoded['TotalChildren'].corr(df_encoded['NumWebVisitsMonth'])

print(f"Correlation between TotalChildren and NumStorePurchases: {correlation_total_children_store_purchases:.2f}")
print(f"Correlation between TotalChildren and NumWebPurchases: {correlation_total_children_web_purchases:.2f}")
print(f"Correlation between TotalChildren and NumWebVisitsMonth: {correlation_total_children_web_visit:.2f}")

plt.figure(figsize = (18, 12))
plt.subplot(1, 3, 1)
sns.scatterplot(x='TotalChildren', y='NumStorePurchases', data = df_encoded)
plt.title('TotalChildren vs. Store Purchases')
plt.xlabel('TotalChildren')
plt.ylabel('Number of Store Purchases')

plt.subplot(1, 3, 2)
sns.scatterplot(x='TotalChildren', y='NumWebPurchases', data = df_encoded)
plt.title('TotalChildren vs. Web Purchases')
plt.xlabel('TotalChildren')
plt.ylabel('Number of Web Purchases')

plt.subplot(1, 3, 3)
sns.scatterplot(x='TotalChildren', y='NumWebVisitsMonth', data=df_encoded)
plt.title('TotalChildren vs. NumWebVisitsMonth')
plt.xlabel('TotalChildren')
plt.ylabel('Number of Web Visited per month')
plt.tight_layout()
#plt.show()


print("-"*30)
print('Hypothesis testing: Other distribution channels may cannibalize sales at the store.')
print("-"*30)
# as the purchases/visit though web or catalog increases, sales at store may decrease
channels_to_check = ['NumWebPurchases', 'NumCatalogPurchases', 'NumDealsPurchases', 'NumWebVisitsMonth']
store_purchases_correlations = {}

for channel in channels_to_check:
    if channel in df_encoded.columns and 'NumStorePurchases' in df_encoded.columns:
        corr_val = df_encoded['NumStorePurchases'].corr(df_encoded[channel])
        store_purchases_correlations[channel] = corr_val
        print(f"Correlation between NumStorePurchases and {channel}: {corr_val:.2f}")

print("\n--- Visualizing NumStorePurchases vs. Other Channels ---")

plt.figure(figsize=(20, 5))
for i, channel in enumerate(channels_to_check):
    if channel in df_encoded.columns and 'NumStorePurchases' in df_encoded.columns:
        plt.subplot(1, len(channels_to_check), i + 1)
        sns.scatterplot(x=channel, y='NumStorePurchases', data=df_encoded)
        plt.title(f'Store Purchases vs. {channel}')
        plt.xlabel(channel)
        plt.ylabel('Number of Store Purchases')

plt.tight_layout()
#plt.show()


print(df_encoded.columns.to_list())

# --- Hypothesis Testing 4: Does the US fare significantly better than the rest of the world in terms of total purchases? ---
print("-"*30)
print("HYPOTHESIS TESTING 4: US vs. REST OF WORLD - TOTAL PURCHASES")
print("-"*30)

# The hypothesis is that US customers have higher total purchases than non-US customers.
# This requires the 'Country_US' column from one-hot encoding.

if 'Country_US' in df_encoded.columns and 'TotalSpending' in df_encoded.columns:
    # Separate data for US and Non-US customers
    us_spending = df_encoded[df_encoded['Country_US'] == 1]['TotalSpending']
    non_us_spending = df_encoded[df_encoded['Country_US'] == 0]['TotalSpending']

    print(f"\n--- Descriptive Statistics for Total Spending ---")
    print(f"US Total Spending (Mean): {us_spending.mean():.2f}")
    print(f"US Total Spending (Median): {us_spending.median():.2f}")
    print(f"Non-US Total Spending (Mean): {non_us_spending.mean():.2f}")
    print(f"Non-US Total Spending (Median): {non_us_spending.median():.2f}")

    # Perform an independent two-sample t-test
    # This tests if the means of two independent samples are significantly different.
    # A low p-value (e.g., < 0.05) suggests a significant difference.
    # Assumptions for t-test: normality (can be relaxed for large N), equal variances (can use Welch's t-test for unequal)
    if len(us_spending) > 1 and len(non_us_spending) > 1: # Need at least 2 samples for t-test
        print("\n--- Performing Independent Samples t-test ---")
        t_stat, p_value = stats.ttest_ind(us_spending, non_us_spending, equal_var=False) # Welch's t-test (assumes unequal variances)
        print(f"T-statistic: {t_stat:.2f}")
        print(f"P-value: {p_value:.3f}")

        alpha = 0.8
        print(f"Significance Level (alpha): {alpha}")
        if p_value < alpha:
            print("Conclusion: Reject the null hypothesis. There is a statistically significant difference in total spending between US and non-US customers.")
            if us_spending.mean() > non_us_spending.mean():
                print("Specifically, US customers tend to have significantly higher total spending.")
            else:
                print("Specifically, non-US customers tend to have significantly higher total spending.")
        else:
            print("Conclusion: Fail to reject the null hypothesis. There is no statistically significant difference in total spending between US and non-US customers at the chosen alpha level.")
    else:
        print("Not enough data in both US and Non-US groups to perform t-test.")

    # Visualize the comparison using a box plot
    print("\n--- Visualizing Total Spending by Country Group ---")
    plot_df = pd.DataFrame({
        'CountryGroup': ['US' if x == 1 else 'Non-US' for x in df_encoded['Country_US']],
        'TotalSpending': df_encoded['TotalSpending']
    })
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='CountryGroup', y='TotalSpending', data=plot_df)
    plt.title('Total Spending: US vs. Rest of World')
    plt.xlabel('Customer Group')
    plt.ylabel('Total Spending')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
   # plt.show()

else:
    print("\nCannot test hypothesis: 'Country_US' or 'TotalSpending' column not found.")
    print("Ensure 'Country' was one-hot encoded and 'TotalSpending' was created.")

print("\n--- Hypothesis Testing 4 Complete ---")

# --- Analysis for Specific Queries ---
print("\n" + "=" * 50)
print("ANALYSIS FOR SPECIFIC QUERIES")
print("=" * 50)

# Query 1: Which products are performing the best, and which are performing the least in terms of revenue?
print("\n--- Product Performance by Revenue ---")
product_revenue_cols = ['MntWines', 'MntFruits', 'MntMeatProducts', 'MntFishProducts', 'MntSweetProducts',
                        'MntGoldProds']
# Ensure all columns exist before summing
actual_product_revenue_cols = [col for col in product_revenue_cols if col in df_encoded.columns]

if actual_product_revenue_cols:
    product_revenues = df_encoded[actual_product_revenue_cols].sum().sort_values(ascending=False)
    print("Total Revenue by Product Category:")
    print(product_revenues)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=product_revenues.index, y=product_revenues.values, palette='viridis')
    plt.title('Total Revenue by Product Category')
    plt.xlabel('Product Category')
    plt.ylabel('Total Revenue')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    print(f"\nBest Performing Product: {product_revenues.index[0]} (Revenue: {product_revenues.values[0]:.2f})")
    print(f"Least Performing Product: {product_revenues.index[-1]} (Revenue: {product_revenues.values[-1]:.2f})")
else:
    print("Product revenue columns not found for analysis.")

# Query 2: Is there any pattern between the age of customers and the last campaign acceptance rate?
print("\n--- Age vs. Last Campaign Acceptance Rate ---")
if 'Age' in df_encoded.columns and 'Response' in df_encoded.columns:
    # Bin ages for better visualization of patterns
    age_bins = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # Ensure there's enough data for Age before creating bins
    if not df_encoded['Age'].empty:
        df_encoded['AgeGroup'] = pd.cut(df_encoded['Age'], bins=age_bins, right=False,
                                        labels=[f'{b}-{b + 9}' for b in age_bins[:-1]])
    else:
        df_encoded['AgeGroup'] = None
        print("Warning: 'Age' column is empty. Cannot create AgeGroup bins.")

    # Calculate acceptance rate per age group
    # Filter out None/NaN AgeGroup before grouping
    campaign_acceptance_by_age = df_encoded.dropna(subset=['AgeGroup']).groupby('AgeGroup')[
        'Response'].mean().reset_index()
    campaign_acceptance_by_age['ResponseRate (%)'] = campaign_acceptance_by_age['Response'] * 100

    if not campaign_acceptance_by_age.empty:
        print("Campaign Acceptance Rate by Age Group:")
        print(campaign_acceptance_by_age)

        plt.figure(figsize=(10, 6))
        sns.lineplot(x='AgeGroup', y='ResponseRate (%)', data=campaign_acceptance_by_age, marker='o')
        plt.title('Last Campaign Acceptance Rate by Age Group')
        plt.xlabel('Age Group')
        plt.ylabel('Acceptance Rate (%)')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        print("\nPattern Observation: Look for increasing, decreasing, or non-linear trends in the line plot.")
    else:
        print("No valid AgeGroup data or campaign acceptance data to calculate rates.")
else:
    print("'Age' or 'Response' column not found for analysis.")

# Query 3: Which Country has the greatest number of customers who accepted the last campaign?
print("\n--- Country with Most Last Campaign Acceptances ---")
# Need to reconstruct 'Country' from one-hot encoded columns for easy grouping or use original if available
# Assuming 'Country' column was dropped, we'll work with one-hot encoded 'Country_X' columns
country_cols = [col for col in df_encoded.columns if col.startswith('Country_') and col != 'Country_USA']
if 'Response' in df_encoded.columns and ('Country' in marketing_data_df.columns or country_cols):
    if 'Country' in marketing_data_df.columns:  # Prefer original 'Country' column if it still exists in marketing_data_df
        country_acceptance = marketing_data_df.groupby('Country')['Response'].sum().sort_values(ascending=False)
        print("Number of last campaign acceptances by Country (from original 'Country' column):")
        print(country_acceptance)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=country_acceptance.index, y=country_acceptance.values, palette='plasma')
        plt.title('Number of Last Campaign Acceptances by Country')
        plt.xlabel('Country')
        plt.ylabel('Number of Acceptances')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        if not country_acceptance.empty:
            print(
                f"\nCountry with the greatest number of acceptances: {country_acceptance.index[0]} ({country_acceptance.values[0]} acceptances)")
        else:
            print("No campaign acceptances found per country.")

    elif country_cols:  # Fallback to one-hot encoded if original 'Country' is gone
        # Create a temporary 'Country' column for plotting from one-hot encoded columns
        temp_country_df = df_encoded.copy()
        # Initialize a generic 'Rest of World' category
        temp_country_df['CountryTemp'] = 'Rest of World'
        # Assign specific country names based on one-hot encoded columns
        # This assumes only one country_X column will be 1 for a given row.
        for col in country_cols:
            country_name = col.replace('Country_', '')
            temp_country_df.loc[temp_country_df[col] == 1, 'CountryTemp'] = country_name

        # Also include USA if its one-hot encoded column exists
        if 'Country_USA' in temp_country_df.columns:
            temp_country_df.loc[temp_country_df['Country_USA'] == 1, 'CountryTemp'] = 'USA'

        country_acceptance = temp_country_df.groupby('CountryTemp')['Response'].sum().sort_values(ascending=False)
        print("Number of last campaign acceptances by Country (from encoded data):")
        print(country_acceptance)

        plt.figure(figsize=(10, 6))
        sns.barplot(x=country_acceptance.index, y=country_acceptance.values, palette='plasma')
        plt.title('Number of Last Campaign Acceptances by Country')
        plt.xlabel('Country')
        plt.ylabel('Number of Acceptances')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.show()

        if not country_acceptance.empty:
            print(
                f"\nCountry with the greatest number of acceptances: {country_acceptance.index[0]} ({country_acceptance.values[0]} acceptances)")
        else:
            print("No campaign acceptances found per country from encoded data.")
    else:
        print("Neither original 'Country' nor sufficient one-hot encoded country columns found for analysis.")

# Query 4: Do you see any pattern in the no. of children at home and total spend?
print("\n--- Total Children vs. Total Spending Pattern ---")
if 'TotalChildren' in df_encoded.columns and 'TotalSpending' in df_encoded.columns:
    correlation_children_spending = df_encoded['TotalChildren'].corr(df_encoded['TotalSpending'])
    print(f"Correlation between TotalChildren and TotalSpending: {correlation_children_spending:.2f}")

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.scatterplot(x='TotalChildren', y='TotalSpending', data=df_encoded)
    plt.title('Total Children vs. Total Spending')
    plt.xlabel('Total Children')
    plt.ylabel('Total Spending')
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.subplot(1, 2, 2)
    sns.boxplot(x='TotalChildren', y='TotalSpending', data=df_encoded)
    plt.title('Total Spending by Number of Children')
    plt.xlabel('Total Children')
    plt.ylabel('Total Spending')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    print(
        "\nPattern Observation: Examine the scatter plot for any linear or non-linear trends. The box plot helps visualize the distribution of spending for each number of children group. A negative correlation would suggest that more children are associated with less spending, and vice-versa for a positive correlation.")
else:
    print("'TotalChildren' or 'TotalSpending' column not found for analysis.")

# Query 5: Education background of the customers who complained in the last 2 years.
print("\n--- Education Background of Complainers in Last 2 Years ---")
# Use marketing_data_df for original 'Education' and 'DtCustomer' before one-hot encoding
if 'Complain' in marketing_data_df.columns and 'Education' in marketing_data_df.columns and 'DtCustomer' in marketing_data_df.columns:
    # Ensure DtCustomer is datetime type in marketing_data_df
    marketing_data_df['DtCustomer'] = pd.to_datetime(marketing_data_df['DtCustomer'], format='%d-%m-%Y', errors='coerce')

    # Get the latest date in the dataset to define "last 2 years"
    latest_date = marketing_data_df['DtCustomer'].max()
    if pd.isna(latest_date):
        print("Warning: 'DtCustomer' column has no valid dates after conversion. Cannot filter by date.")
    else:
        two_years_ago = latest_date - pd.DateOffset(years=2)
        print(f"Latest Customer Date: {latest_date.strftime('%Y-%m-%d')}")
        print(
            f"Filtering for complaints from: {two_years_ago.strftime('%Y-%m-%d')} to {latest_date.strftime('%Y-%m-%d')}")

        # Filter customers who complained and joined in the last 2 years
        complainers_last_2_years = marketing_data_df[
            (marketing_data_df['Complain'] == 1) &
            (marketing_data_df['DtCustomer'] >= two_years_ago) &
            (marketing_data_df['DtCustomer'] <= latest_date)
            ]

        if not complainers_last_2_years.empty:
            education_complaints = complainers_last_2_years['Education'].value_counts()
            print("\nEducation Background of Customers who Complained in the Last 2 Years:")
            print(education_complaints)

            plt.figure(figsize=(10, 6))
            sns.barplot(x=education_complaints.index, y=education_complaints.values, palette='coolwarm')
            plt.title('Education Background of Recent Complainers')
            plt.xlabel('Education Level')
            plt.ylabel('Number of Complainers')
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            plt.show()
        else:
            print("No customers found who complained in the last 2 years with valid date and complaint data.")
else:
    print("Required columns ('Complain', 'Education', or 'DtCustomer') not found for this analysis.")
