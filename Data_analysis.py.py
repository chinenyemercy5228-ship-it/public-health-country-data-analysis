import pandas as pd # type: ignore
import numpy as np # type: ignore
# 1. Import the CSV as df_countries
df_countries = pd.read_csv("Countries.csv")

# 2. Check it is truly a DataFrame
print("Is DataFrame:", isinstance(df_countries, pd.DataFrame))
# Alternative: type() check
print("Type:", type(df_countries))

# 3. Print all column names
print("\nColumn names:")
print(df_countries.columns)

# 4. Shape of the dataset
print("\nShape:", df_countries.shape)

# 5. Store number of rows in 'nrow'
nrow = df_countries.shape[0]
print("Number of rows (nrow):", nrow)
# 6. Rename specified columns
df_countries = df_countries.rename(columns={
    'Agriculture (% GDP)'       : 'agriculture',
    'Education Expenditure (% GDP)': 'education_expenditure',
    'Export (% GDP)'            : 'export',
    'Health Expenditure (% GDP)': 'health_expenditure',
    'Industry (% GDP)'          : 'industry',
    'Service (% GDP)'           : 'service'
})
print("\nColumns after rename:")
print(df_countries.columns.tolist())

# 7. Investigate data types & convert if needed
print("\nData types:")
print(df_countries.dtypes)

# Year should be an integer — already int64, but ensure it as a period/int
# Country Name & Continent Name should be string/object — already string
# GDP, Population etc. are numeric — already float64/int64
# No changes needed beyond the date conversion in step 8.
# 8. Convert 'Year' to datetime type

df_countries['Year'] = pd.to_datetime(df_countries['Year'], format='%Y')
print("\nYear dtype after conversion:", df_countries['Year'].dtype)
print(df_countries['Year'].head(3))

# 9. Check for missing values and duplicates
print("\nMissing values per column:")
print(df_countries.isnull().sum())

print("\nTotal duplicate rows:", df_countries.duplicated().sum())

# 10. Drop rows with missing values and duplicates
df_countries = df_countries.dropna()
df_countries = df_countries.drop_duplicates()
print("\nShape after cleaning:", df_countries.shape)
# 11. Save cleaned dataset

df_countries.to_csv("cleaned_data.csv", index=False)
print("\nSaved cleaned_data.csv")
# 12. Select only Country Name, Year, and GDP

df_subset = df_countries[['Country Name', 'Year', 'GDP']]
print("\nCountry Name / Year / GDP (first 5):")
print(df_subset.head())

# 13. New column: GDP_in_billions
df_countries['GDP_in_billions'] = df_countries['GDP'] / 1e9
print("\nGDP_in_billions (first 3):")
print(df_countries[['Country Name', 'GDP', 'GDP_in_billions']].head(3))

# 14. New column: High_Population (pop > 100 million)
df_countries['High_Population'] = df_countries['Population'] > 100_000_000
print("\nHigh_Population value counts:")
print(df_countries['High_Population'].value_counts())

# 15. New column: category (Low / Medium / High GDP per capita)
def gdp_category(gdp_pc):
    if gdp_pc < 1000:
        return 'Low'
    elif gdp_pc < 10000:
        return 'Medium'
    else:
        return 'High'

df_countries['category'] = df_countries['GDP Per Capita'].apply(gdp_category)
# Alternative using pd.cut:
# df_countries['category'] = pd.cut(
#     df_countries['GDP Per Capita'],
#     bins=[-np.inf, 999.99, 9999.99, np.inf],
#     labels=['Low', 'Medium', 'High']
# )
print("\nCategory distribution:")
print(df_countries['category'].value_counts())

# 16. Sort by GDP descending

df_sorted_gdp = df_countries.sort_values('GDP', ascending=False)
print("\nTop 5 highest GDP countries:")
print(df_sorted_gdp[['Country Name', 'Year', 'GDP']].head())


# 17. Within each continent, sort by Population Density ascending

df_sorted_continent = df_countries.sort_values(
    ['Continent Name', 'Population Density'],
    ascending=[True, True]
)
print("\nSorted by Continent then Population Density (first 5):")
print(df_sorted_continent[['Country Name', 'Continent Name', 'Population Density']].head())


# 18. Average GDP per capita for each continent

avg_gdp_per_capita = df_countries.groupby('Continent Name')['GDP Per Capita'].mean()
print("\nAverage GDP Per Capita by Continent:")
print(avg_gdp_per_capita)
# 19. Number of records per country
records_per_country = df_countries.groupby('Country Name').size()
print("\nRecords per country (first 10):")
print(records_per_country.head(10))

# 20. Total education expenditure per continent in 2010

df_2010 = df_countries[df_countries['Year'].dt.year == 2010]
edu_2010 = df_2010.groupby('Continent Name')['education_expenditure'].sum()
print("\nTotal Education Expenditure by Continent (2010):")
print(edu_2010)

# 21. Mean, min, max of health_expenditure grouped by Continent

health_stats = df_countries.groupby('Continent Name')['health_expenditure'].agg(
    mean_health='mean',
    min_health='min',
    max_health='max'
)
print("\nHealth Expenditure stats by Continent:")
print(health_stats)

# 22. Average GDP Per Capita per continent where education_expenditure > 4%

avg_gdp_edu_filter = (
    df_countries[df_countries['education_expenditure'] > 4]
    .groupby('Continent Name')['GDP Per Capita']
    .mean()
)
print("\nAvg GDP Per Capita (education_expenditure > 4%) by Continent:")
print(avg_gdp_edu_filter)

# 23. Filter pop > 50M, add GDP_in_billions, top 5 GDP
#     + count unique countries in the full dataset

df_pop50 = df_countries[df_countries['Population'] > 50_000_000].copy()
df_pop50['GDP_in_billions'] = df_pop50['GDP'] / 1e9
top5_gdp = df_pop50.sort_values('GDP_in_billions', ascending=False).head(5)
print("\nTop 5 GDP (pop > 50M):")
print(top5_gdp[['Country Name', 'Year', 'GDP_in_billions']])

unique_countries = df_countries['Country Name'].nunique()
print("\nUnique countries in dataset:", unique_countries)


# 24. Distinct combinations of Country Name & Continent Name

distinct_country_continent = df_countries[['Country Name', 'Continent Name']].drop_duplicates()
print("\nDistinct Country-Continent combinations (first 5):")
print(distinct_country_continent.head())
print("Total:", len(distinct_country_continent))


# 25. Number of distinct years
distinct_years = df_countries['Year'].nunique()
print("\nDistinct years:", distinct_years)
print(sorted(df_countries['Year'].dt.year.unique()))

# 26. First 10 rows sorted by Population Density for year 2020

df_2020 = df_countries[df_countries['Year'].dt.year == 2020]
df_2020_sorted = df_2020.sort_values('Population Density')
print("\nTop 10 by Population Density (2020):")
print(df_2020_sorted[['Country Name', 'Population Density']].head(10))

# 27. Last 5 GDP entries for Nigeria

df_nigeria = df_countries[df_countries['Country Name'] == 'Nigeria']
print("\nLast 5 GDP entries for Nigeria:")
print(df_nigeria[['Country Name', 'Year', 'GDP']].tail(5))

# 28. Slice 3rd to 7th rows of South American countries
df_south_america = df_countries[df_countries['Continent Name'] == 'South America'].reset_index(drop=True)
print("\nSouth American countries — rows 3 to 7 (0-indexed 2:7):")
print(df_south_america.iloc[2:7])
