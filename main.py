import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import  numpy
from scipy import stats


data = pd.read_csv('cng514-covid-survey-data.csv')

def find_CentralTendency_AND_DispersionMeasures(column,column_name,column_data_type):

    if column_data_type == "nominal":
        mode_value = column.mode().iloc[0]

    elif column_data_type == "ordinal":
        mode_value = column.mode().iloc[0]
        median_value = column.median()
        quartiles = column.quantile([0.25, 0.50, 0.75]).to_dict()
        variance_value = column.var()


    elif column_data_type == "interval" or column_data_type == "ratio":
        mode_value = column.mode().iloc[0]
        median_value = column.median()
        mean_value = column.mean()
        max_value = column.max()
        min_value = column.min()
        range_value = max_value - min_value
        quartiles = column.quantile([0.25, 0.50, 0.75]).to_dict()
        variance_value = column.var()
        CreateBoxPlot(column=column,column_name=column_name)
        CreateHistogram(column=column,column_name=column_name)




    print(f"{column_name} Measures:")
    print("*********************************************************** ")
    if column_data_type == "nominal":
        print("Mode:", mode_value)
    elif column_data_type == "ordinal":
        print("Median:", median_value)
        print("Mode:", mode_value)
        print("Quartiles:\n", quartiles)
        print("Variance:", variance_value)

    elif column_data_type == "interval" or column_data_type == "ratio":
        print("Mode:", mode_value)
        print("Median:", median_value)
        print("Mean:", mean_value)
        print(f"Max value:{max_value}, Min value: {min_value}, and  Range is:", range_value)
        print("Variance:", variance_value)
        print("Quartiles:\n", quartiles)
    print("*********************************************************** ")
    print("\n")

def CreateBoxPlot(column,column_name):

    plt.figure(figsize=(8, 6))
    sns.boxplot(x=column)
    plt.title(f'Boxplot for {column_name}')
    plt.show()

def CreateHistogram(column,column_name):

    plt.figure(figsize=(8, 6))
    plt.hist(column, bins=20, edgecolor='black')
    plt.title(f'Histogram for {column_name}')
    plt.xlabel(f'{column} Values')
    plt.ylabel('Frequency')
    plt.show()


def CreateScatterPlot(x_column, y_column, x_column_name, y_column_name):
    plt.figure(figsize=(8, 6))
    plt.scatter(x_column, y_column)
    plt.title(f'Scatter Plot for {x_column_name} vs {y_column_name}')
    plt.xlabel(f'{x_column_name} Values')
    plt.ylabel(f'{y_column_name} Values')
    plt.show()

def trimmingOperation(data, lower_percentile=5, upper_percentile=95):

    lower_index = int(lower_percentile / 100 * len(data))
    upper_index = int(upper_percentile / 100 * len(data))

    trimmed_data = data[lower_index:upper_index]

    return trimmed_data

def drop_missing_values(data, column_name):

    non_missing_values = data[column_name].dropna()
    return non_missing_values

def calculate_Empty_values(data,columnName):
    return data[columnName].isnull().sum()


column_AnnualHouseholdIncome = 'AnnualHouseholdIncome'
resultFor_column_AnnualHouseholdIncome = find_CentralTendency_AND_DispersionMeasures(data[column_AnnualHouseholdIncome],column_AnnualHouseholdIncome,"ratio")

column_BirthYear2020 = 'BirthYear2020'
resultFor_column_BirthYear2020 = find_CentralTendency_AND_DispersionMeasures(data[column_BirthYear2020],column_BirthYear2020,"interval")

column_CoronavirusConcern2 = 'CoronavirusConcern2'
resultFor_column_CoronavirusConcern2 = find_CentralTendency_AND_DispersionMeasures(data[column_CoronavirusConcern2],column_CoronavirusConcern2,"ordinal")



CreateScatterPlot('AnnualHouseholdIncome', 'BirthYear2020')
CreateScatterPlot('AnnualHouseholdIncome', 'CoronavirusConcern2')
CreateScatterPlot('CoronavirusConcern2', 'BirthYear2020')



missing_values_for_annual_household_income = calculate_Empty_values(data, 'AnnualHouseholdIncome')
missing_values_for_birth_year_2020 = calculate_Empty_values(data, 'BirthYear2020')
missing_values_for_coronavirus_concern_2 = calculate_Empty_values(data, 'CoronavirusConcern2')

print("Missing Values:")
print(f"AnnualHouseholdIncome: {missing_values_for_annual_household_income}, BirthYear2020: {missing_values_for_birth_year_2020}, CoronavirusConcern2: {missing_values_for_coronavirus_concern_2}")




non_missing_values_For_Annual_income = drop_missing_values(data,'AnnualHouseholdIncome')
non_missing_values_For_BirthYear2020 = drop_missing_values(data,'BirthYear2020')
non_missing_values_For_CoronavirusConcern2 = drop_missing_values(data,'CoronavirusConcern2')

print("After delete missing values:")
print(f"AnnualHouseholdIncome: {non_missing_values_For_Annual_income.isnull().sum()}, BirthYear2020:  {non_missing_values_For_BirthYear2020.isnull().sum()}, CoronavirusConcern2: {non_missing_values_For_CoronavirusConcern2.isnull().sum()}")


sorted_annual_income = non_missing_values_For_Annual_income.sort_values()
processed_annualincome = trimmingOperation(sorted_annual_income)



sorted_BirhYear2020 = non_missing_values_For_BirthYear2020.sort_values()
processed_BirhYear2020 = trimmingOperation(sorted_BirhYear2020)


ages = 2020 - numpy.array(processed_BirhYear2020)

normalized_ages = pd.Series(ages)



processed_CoronavirusConcern2 = data[(data['CoronavirusConcern2'] >= 0) & (data['CoronavirusConcern2'] <= 10)]['CoronavirusConcern2'].sort_values()



CreateHistogram(processed_CoronavirusConcern2,'CoronavirusConcern2')


num_bins = 10
binned_data, bin_edges = numpy.histogram(processed_annualincome, bins=num_bins)
binned_Annunal = numpy.repeat(bin_edges[:-1], binned_data)


binned_data, bin_edges = numpy.histogram(processed_CoronavirusConcern2, bins=num_bins)
binned_CoronavirusConcern2 = numpy.repeat(bin_edges[:-1], binned_data)

binned_data, bin_edges = numpy.histogram(processed_BirhYear2020, bins=num_bins)
binned_BirhYear2020 = numpy.repeat(bin_edges[:-1], binned_data)



bin_edges = numpy.linspace(processed_annualincome.min(), processed_annualincome.max(), num_bins + 1)

plt.figure(figsize=(10, 6))
counts, bins, _ = plt.hist(processed_annualincome, bins=bin_edges, edgecolor='black')
plt.title('Equal-width Binning for Annual Household Income')
plt.xlabel('Annual Household Income')
plt.ylabel('Frequency')
plt.show()



bins = [0, 3, 6, numpy.inf]
labels = ['Not at all', 'Somewhat', 'Extremely concerned']

hist, _ = numpy.histogram(processed_CoronavirusConcern2, bins=bins)

plt.bar(labels, hist, edgecolor='black')

plt.xlabel('Concern Level')
plt.ylabel('Frequency')
plt.title('Histogram of Coronavirus Concern Level')

plt.show()


currentAge = [2020 - int(i) for i in processed_BirhYear2020]

age_bins = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
age_labels = [f"{age}-{age+10}" for age in age_bins[:-1]]
age_labels.append(f"{age_bins[-2]}+")


plt.hist(currentAge, bins=age_bins, edgecolor='black')
plt.xlabel('Age Range')
plt.ylabel('Number of Individuals')
plt.title('Age Distribution')

plt.xticks(age_bins, age_labels)
plt.grid(True)
plt.show()


#
concern_data = data['AnnualHouseholdIncome']
discretize_Concern = []

for i in processed_annualincome:
    if i < 100000:
        discretize_Concern.append(1)
    elif i < 200000:
        discretize_Concern.append(2)
    elif i < 300000:
        discretize_Concern.append(3)
    elif i < 400000:
        discretize_Concern.append(4)
    elif i < 500000:
        discretize_Concern.append(5)


hist, bins, _ = plt.hist(discretize_Concern, bins=[ 1, 2, 3,4,5,6], edgecolor='black')
plt.xlabel('AnnualHouseholdIncome Level')
plt.ylabel('Number of Individuals')
plt.title('Annual Household Income')

plt.xticks(bins, ['0', '100000', '200000', '300000','400000','500000'])


plt.grid(True)
plt.show()


from scipy.stats import spearmanr
from scipy.stats import pearsonr
def max_min_normalization(series):

    min_val = series.min()
    max_val = series.max()
    normalized_series = (series - min_val) / (max_val - min_val)
    return normalized_series


def z_score_normalization(series):

    z_scores = stats.zscore(series)
    normalized_series = pd.Series(z_scores, index=series.index)
    return normalized_series


normalized_AnnualIncome = max_min_normalization(data['AnnualHouseholdIncome'])
normalized_CoronaConcern2 =max_min_normalization(data['BirthYear2020'])


# Convert numpy arrays to Pandas Series
normalized_AnnualIncome_series = pd.Series(normalized_AnnualIncome)
normalized_age_series = pd.Series(currentAge)
normalized_age = z_score_normalization(normalized_age_series)

normalized_CoronaConcern2 = pd.Series(normalized_CoronaConcern2)

def spearman_correlation_with_p(x, y):

    spearman_corr, p_value = spearmanr(x, y)
    return spearman_corr, p_value

def pearson_correlation_with_p(x, y):

    pearson_corr, p_value = pearsonr(x, y)
    return pearson_corr, p_value


correlation_3, p_value_3 = spearman_correlation_with_p(normalized_AnnualIncome_series, normalized_CoronaConcern2)
print("Spearman Correlation Coefficient:", correlation_3)
print("P Value:", p_value_3)


correlation_pearson, p_value_pearson = pearson_correlation_with_p(normalized_AnnualIncome_series, normalized_age_series)
print("Pearson Correlation Coefficient:", correlation_pearson)
print("P Value:", p_value_pearson)