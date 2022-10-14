import numpy as np
import pandas as pd
from scipy import stats 
import statsmodels.api as sm 
from statsmodels.formula.api import ols
from statsmodels.graphics.factorplots import interaction_plot as meansPlot
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model 
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import KNNImputer


#Big Data Analysis Project Manjiri

#Cleaned up --> Removing commas and quotation marks in file

middleSchoolData = pd.read_csv("middleSchoolData.csv")

#Cleaning up the data
middleSchoolData.replace("", np.NaN)

#What is the correlation between the number of applications and admissions to HSPHS?

#Separating the data, getting the columns of interest and cleaning up the data
applicationsAcceptances = middleSchoolData[["applications","acceptances"]].dropna()
print(applicationsAcceptances.info())
#admissionsToHSPHS = middleSchoolData["acceptances"].dropna()

#Finding Pearson Correlation
pearsonCorrAA = applicationsAcceptances.corr(method='pearson')
print("Question 1:",0.801727)

import seaborn as sns

sns.regplot(x="applications", y="acceptances", data=applicationsAcceptances, scatter_kws={"color": "black"}, line_kws={"color": "red"}, ci=None)
plt.title("Scatter plot with Best Fit Line for Applications and Acceptances")
plt.xlabel('Number of Applications')
plt.ylabel('Number of Admissions')
applicationsAcceptances.plot.scatter(x='applications', y='acceptances', title= "Scatter plot for Applications and Acceptances");
#m, b = np.polyfit(x='applications', y='acceptances', 1) #m = slope, b=intercept
#plt.plot(x, m*x + b) #add line of best fit.
plt.show(block=True)

print()

#Dividing it by school size

#Cleaning up the data frame
applicationsApplicationsRateDF = middleSchoolData[["applications","school_size","acceptances"]].dropna()

applicationsApplicationsRateSeries = applicationsApplicationsRateDF["applications"]/applicationsApplicationsRateDF["school_size"]

correlationARAA = applicationsApplicationsRateSeries.corr(applicationsApplicationsRateDF["acceptances"])

sns.regplot(x = applicationsApplicationsRateSeries.to_numpy(), y = applicationsApplicationsRateDF["acceptances"].to_numpy(), scatter_kws={"color": "black"}, line_kws={"color": "red"}, ci=None)
plt.title("Scatter plot with Best Fit Line for Application Rate and Acceptances")
plt.xlabel('Application Rate')
plt.ylabel('Number of Admissions')
plt.show(block=True)

print("Question 2: Between Application Rates and Acceptances", correlationARAA**2)
print("Question 2: Between Applications and Acceptances", 0.801727**2)
print("Applications rather than application rate is a better predictor of admission to HSPHS!")

#Calculate Odds (acceptance rate / 1- acceptance rate)

#No missing data for either of the 3 columns of interest  --> proceeded with analysis

Odds = []

applicationsAcceptanceRateDF = middleSchoolData[["school_name","acceptances","applications"]].dropna()

#How many got in from those who applied
AcceptanceRate = applicationsAcceptanceRateDF['acceptances']/ applicationsAcceptanceRateDF["applications"]

for prob in AcceptanceRate:
    
    if type(prob) == float:
        odd = (prob) / (1 - prob)
        
        Odds.append(odd)
            
        #print(odd)
    else:
        Odds.append(prob)

#Add the Odds list to the data frame as a column
applicationsAcceptanceRateDF.insert(3,"Odds", Odds, True)

#Finding school with Best odds = highest odds
column = applicationsAcceptanceRateDF["Odds"]
max_value = column.max()
indexMax = column.idxmax()

print()
print("Question 3")
print("School with Best Odds: THE CHRISTA MCAULIFFE SCHOOL\I.S. 187")
print("Odds: 205:46 --> For every  251 students that apply, 205 students get in")
print()

#PCA

orig = middleSchoolData[["rigorous_instruction","collaborative_teachers","supportive_environment","effective_school_leadership","strong_family_community_ties","trust","student_achievement","reading_scores_exceed","math_scores_exceed"]]
PCAQ4 = middleSchoolData[["rigorous_instruction","collaborative_teachers","supportive_environment","effective_school_leadership","strong_family_community_ties","trust","student_achievement","reading_scores_exceed","math_scores_exceed"]].dropna()
only_na = orig[~orig.index.isin(PCAQ4.index)]

print(orig.isnull().sum()) #This tells us that student achievement has the most number of nans so for the other PC we delete the data accordingly
#whereAchievementNan = 

#Conducting PCA on how students percieve school

#Isolate the 6 predictors for student perception
studentsPerceptionPredictors = PCAQ4[["rigorous_instruction","collaborative_teachers","supportive_environment","effective_school_leadership","strong_family_community_ties","trust"]].to_numpy()

#Getting the correlation matrix for the predictors + Plotting Color Block
rStudentsQ4 = np.corrcoef(studentsPerceptionPredictors,rowvar=False)
plt.imshow(rStudentsQ4) 
plt.colorbar()
plt.show(block=True)

'''
# The variables are not uncorrelated. There is a correlation structure
# The correlation structure suggests that there will be 2 meaningful.
# factors. 1-3 are correlated (1 cluster) and 4-6 are correlated (2nd
# cluster), but they are not correlated between clusters.
# The intercorrelations in one cluster are slightly higher than in
# another, so we predict that eigenvalues in 1 are going to be slightly
# higher.

PCA is only helpful when most of the correlations are greater than 0.3 --> PCA must be conducted. Literally all the correlations in the plot are over 0.3.
PCA is only done on zscore.
'''

zScoredDataStudQ4 = stats.zscore(studentsPerceptionPredictors)
pca1 = PCA()
pca1.fit(zScoredDataStudQ4)
eigValues1 = pca1.explained_variance_  #How much variance is explained by each predictor
loadings1 = pca1.components_
rotatedData1 = pca1.fit_transform(zScoredDataStudQ4) #Rotated Data

#Scree Plot
plt.bar(range(1, len(eigValues1)+1),eigValues1)
plt.xlabel('Components')
plt.ylabel('Explained Variance')
plt.show(block=True);

plt.plot(range(1,len(eigValues1)+1),np.cumsum(eigValues1), c = "red", label = "Cumulative Explained Variance")
plt.legend(loc = 'upper left')
plt.show(block=True)

#Plotting Cumulative Distribution
numPredictors = 6
plt.plot(eigValues1)
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.plot([0,numPredictors], [1,1], color = "red", linewidth=1)
plt.show(block=True)

#Plotting Scree Plot 2
plt.bar(np.linspace(1,numPredictors,numPredictors),eigValues1)
plt.title('Scree plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalues')
plt.show(block=True)

print("Elbow Criteria: Principal Component 1")
print("Kaiser Criteria: Principal Component 0 (explains most variance)")

''' ADDITIONAL NOTESSS
print(loadings1[0,:])
print(loadings1[1,:])
'''

#Conducting PCA on how the school performs on objective measures of achievement
#Isolate the 3 predictors for school outcomes
schoolOutcomes = PCAQ4[["student_achievement","reading_scores_exceed","math_scores_exceed"]].to_numpy()


rSchoolsQ4 = np.corrcoef(schoolOutcomes,rowvar=False)
plt.imshow(rSchoolsQ4) 
plt.colorbar()
plt.show(block=True)

zScoredDataSchoolQ4 = stats.zscore(schoolOutcomes)
pca2 = PCA()
pca2.fit(zScoredDataSchoolQ4)
eigValues2 = pca2.explained_variance_  #How much variance is explained by each predictor
loadings2 = pca2.components_
rotatedData2 = pca2.fit_transform(zScoredDataSchoolQ4) #Rotated Data (getting new coordinates)

#Scree Plot
plt.bar(range(1, len(eigValues2)+1),eigValues2)
plt.xlabel('Component')
plt.ylabel('Explained Variance')
plt.show(block=True);

#Cumulative Explained Variance
plt.plot(range(1,len(eigValues2)+1),np.cumsum(eigValues2), c = "red", label = "Cumulative Explained Variance")
plt.legend(loc = 'upper left')
plt.show(block=True)

#Plotting Cumulative Distribution
numPredictors2 = 3
plt.plot(eigValues2)
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.plot([0,numPredictors2], [1,1], color = "red", linewidth=1)
plt.show(block=True)

#Plotting Scree Plot 2
plt.bar(np.linspace(1,numPredictors2,numPredictors2),eigValues2)
plt.title('Scree plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalues')
plt.show(block=True)

print("Elbow Criteria Outcome: Principal Component 1")
print("Kaiser Criteria Outcome: Principal Component 0 (explains most variance)")

#Plotting principal components
plt.bar(np.linspace(1,6,6),loadings1[0])
plt.title('Principal Component 1')
plt.xlabel('Variable')
plt.ylabel('Loading')
plt.show(block=True)

#Plotting principal components
plt.bar(np.linspace(1,3,3),loadings2[0])
plt.title('Principal Component 1')
plt.xlabel('School Outcome')
plt.ylabel('Loading')
plt.show(block=True)

#Finding the relationship between the chosen predictors and outcomes

#Running Simple Linear Regression 
corrcoefPCA = np.corrcoef(rotatedData1[:,0],rotatedData2[:,0])
sns.regplot(x=rotatedData1[:,0], y=rotatedData2[:,0], data=applicationsAcceptances, scatter_kws={"color": "black"}, line_kws={"color": "red"}, ci=None)
#plt.plot(rotatedData1[:,0],rotatedData2[:,0],'o',markersize=5)
plt.xlabel('Overall School Climate')
plt.ylabel('Overall Achievement')
plt.show(block=True)
print("Pearson correlation coefficient:", -0.367446)

#Running Multiple Linear Regression
'''
X = np.transpose([rotatedData1[:,0],rotatedData1[:,1]]) # PC1 and PC2
Y = rotatedData2[:,0] # income
regr = linear_model.LinearRegression() # linearRegression function from linear_model
regr.fit(X,Y) # use fit method 
rSqr = regr.score(X,Y) # - realistic - life is quite idiosyncratic
r = rSqr**1/2
betas = regr.coef_ # m
yInt = regr.intercept_  # b
'''
#CovarExplained 
covarExplained1 = eigValues1/sum(eigValues1*100)
covarExplained2 = eigValues2/sum(eigValues2*100)

#5 Null Hypothesis Test (T-test)

#IV: Charter vs Not Charter
#DV: Number of Admission

#Null hypothesis: There is no difference in HSPHS admissions between charter and non-charter schools.
#Alternate hypothesis: There is a difference in HSPHS admissions between charter and non-charter schools.


CharterSchools = middleSchoolData.loc[485:,['applications','acceptances']].dropna()
NonCharterSchools = middleSchoolData.loc[:485, ['applications','acceptances']].dropna()

#Finding Admissions Rate Instead of Just Number of Admissions (acceptances/applications)

CharterSchoolsAR = CharterSchools["acceptances"]/CharterSchools["applications"]
NonCharterSchoolsAR = NonCharterSchools["acceptances"]/NonCharterSchools["applications"]
CharterSchoolsAR.dropna(inplace = True)
NonCharterSchoolsAR.dropna(inplace = True)

pvaluesList = []
for i in range(1000):
#To make the sample size equal for both groups imma randomly select 105 values from Non-Charter Schools
    NonCharterSchoolsEqualized = NonCharterSchoolsAR.sample(n=105, replace = False)
    t_statisticQ5, p_valueQ5 = stats.ttest_ind(CharterSchoolsAR.to_numpy(),NonCharterSchoolsEqualized.to_numpy())
    pvaluesList.append(p_valueQ5);

#Draw Vertical line at x = 0.05

plt.hist(pvaluesList, density=False)
plt.ylabel('Count')
plt.xlabel('P-Value')
plt.title('P-Value Frequency')
plt.show(block=True)

print()
print("Question 5")
print("P-value for Independent t-test between charter and non-charter:", p_valueQ5)

#Find the correlation matrix 

relevantData = middleSchoolData[["per_pupil_spending","avg_class_size","acceptances"]].dropna()

corrMatrix = relevantData.corr()

#relevantData.plot.scatter(x='per_pupil_spending', y='student_achievement', title= "Scatter plot for Per Student Spending and Achievement");
relevantData.plot.scatter(x='per_pupil_spending', y='acceptances', title= "Scatter plot for Per Student Spending and Acceptances");
#relevantData.plot.scatter(x='avg_class_size', y='student_achievement', title= "Scatter plot for Class Size and Achievement");
relevantData.plot.scatter(x='avg_class_size', y='acceptances', title= "Scatter plot for Class Size and Acceptances");
plt.show(block=True)

#Rank, and find the top 90% (keeping addinng it when you hit the top 90%) (and bottom 90%) choose 1

totalAccepted = sum(middleSchoolData['acceptances'].to_numpy())

#Finding 90% of 4461  = 4014.9 = 4015

#Ranking the total accepted in each school

AcceptedDF = middleSchoolData[['school_name','acceptances']].dropna()

AcceptedDF.sort_values(by=['acceptances'], inplace=True, ascending=False)

count = 0
schools = 0
for accepted in AcceptedDF['acceptances']:
    
    if count < 4015:  
        count += accepted
        schools += 1
    else:
        break


    
AcceptedDF.plot.bar(y='acceptances')
x_ticks = [0,61,122,183,244,305,366,427,488]
plt.xticks(x_ticks, labels = [0,61,122,183,244,305,366,427,488])
plt.title('Number of Acceptances By School In Descending Order')
plt.xlabel('School (Index)')
plt.ylabel('Number of Acceptances')
plt.show(block=True)

AcceptedDF[:123].plot.bar(y='acceptances')
x_ticks = range(0,120,10)
plt.xticks(x_ticks)
plt.title('Acceptances Ranked By School Adding Up To 90% Of All Students Accepted')
plt.xlabel('School')
plt.ylabel('Acceptances')
plt.show(block=True)


print("Proportion:", schools/594, "(123/594)")

#All Predictors and DV (Sending students to HSPS --> Acceptance Rate)
race_predictors = middleSchoolData[["asian_percent","black_percent","hispanic_percent","multiple_percent","white_percent","acceptances"]].dropna() #.to_numpy()

corrMatrixBlackAceeptances = race_predictors["black_percent"].corr(race_predictors["acceptances"])

'''
#Conduct a PCA for Race Predictors
#2 rows were missing data so they just got dropped
race_predictors = middleSchoolData[["asian_percent","black_percent","hispanic_percent","multiple_percent","white_percent"]].dropna() #.to_numpy()

rRaceQ8 = np.corrcoef(race_predictors,rowvar=False)
plt.imshow(rRaceQ8) 
plt.colorbar()
plt.show(block=True)

zScoredrRaceQ8 = stats.zscore(race_predictors)
pcaRaceQ8 = PCA()
pcaRaceQ8.fit(zScoredrRaceQ8)
eigValuesRace = pcaRaceQ8.explained_variance_  #How much variance is explained by each factor
loadingsRace = pcaRaceQ8.components_
rotatedDataRace = pcaRaceQ8.fit_transform(zScoredrRaceQ8) #Rotated Data (getting new coordinates)

#Cumulative Explained Variance
plt.plot(range(1,len(eigValuesRace)+1),np.cumsum(eigValuesRace), c = "red", label = "Cumulative Explained Variance")
plt.legend(loc = 'upper left')
plt.show(block=True)

#Plotting Cumulative Distribution
numPredictorsRaceQ8 = 5
plt.plot(eigValuesRace)
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalue')
plt.plot([0,numPredictorsRaceQ8], [1,1], color = "red", linewidth=1)
plt.show(block=True)

#Plotting Scree Plot
plt.bar(np.linspace(1,numPredictorsRaceQ8,numPredictorsRaceQ8),eigValuesRace)
plt.title('Scree plot')
plt.xlabel('Principal Component')
plt.ylabel('Eigenvalues')
plt.show(block=True)

#Determining what the PCs mean (Kaiser PC1 and PC2 should be used)

#Viewing PC1
plt.bar(np.linspace(1,numPredictorsRaceQ8,numPredictorsRaceQ8),loadingsRace[0,:])
plt.title("PC1")
plt.show(block=True)

#Asian, Multiple and White --> Usually those with better jobs, more money

#Viewing PC2
plt.bar(np.linspace(1,numPredictorsRaceQ8,numPredictorsRaceQ8),loadingsRace[1,:])
plt.title("PC2")
plt.show(block=True)

#Hispanic and Blacks --> Usually those who are underprivileged/minorities/less opportunities, less-well off socioeconomically 

#sending students to HSPHS
#XQ8 = np.transpose(np.array([rotatedDataRace[:,0],rotatedDataRace[:,1]]))
#YQ8a = applicationsApplicationsRateDF["acceptances"] #Same Data is Missing


#achieving high scores on objective measures of achievement?
#YQ8b = rotatedData2[:,0]


import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
from scipy import stats
import statsmodels.formula.api as smf

middleSchoolData['acceptance_rate'] = middleSchoolData['acceptances']/middleSchoolData['applications']
middleSchoolData = middleSchoolData.iloc[:, 4:25]

middleSchoolData = middleSchoolData.dropna()

# PCA for percep (Q4)

percep = middleSchoolData.iloc[:, 7:13]
corr = np.corrcoef(percep, rowvar = False)
plt.imshow(corr)
plt.colorbar()

zscoredDataQ8percep = stats.zscore(percep)
pcaQ8percep = PCA()
pcaQ8percep.fit(zscoredDataQ8percep)

eigenValspercep = pcaQ8percep.explained_variance_

loadingspercep = pcaQ8percep.components_
rotated_dataPercep = pcaQ8percep.fit_transform(zscoredDataQ8percep)
covar_explainedPercep = eigenValspercep / sum(eigenValspercep) * 100

plt.figure()
numPredictorsPercep = 6
plt.bar(np.linspace(1,numPredictorsPercep,numPredictorsPercep), eigenValspercep)
plt.xlabel("Principal component")
plt.ylabel("Eigenvalue")
plt.plot([1,numPredictorsPercep], [1,1], color = 'red', linewidth = 0.7)
plt.show(bar = True)

which_principal_component = 0
plt.figure()
plt.bar(np.linspace(1,numPredictorsPercep,numPredictorsPercep), loadingspercep[:, which_principal_component])
plt.xlabel("Variable")
plt.ylabel("Loadings")
plt.show(bar = True)

PC1Percep = rotated_dataPercep[:,0]
middleSchoolData = middleSchoolData.drop(columns = ['rigorous_instruction', 'collaborative_teachers', 'supportive_environment', 'effective_school_leadership', 'strong_family_community_ties', 'trust'])
middleSchoolData['school_climate'] = PC1Percep #Creating a new column for the school climate according to students

# pca for diversity 

diversity = middleSchoolData.iloc[:, 2:10]
print(diversity)
corr = np.corrcoef(diversity, rowvar = False)

plt.imshow(corr)
plt.colorbar()

zscoredData = stats.zscore(diversity)
pca = PCA()
pca.fit(zscoredData)

eigenValsRace = pca.explained_variance_
loadingsDiversity = pca.components_
rotated_dataDiversity = pca.fit_transform(zscoredData)

covar_explainedRace = eigenValsRace / sum(eigenValsRace) * 100
plt.figure()

numPredictorsRace = 8
plt.bar(np.linspace(1, numPredictorsRace, numPredictorsRace), eigenValsRace)
plt.xlabel("Principal component")
plt.title("Scree Plot")
plt.ylabel("Eigenvalue")
plt.show(bar = True)

plt.plot([1, numPredictorsRace], [1,1], color = 'red', linewidth = 0.7)

which_principal_component = 0
plt.figure()
plt.bar(np.linspace(1, numPredictorsRace, numPredictorsRace), loadings[:, which_principal_component])
plt.xlabel('Variable')
plt.ylabel('Loadings')
plt.title("Principal Component 1")
plt.show(bar = True)

which_principal_component = 1 
plt.figure()
plt.bar(np.linspace(1, numPredictorsRace, numPredictorsRace), loadings[:, which_principal_component])
plt.xlabel('Variable')
plt.ylabel('Loadings')
plt.title("Principal Component 2")
plt.show(bar = True)

which_principal_component = 2 
plt.figure()
plt.bar(np.linspace(1, numPredictorsRace, numPredictorsRace), loadings[:, which_principal_component])
plt.xlabel('Variable')
plt.ylabel('Loadings')
plt.title("Principal Component 3")
plt.show(bar = True)

diversity = rotated_dataDiversity[:, 0] #Overall Diversity
disabilities = rotated_dataDiversity[: ,1]  #People with Disabilities
minorities = rotated_dataDiversity[: ,2] #Minorities (Black People and Hispanic)

data = data.drop(columns = ['asian_percent', 'black_percent', 'hispanic_percent', 'multiple_percent', 'white_percent', 'disability_percent', 'poverty_percent', 'ESL_percent'])
data['minorities'] = minorities
data['diversity'] = diversity
data['disabilities'] = disabilities

# pca for achievement (Q4)

achievementQ8 = middleSchoolData.iloc[:, 3:6]
print(achievementQ8)
corrAchievement = np.corrcoef(achievementQ8, rowvar = False)

plt.imshow(corrAchievement)
plt.colorbar()
plt.show(bar = True)

zscoredDataAchievementQ8 = stats.zscore(achievementQ8)
pca = PCA()
pca.fit(zscoredDataAchievementQ8)

eigenValsAchievement = pca.explained_variance_
loadingsAchievement = pca.components_
rotated_dataAchievement = pca.fit_transform(zscoredDataAchievementQ8)

covar_explainedAchievment = eigenValsAchievement / sum(eigenValsAchievement) * 100
plt.figure()

numPredictorsAchievement = 3
plt.bar(np.linspace(1, numPredictorsAchievement, numPredictorsAchievement), eigenValsAchievement)
plt.xlabel("Principal component")
plt.ylabel("Eigenvalue")
plt.show(bar = True)

plt.plot([1, numPredictorsAchievement], [1,1], color = 'red', linewidth = 0.7)

which_principal_component = 2 
plt.figure()
plt.bar(np.linspace(1, numPredictors, numPredictors), loadings[:, which_principal_component])
plt.xlabel('Question')
plt.ylabel('Loading')
plt.figure()

achievement = rotated_data[:,0]
data = data.drop(columns = ['student_achievement', 'reading_scores_exceed', 'math_scores_exceed'])

data['achievement'] = achievement

# mult regression

acc = data['acceptance_rate']
data = data.drop(columns = 'acceptance_rate')
data['acceptance_rate'] = acc

# with acceptance rate

x = data.iloc[:, 0:8]

y = data['acceptance_rate']

mod = smf.ols('y ~ x' , data = data)
res = mod.fit()

print(res.summary())

# with achievement

x = data.iloc[:, 0:7]

y = data['achievement']

mod = smf.ols('y ~ x' , data = data)
res2 = mod.fit()

print(res2.summary())
'''
