# File : Titanic.R
# Author : Sharanya Palavalli
# Date : 06/14/2017

# Step 0: Set your working directory for your convenience
setwd("D:\\Data Science\\Revision\\Beginner\\Titanic");

# Step 1 - Load data
trainTitanic <- read.csv("train.csv");
testTitanic <- read.csv("test.csv");

# Step 2 - Understand Data
# Description of variables available @ https://www.kaggle.com/c/titanic/data

# Display variables and understand their data types
str(trainTitanic)

# Examine what percentage of passengers survived  
prop.table(table(trainTitanic$Survived))

# Examining passengers' age
summary(trainTitanic$Age)

# Identifying children among passengers 
trainTitanic$Child <- 0
trainTitanic$Child[train$Age < 18] <- 1

# Examining data to see if female children had a higher chance to 
# survive compared to male children
aggregate(Survived ~ Child + Sex, data=trainTitanic, FUN=sum)
aggregate(Survived ~ Child + Sex, data=trainTitanic, FUN=length)
aggregate(Survived ~ Child + Sex, data=trainTitanic, FUN=function(x) {sum(x)/length(x)})

# Combine training and validation data sets for convenience
testTitanic$Survived <- NA
titanic <- rbind(trainTitanic, testTitanic)

# Step 3 - Check for missing values in all fields
summary(titanic)

# Impute missing values for Age
Agefit <- rpart(Age ~ Pclass + Sex + SibSp + Parch + Fare + Embarked + Title + FamilySize,
          data=titanic[!is.na(titanic$Age),], method="anova")

titanic$Age[is.na(titanic$Age)] <- predict(Agefit, titanic[is.na(titanic$Age),])

# Impute missing values for Embarked
summary(titanic$Embarked)
which(titanic$Embarked == '')

# 2 values for this field are missing
# Since majority of the passengers embarked from Southampton
# we imputed the missing values with S

titanic$Embarked[c(62,830)] = "S"
titanic$Embarked <- factor(titanic$Embarked)

# Examine missing values for Fare
summary(titanic$Fare)

# 1 value is missing
# Identify passenger whose fare details are missing
# Impute it with the median of this field

which(is.na(titanic$Fare))
titanic$Fare[1044] <- median(titanic$Fare, na.rm=TRUE)

# Step 4 - Feature Engineering

#Display few names
head(titanic$Name)

## As seen, The name is a combination of first name, last name and title
## Parse this field for both training and validation data sets

# Convert all names from factors to strings
titanic$Name <- as.character(titanic$Name)

# Extract the title from the name

## Split the name using delimiters , and .
strsplit(titanic$Name[1], split='[,.]')

##Access array of last name, title and first name
strsplit(titanic$Name[1], split='[,.]')[[1]]

## Reference second element in this array to retrieve title
strsplit(titanic$Name[1], split='[,.]')[[1]][2]

# Engineered variable: Title
titanic$Title <- sapply(titanic$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
titanic$Title <- sub(' ', '', titanic$Title)

# Inspect the new feature
table(titanic$Title)

# Combine small title groups
titanic$Title[titanic$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
titanic$Title[titanic$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
titanic$Title[titanic$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'

# Convert to a factor
titanic$Title <- factor(titanic$Title)

# Engineered variable: Family size
titanic$FamilySize <- titanic$SibSp + titanic$Parch + 1

# Engineered variable: Family
titanic$Surname <- sapply(titanic$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][1]})
titanic$FamilyID <- paste(as.character(titanic$FamilySize), titanic$Surname, sep="")
titanic$FamilyID[titanic$FamilySize <= 2] <- 'Small'

# Inspect new feature
table(titanic$FamilyID)

# Delete erroneous family IDs
famIDs <- data.frame(table(titanic$FamilyID))
famIDs <- famIDs[famIDs$Freq <= 2,]
titanic$FamilyID[titanic$FamilyID %in% famIDs$Var1] <- 'Small'

# Convert to a factor
titanic$FamilyID <- factor(titanic$FamilyID)

# Split back into test and train sets
trainNew <- titanic[1:891,]
testNew <- titanic[892:1309,]

#####################################

##### Approach 1 - Decision Trees

#####################################


# Install and load required packages for fancy decision tree plotting
install.packages('rpart')
install.packages('rattle')
install.packages('rpart.plot')
install.packages('RColorBrewer')

library(rpart)
library(rattle)
library(rpart.plot)
library(RColorBrewer)

# Build a new tree with our new features
dtreeModel <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
              data=trainNew, method="class")

fancyRpartPlot(dtreeModel)

# Generate predictions
predictSurvival <- predict(dtreeModel, testNew, type = "class")
forecast <- data.frame(PassengerId = testNew$PassengerId, Survived = as.numeric(predictSurvival)-1)

# Import Actuals for evaluating models performance
actuals <- read.csv("gender_submission.csv")

# Evaluate accuracy of model
dtreeAccuracy <- sum(forecast$Survived == actuals$Survived)/length(forecast$Survived)

#####################################

##### Approach 2 - Random Forest

#####################################

# Install and load required packages
library(rpart)

install.packages('randomForest')
library(randomForest)

install.packages('party')
library(party)

# New factor for Random Forests, only allowed <32 levels, so reduce number
titanic$FamilyID2 <- titanic$FamilyID

# Convert back to string
titanic$FamilyID2 <- as.character(titanic$FamilyID2)
titanic$FamilyID2[titanic$FamilySize <= 3] <- 'Small'

# And convert back to factor
titanic$FamilyID2 <- factor(titanic$FamilyID2)

# Split back into test and train sets
trainRf <- titanic[1:891,]
testRf <- titanic[892:1309,]

# Build Random Forest Model
set.seed(415)
rfModel <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID2,
           data=trainRf, importance=TRUE, ntree=2000)

# Examine importance of variables
varImpPlot(rfModel)

# Forecast using this model
rfResult <- predict(rfModel, testRf)
dfResult <- data.frame(PassengerId = testRf$PassengerId, Survived = rfResult)

rfAccuracy <- sum(dfResult$Survived == actuals$Survived)/length(dfResult$Survived)

# Approach b - Build a condition inference tree 
set.seed(415)
cfModel <- cforest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title + FamilySize + FamilyID,
           data = trainRf, controls=cforest_unbiased(ntree=2000, mtry=3))

# Forecast using condition inference tree
cfResult <- predict(cfModel, testRf, OOB=TRUE, type = "response")
dfResult1 <- data.frame(PassengerId = testRf$PassengerId, Survived = cfResult)

cfAccuracy <- sum(dfResult1$Survived == actuals$Survived)/length(dfResult1$Survived)

#####################################

##### Approach 3 - Logistic Regression

#####################################

library(stats)

logRegModel <- glm( Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + Title, data = trainRf, family = "binomial",maxit = 1000)
logResult <- predict(logRegModel, newdata = testRf, type = "response")

# Logistic regression computes probabilities depicting each passengers likelihood to survive
# As probabilities can take values between 0 and 1, we need to round them up

# Define user defined function roundUp 
roundUp <- function(x){
if (x > 0.5){
1
}
else
{
0
}
}

logResult <- lapply(logResult, roundUp)

# Compute Accuracy
logAccuracy <- sum(logResult == actuals$Survived)/length(logResult)

# Evaluating logistic regression model using ROCR package
library(ROCR)
anova(logRegModel, test="Chisq")

tr<-prediction(as.numeric(logResult),as.numeric(actuals$Survived))

# Plotting ROC curve
perf1 <- performance(tr, measure="tpr", x.measure="fpr")
plot(perf1)

# Computing Area under curve
perf2 <- performance(tr, "auc")


########## End of File ##########

