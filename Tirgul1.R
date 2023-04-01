### ML- TUTORIAL 1- Intro ###
## CARET package----------------------------------------------------------------------------------

# We will also use "caret": 
library(caret)

# The caret package (short for Classification And Regression Training) is a set of functions that 
# attempt to streamline the process for creating predictive models. The package contains tools for:

# - data splitting
# - pre-processing
# - model tuning using resampling
# - model fitting and prediction

# (see: https://topepo.github.io/caret/)

# THE DATA- Auto & THE PROBLEM\ QUESTION- regression : ----------------------------------------------

# For datasets we will mostly use the Book official package ISLR
library(ISLR) #The "ISLR" package includes data sets we will use this semester

# The Auto Dataset contains information about ... cars.
# For each car, the following vars were recorded:
#  - cylinders
#     Number of cylinders between 4 and 8
# - displacement
#     Engine displacement (cu. inches)
# - horsepower
#     Engine horsepower
# - weight
#     Vehicle weight (lbs.)
# - acceleration
#     Time to accelerate from 0 to 60 mph (sec.)
# - year
#     Model year (modulo 100)
# - origin
#     Origin of car (1. American, 2. European, 3. Japanese)
Auto$origin <- factor(Auto$origin)
# - name
#     Vehicle name

# What we are interested is gas consumption: MPG (miles per gallon)

# Getting to know the data:
?Auto
dim(Auto)
names(Auto)
head(Auto)
str(Auto)
dplyr::glimpse(Auto)

# Data Splitting - Train & Test Data------------------------------------------------------------

# We will TRAIN the model (i.e. fit) on the 70% of the observations randomly assigned
# and TEST the model (i.e. predict and assess performance) on the 30% that were left

# We can use createDataPartition() function from caret. we will use simple splitting based on the outcome
# More examples of data splitting (e.g., based on predictors):
# # https://topepo.github.io/caret/data-splitting.html

set.seed(1) # because we will use random sampling we need to set a 
# random seed in order to replicate the results

indxTrain <- createDataPartition(y = Auto$mpg, p = 0.7, list = FALSE)
train.data <- Auto[indxTrain,] # We will train the model using 70% of the obs.
test.data <- Auto[-indxTrain,] # We will test the model on 30% of the obs.


# The general process for all types of questions (regression\ classification) and fitting methods will be:

# (A) Fitting >>> (B) Predicting >>> (C) assessing performance

## (A) Model FITTING using train() on the TRAIN data:-------------------------------------------------------------

# For the fitting process we will use the train() function from caret which train a model
# using a variety of functions (this time- KNN)

# But one small thing before....
set.seed(1)
# We set a random seed because there is a resampling process which is inherent to KNN 
# method - the process of searching for the neighbors. 
# If several observations are tied as nearest neighbors, 
# R will randomly break the tie. (you don't have to understand this part)
# Therefore, a seed must be set in order to ensure reproducibility of results.


# lets' explore train():

tg <- expand.grid(k = 5)
# (An OPTIONAL\MUST argument depends on the method used)
# A data frame with possible tuning values (also called hyper-parameters).
# For knn we MUST enter the k. *
# For now we will use only one k for the fitting procedure 
# (next week we will see that we will be able to test for different hyper-parameters). 

tc <- trainControl(method = "none")
# (an OPTIONAL argument- option which we usually use for ML)
# method of controlling the computational process of fitting.
# For now, we used "none" (no training method)
#  i.e., now this argument does nothing.
# Next week it will mean a lot!

knn.fit5 <- train(
  mpg ~ horsepower + weight, # model syntax (a MUST argument)
  data = train.data, # the data (a MUST argument)
  method = "knn", # method used for fitting the model (now- knn)*
  tuneGrid = tg,
  trControl = tc,
  preProcess = c("center", "scale") 
  # (an OPTIONAL argument)
  # There several functions to pre-process the predictor data 
  # (i.e. prepare the data for fitting process)
  # one very common function is for centering and scaling the predictors
  # for knn this is VERY IMPORTANT- Because KNN method identify the observations 
  # according to their distance, the scale of the variables matters: 
  # large scale-> larger effect on the distance between the observations, 
  # We may standardize data prior to fitting OR use this argument
  # (See lesson for why we DONT do this directly on the train-data)
)

# Here are all available fitting methods within train(): https://topepo.github.io/caret/available-models.html
# Note- for each model you can check for each TYPE of question it is used and what are the tuning parameters.




## (B) PREDICTING using predict() - using the fitted model for prediction on the TEST data:----------------------

# predict() function uses the fitted model + given values of the predictors to predict the probability Y. 

predicted_mpg <- predict(knn.fit5,  # fitted model used for prediction
                        newdata = test.data)  # the data to predict from (i.e.,
# values for the PREDICTORS will be taken from the TEST data)


plot(test.data$mpg, predicted_mpg) # true vs predicted values



## (C) Assessing model performance on the TEST data ---------------------------------------------------------------------

# How we assess model performance? 
# For regression problems- R-squared, MSE, RMSE, MAE...

c(
  Rsq = cor(test.data$mpg, predicted_mpg)^2,
  MSE = mean((test.data$mpg - predicted_mpg)^2),
  RMSE = sqrt(mean((test.data$mpg - predicted_mpg)^2)),
  MAE = mean(abs(test.data$mpg - predicted_mpg))
)



## Playing with K --------------------------------------------------------------------------------------------------

# We chose K=5, but what will happen if we will use bigger k? like 10?

# (A) Fitting the same model using KNN (with k=10) on the train data:
tg2 <- expand.grid(k = 10)

knn.fit10 <- train(mpg ~ horsepower + weight,
                   data = train.data,
                   method = "knn", 
                   tuneGrid = tg2,
                   trControl = tc,
                   preProcess = c("center", "scale") 
)

# (B) PREDICTING for the test data
predicted_mpgK10 <- predict(knn.fit10,newdata = test.data)

# (C) ASSESSING preformance
c(
  Rsq = cor(test.data$mpg, predicted_mpgK10)^2,
  MSE = mean((test.data$mpg - predicted_mpgK10)^2),
  RMSE = sqrt(mean((test.data$mpg - predicted_mpgK10)^2)),
  MAE = mean(abs(test.data$mpg - predicted_mpgK10))
)
# Gives very similar results to K=5...



## Playing with the model itself -----------------------------------------------------------------

# Well, maybe the problem is that 
# mpg ~ horsepower + weight,
# is just not a good model....

# That make sense as I know nothing about cars...
# (hopefully for our research questions we use more thinking)

# Let's see if a different model (using KNN method with k=10), will preform better.
# Let's try to use all the 8 available predictors in the data.
# We can list all of them, or just use: Outcome ~ . 

# (A) Fitting the model:
knn.fit10.B <- train(mpg ~ . - name, # all predictors, EXPECT for the name of the model (meaningless!)
                     data = train.data,
                     method = "knn", 
                     tuneGrid = tg2,
                     trControl = tc,
                     preProcess = c("center", "scale") 
)

# (B) PREDICTING for the test data
predicted_mpgK10.B <- predict(knn.fit10.B, newdata = test.data)

library("tidyverse")
test.data |>
  mutate(pred = predicted_mpgK10.B) |>
  ggplot(aes(x = mpg, y = pred)) +
  geom_point() +
  geom_smooth(method = "lm") +
  theme_classic()


# (C) ASSESSING preformance
c(
  Rsq = cor(test.data$mpg, predicted_mpgK10.B)^2,
  MSE = mean((test.data$mpg - predicted_mpgK10.B)^2),
  RMSE = sqrt(mean((test.data$mpg - predicted_mpgK10.B)^2)),
  MAE = mean(abs(test.data$mpg - predicted_mpgK10.B))
)

# Much better!

# Exercise 1 ---------------------------------------------------------------------

## New Data!
# Wage dataset from ISLR includes 3000 obs with 10 predictors 
?Wage

# We want to predict `wage`.


## Your task:
library("tidyverse")

# Note- when needed- use set.seed(1) for replicability
set.seed(14)

# (1) Split the data to train and test (use p=0.7)
trainIndices <- createDataPartition(Wage$wage, p = 0.7, list = F)
train_data <- Wage[trainIndices,]
test_data <- Wage[-trainIndices,]

# (2) Predict wage out 3 of the other variables with a knn model with 
#     (k=5). That is, fit and predict.
tunegrid <- expand.grid(k = 3)

knn_3 <- train(wage ~ age + race + education,
               data = train_data,
               method = "knn",
               tuneGrid = tunegrid,
               trControl = trainControl(method = "none"),
               preProcess = c("center", "scale"))

predicted_wage <- predict(knn_3, newdata = test_data)

# (3) Assess performance using the metrics you've learned
c(
  Rsq = cor(test_data$wage, predicted_wage)^2,
  MSE = mean((test_data$wage - predicted_wage)^2),
  RMSE = sqrt(mean((test_data$wage - predicted_wage)^2)),
  MAE = mean(abs(test_data$wage - predicted_wage))
)

test_data |>
  mutate(pred = predicted_wage) |>
  ggplot(aes(x = wage, y = predicted_wage)) +
  geom_point() +
  theme_bw()
# (4) To improve flexibility, try a different k. Will you use bigger\ smaller k?
tunegrid <- expand.grid(k = 50)

knn_50 <- train(wage ~ age + race + education,
               data = train_data,
               method = "knn",
               tuneGrid = tunegrid,
               trControl = trainControl(method = "none"),
               preProcess = c("center", "scale"))

predicted_wage <- predict(knn_50, newdata = test_data)

c(
  Rsq = cor(test_data$wage, predicted_wage)^2,
  MSE = mean((test_data$wage - predicted_wage)^2),
  RMSE = sqrt(mean((test_data$wage - predicted_wage)^2)),
  MAE = mean(abs(test_data$wage - predicted_wage))
)

test_data |>
  mutate(pred = predicted_wage) |>
  ggplot(aes(x = wage, y = predicted_wage)) +
  geom_point() +
  theme_bw()

# (5) If you tried smaller k try now bigger k (or vice versa). what will you earn from
#     this and what will you loose? (in terms of performance indices)
tunegrid <- expand.grid(k = 2)

knn_2 <- train(wage ~ age + race + education,
                data = train_data,
                method = "knn",
                tuneGrid = tunegrid,
                trControl = trainControl(method = "none"),
                preProcess = c("center", "scale"))

predicted_wage <- predict(knn_2, newdata = test_data)

c(
  Rsq = cor(test_data$wage, predicted_wage)^2,
  MSE = mean((test_data$wage - predicted_wage)^2),
  RMSE = sqrt(mean((test_data$wage - predicted_wage)^2)),
  MAE = mean(abs(test_data$wage - predicted_wage))
)

test_data |>
  mutate(pred = predicted_wage) |>
  ggplot(aes(x = wage, y = predicted_wage)) +
  geom_point() +
  theme_bw()

