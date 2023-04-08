### ML- TUTORIAL 2 - classification and CV ###

# Load packages:
library(dplyr)
library(ISLR)
library(caret) 
library(yardstick)

# Warming up- a classification problem with logistic regression-------------------------------------------

# In Tutorial 1 we've used caret for a regression problem.
# Today we are looking at classification with the Smarket dataset:
# Smarket (from ISLR) consists of percentage returns for the a stock index over 1,250 days.
# For each date, the following vars were recorded:
# Lag1 through Lag5 - percentage returns for each of the five previous trading days.
# Volume- the number of shares traded on the previous day(in billions).
# Today- the percentage return on the date in question.
# Direction- whether the market was Up or Down on this date.

# Assume the following classification task on the Smarket data:
# predict Direction (Up/Down) using the features Lag1 and Lag2.
# If we are not sure how Direction is coded we can use contrasts():
contrasts(Smarket$Direction)

table(Smarket$Direction)
# The base rate probability:
648/(648+602)

## We'll start by using a parametric method - logistic regression:

# Data Splitting (SAME as in tutorial 1):
set.seed(1) 
Smarket.indxTrain <- createDataPartition(y = Smarket$Direction,p = 0.7, list = FALSE)
Smarket.train <- Smarket[Smarket.indxTrain,] 
Smarket.test <- Smarket[-Smarket.indxTrain,] 

## (A) Fitting logistic regression on train data using caret:
tc <- trainControl(method = "none") # remember "none" = no training method.
# for the warm-up we will leave it like that...

# We will use train().
LogRegfit <- train(
  Direction ~ Lag1+Lag2, # model syntax
  data = Smarket.train, # the data
  method = "glm", family = binomial("logit"), # For logistic regression
  trControl = tc, # although this is redundant when set to "none"
  metric = "Accuracy" # NOTE 1: this specify the summary metric will be used to select
  # the optimal model. By default, possible values are "RMSE" and "Rsquared" 
  # for regression and "Accuracy" and "Kappa" for classification.
  # For now, since no training method was chosen- this argument is currently meaningless
  # Anyway, note that for Accuracy caret uses a threshold of 0.5 by default!
)

# NOTE 2- here we don't have a tuneGrid with possible hyper-parameters as these are not needed for logistic
#       regression. We also don't use any preProcess arguments (as we used it for scaling in knn), since 
#       our simple straightforward logistic regression don't require any pre processing.

# NOTE 3- yes, not using any training method would leave us with the same results as if we've used a simple
#         glm() function with "binomal" argument.
# This is how we would do this using regular R method, 
# LogRegfit <- glm(Direction ~ Lag1+Lag2,
#                  data = Smarket.train,
#                  family = binomial("logit"))
# But... We want the power of {caret}!

# You _can_ interpret the model coefficients using exp just as you've learned in
# semester A. But, let's NOT, and instead focus on the process in the ML
# course...

## (B) Prediction using predict() :
predicted.probs <-predict(LogRegfit,newdata = Smarket.test,type="prob")
# output probabilities of the form p(Y = 1|X)->  p(direction is UP given the specific Xs).
# This is relevant for classification problems of course...
head(predicted.probs)  # predicted outcomes for the first 6 obs.
predicted.probs[1:10,2] # probability for the market to go up for the 10 first observations

# Here we predicted probabilities, but what if we want to predict classes?
# use "raw" instead of "prob" and get the class prediction (based on 0.5 cutoff)
predicted.classes <-predict(LogRegfit, newdata = Smarket.test,type="raw") 
# raw is the default so this will give us the same:
predicted.classes <-predict(LogRegfit, newdata = Smarket.test) 

predicted.classes[1:10] # predicted.classes for the market to go up for the 10 first observations

# OR, if from some reason we don't want to use the 0.5 cutoff, we can convert the predicted probabilities
# to a binary variable based on selected cutoff criteria.
# E.g. for making a binary decision of >0.55 
predicted.classes2 <- factor(predicted.probs[2]>0.55, labels = c("Down", "Up")) # [2] is in order to take only "Up" column
# as columns complete to 1.
predicted.classes2[1:10]


## (C) Assessing model performance

# How we assess model performance? 
# For regression problems- MSE, RMSE...
# For classification problems- performance indices based on the confusion matrix
confusionMatrix(predicted.classes,       # The predicted classes 
                Smarket.test$Direction,  # The Reference classes (Real classes) 
                positive = "Down")
# Note- 'Positive' Class : Down
#       That is, "Hit"\"TP" will be- saying the market is Down when it is really Down
#       an "False alarm"\"FP" will be- saying the market is Down when it is really Up

# It seems that it will be more intuitive to look at "Up" as positive, we can flip it using
# "positive" argument:
confusionMatrix(predicted.classes, Smarket.test$Direction,
                positive = "Up") 

# For example,  
# One performance indices is Accuracy (ACC)- 
# ratio of correct predictions (fraction of days for which the prediction was correct):
(30 + 156 )/374
# That is- TEST ERROR is 1-0.4973262=0.5026738

# Also- sensitivity, recall, hit rate, or true positive rate (TPR) are all the same names for
(156)/(38+156)
# https://en.wikipedia.org/wiki/Confusion_matrix for more about terminology and derivations
# of a confusion matrix

# ANYWAY, all indices tells us that this model wasn't that amazing (for accuracy flipping a coin 
# is similar...)

## We can also look at the ROC curves!

# for creating ROC curve we need to create a column of the predicted probs for our "positive"
# class ("Up") within the test data:
Smarket.test$prob_logisticReg <- predict(LogRegfit, newdata = Smarket.test, type = "prob")[,"Up"]

# we will use yardstick pack.
library(yardstick)

roc_curve(Smarket.test, Direction, prob_logisticReg, event_level = "second") %>%  ggplot2::autoplot()

# Here we can see how our modeled classifier acts (in terms of
# True Positive Rate and False Positive Rates) using different thresholds.
# It seems that for some varied thresholds our classifier isn't much better than a random classifier.


# Cross-Validation exemplified on KNN model ---------------------------------------------------------

# We will show how Cross-Validation can help us choose the hyper-parameter k for KNN.

# Let's take another CLASSIFICATION problem on this data. but with few changes:
# 1. change the model itself- predict Direction from all predictors!
# 2. change the method- KNN (with binary outcome) instead of logistic regression
# 3. TUNE THE MODEL WHILE FITTING. That is- use CV. Specifically, use CV to also choose the best K.


## Leave-One-Out Cross-Validation- LOOCV ------------------------------------------------------------

# One type of CV is LOOCV.

# For N obs. we will re-sample the data n times and for each sample:
# n-1 obs. will be the training set, and the one left out obs. will be the validation set.

# How?
# Finally using "trainControl"!
# The LOOCV estimate can be automatically computed for any fitted model with train().
# The cool thing about caret is that it enables us to use the "trainControl"
# argument when fitting any model, so we don't need to use any other packages that
# are specific for preforming CV to each type of models (e.g. boot).

## (A) FITTING a KNN model using LOOCV 

tc <- trainControl(method = "LOOCV") # remember we used "none" in the previous examples?
# that meant we told R 'don't use resampling when fitting the data'
# now we tell R to do resampling, and specifically- LOOCV
# We can just use this fitting method to make the fitting procedure 
# to be more reliable (for any chosen k), or, we can also make use 
# of it in order to CHOOSE the best K (i.e., hyperparameter)!

# Let's try now some options for k to better understand:
tg <- expand.grid(k = c(2, 5, 10, 50, 200)) 
tg # (since we have only 1 hyper-parameter, this is only a vector for now)

set.seed(1)  # For KNN fitting process (see tutorial 1)

knn.fit.LOOCV <- train(
  Direction ~ .,                     # our new model
  data = Smarket.train, 
  method = "knn",                    # knn instead of logistic regression
  tuneGrid = tg,                     # our K's for KNN
  trControl = tc,                    # the method for tuning (LOOCV)
  preProcess = c("center", "scale"), # don't forget scaling for KNN... (see tutorial 1)
  metric = "Accuracy"
)

#(notice the time it takes to fit using LOOCV)

knn.fit.LOOCV$results # Here we see accuracy for all values of K
# 1- accuracy will be the VALIDATION ERROR
# (it is somewhat in-between train and test errors)
# *Kappa is a metric that compares an Observed Accuracy with an 
# Expected Accuracy (random chance).  

val.error<- 1-knn.fit.LOOCV$results$Accuracy
plot(knn.fit.LOOCV$results$k,val.error)
knn.fit.LOOCV$bestTune 

# Best K is k=50 , where Accuracy= 0.87899 and the validation error is
# 1-0.899= 0.101

# Final model is automatically chosen based on the best tuning hyperparameter(s)
# (for now only one hyper-parameter - k)

# final Model is chosen during training and is set to k=50
knn.fit.LOOCV$finalModel

## (B) PREDICTING on test data:
predicted.classes.LOOCV <-predict(knn.fit.LOOCV, newdata = Smarket.test, type="raw") 

## (C) Assessing performance: 
confusionMatrix(predicted.classes.LOOCV,Smarket.test$Direction, positive = "Up")
# Accuracy (0.91) is great, even in contrast to the validation error! 
# It might be that our model is just good\or that CV help us to create a stable
# model (also, remember that accuracy isn't everything and we have many
# other fit indices that deserve attention) 

# roc curve:
Smarket.test$prob_KNN5 <- predict(knn.fit.LOOCV, newdata = Smarket.test, type = "prob")[,"Up"]
roc_curve(Smarket.test, Direction, prob_KNN5, event_level = "second")  %>%  ggplot2::autoplot()
#pretty!!!

## k-Fold Cross-Validation -----------------------------------------------------------------

# LOOCV might be time consuming, and K-fold CV can give us very similar results...

# In K-folds CV we split data into k parts and re-sample these parts k times.
# For each round of sampling k-1 parts are used to train the model, 
# and one left out part of the data remains as the validation set.
# let's try k = 10 (a common choice for k).

## (A) FITTING a KNN model using 10-folds CV:

tc <- trainControl(method = "cv", number = 10)
# for preforming k-Fold Cross-Validation just switch "trainControl" to:
# trainControl(method = "cv", number = k)

# Let's try again the same options for knn's k:
tg

set.seed(1)  # For KNN fitting process, but also-
# we must set a random seed for, since the obs. are sampled into 
# the one of the k folds randomly 

knn.fit.10CV <- train(
  Direction ~ ., 
  data = Smarket.train, 
  method = "knn",
  tuneGrid = tg,
  trControl = tc,
  preProcess = c("center", "scale"), 
  metric = "Accuracy"
)

# You might have not noticed but the computation time is much shorter than that of LOOCV.
# With more complex models it will be more aparent.

knn.fit.10CV$finalModel # The best tuning parameter based on 10-folds cv is again 50!
knn.fit.10CV$results # we can see mean accuracy (across all folds for a given
# k neighbors, as well as Accuracy SD)
# Validation error: 1-0.8847179= 0.11

## (B) PREDICTING on test data:
predicted.classes.10CV <-predict(knn.fit.10CV, newdata = Smarket.test, type="raw") 

## (C) Assessing preformance: 
confusionMatrix(predicted.classes.10CV,Smarket.test$Direction, positive = "Up")
# ACC on test set based on the 50-nearest neighbor model that was fitted using the
# 10-fold CV is: 0.9037
# Test error: 1-0.9037= 0.096
# which is at least as good as the LOOCV fit...


# IMPORTANT NOTE!
# THE BEST is to use LOOCV\ K-folds CV when fitting the train data, 
# and to test the fitted model with new test data.

# BUT, for very small samples and\or for very low base rates we might
# choose to not split to train and test data sets so we would be able
# to utilize all available data. Still, using CV while fitting will 
# help us getting a more reliable model and to choose hyper-parameters.



# Exercises: ----------------------------------------------------------------------

# Note- * when needed- use set.seed({some number}) for replicability
#       * work with split to train =70% of the data (and test= 30%)

# A) Use the Smarket dataset and predict Direction from Lag1 + Lag2 + Lag3.
#    But now fit a *logistic regression* using 10-folds CV and assess performance on test data.
#    what are the cv error and test error?

# Training
train_control_glm <- trainControl(method = "cv", number = 10)

logistic_reg <- train(Direction ~ Lag1 + Lag2 + Lag3,
                      data = Smarket.train,
                      method = "glm", family = "binomial",
                      trControl = train_control_glm,
                      metric = "Accuracy")

# CV error
logistic_reg$results

# Test error
predicted_classes <- predict(logistic_reg, newdata = Smarket.test, type = "raw") 

confusionMatrix(predicted_classes, Smarket.test$Direction, positive = "Up")



# B) Fit knn while choosing the best K with 10-folds CV, use the Caravan dataset:

# Caravan dataset includes 85 predictors that measure demographic characteristics 
# for 5,822 individuals. The response variable (column 86) is:
# Purchase- whether or not a given individual purchases a caravan insurance policy. 
Caravan$Purchase # Purchase is a factor with 2 levels "No","Yes"
str(Caravan) # all other variables are numeric
psych::describe(Caravan)
# In this task we will predict Purchase out this variables:MOSTYPE,MOSHOOFD, MOPLLAAG
# That is, Purchase~ MOSTYPE + MOSHOOFD + MOPLLAAG

# 1. Fit KNN with k=1, 5 and 20 using 10-folds CV and assess performance on test data.
#    what were the chosen tuning parameter, cv error and test error?

# Splitting the data
set.seed(14)
Caravan_indexTrain <- createDataPartition(y = Caravan$Purchase, p = 0.7, list = FALSE)
Caravan_train <- Caravan[Caravan_indexTrain,] 
Caravan_test <- Caravan[-Caravan_indexTrain,] 

# Hyper-parameter tuning
tg_knn <- expand.grid(k = c(1, 5, 20))

train_control_knn <- trainControl(method = "cv", number = 10)

# Training
knn_model <- train(Purchase ~ MOSTYPE + MOSHOOFD + MOPLLAAG,
                   data = Caravan_train,
                   method = "knn",
                   trControl = train_control_knn,
                   tuneGrid = tg_knn,
                   preProcess = c("center", "scale"), 
                   metric = "Accuracy")
# Best fit: k = 20
knn_model$finalModel

# CV error
knn_model$results

# Test error
knn_prediction <- predict(knn_model, newdata = Caravan_test, type = "raw")

confusionMatrix(knn_prediction, Caravan_test$Purchase, positive = "Yes")


# 2. Fit KNN with k=1, 5 and 20 using LOOCV and assess performance on test data.
tg_knn_LOOCV <- expand.grid(k = c(1, 5, 20))

train_control_knn_LOOCV <- trainControl(method = "LOOCV")

# Training
knn_model_LOOCV <- train(Purchase ~ MOSTYPE + MOSHOOFD + MOPLLAAG,
                   data = Caravan_train,
                   method = "knn",
                   trControl = train_control_knn_LOOCV,
                   tuneGrid = tg_knn_LOOCV,
                   preProcess = c("center", "scale"), 
                   metric = "Accuracy")
# Best fit: k = 20
knn_model_LOOCV$finalModel

# CV error
knn_model_LOOCV$results

# Test error
knn_prediction_LOOCV <- predict(knn_model_LOOCV, newdata = Caravan_test, type = "raw")

confusionMatrix(knn_prediction_LOOCV, Caravan_test$Purchase, positive = "Yes")


#    Fitting time will be much longer than the time it took to fit the knn model
#     on the Auto dataset. How can you explain it?
## Answer: Training a model with Leave-One-Out-Cross-Validation is much more computationally heavy then k-fold CV.
## That is because the model is fitted on N-1 data-points and then validated on the remaining point.
## It is not different from k-fold CV where k = N-1, which usually a much higher number then the traditional 5 or 10.