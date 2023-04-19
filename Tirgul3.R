### Tutorial 3- Model Selection and Regularization ###

# Load pack.
library(ISLR)
library(caret)
library(dplyr)
library(tidyverse)

# Recommendation read Feature Selection Overview on caret website
# httpstopepo.github.iocaretfeature-selection-overview.html

## Hitters DATA and the PROBLEM (a REGRESSION problem)---------------------------------

# Hitters Dataset Baseball Data from the 1986 and 1987 seasons
# A data frame with 322 observations of major league players on the following 20 variables.
dim(Hitters)
names(Hitters)
#AtBat- Num. of times at bat in 1986.       #Hits- hits num. in 1986
#HmRun- home runs num. in 1986              #Runs- runs num. in 1986 
#RBI- Number of runs batted in 1986         #Walks- walks num. in 1986        
#Years- Num. of years in the major leagues  #CAtBat- times at bat during career
#CHits- hits during career                  #CHmRun- home runs during career
#CRuns- runs during career                  #CRBI Number of runs batted in during his career
#CWalks- walks during  career
#League- A factor with levels A and N indicating player's league at the end of 1986
#Division- A factor with levels E and W indicating player's division at the end of 1986
#PutOuts- Number of put outs in 1986        #Assists- Number of assists in 1986
#Errors- Number of errors in 1986
#Salary- 1987 annual salary on opening day in thousands of dollars
#NewLeague- A factor with levels A and N indicating player's league at the beginning of 1987 

# We wish to predict a baseball player's Salary on the basis of performance variables in the previous year.
# Which 19 predictors will be best for predicting Salary

#a little note regarding our dependent variable
sum(is.na(Hitters$Salary)) # Salary variable is missing for 59 of the players.
Hitters <- Hitters %>%  # na.omit()  removes all rows that have missing values in any variable.
  drop_na(Salary)
dim(Hitters)               #we are now left with 263 rows with full data




### PART A Model Selection- Best Subset Selection Method -----------------------------------

# We will see how to fit Best Subset Selection.

# Our data is REALLY SMALL such that splitting the data to train and test might leave us with very small 
# datasets. Let's focus with finding the best subset of features for the full data.
# When possible, we will want to first split the data, select features on the train data and test them on the
# test data.

# Best Subset Selection is generally better than the stepwiseforwardbackward methods (see the lesson 3)
# However, if you really want to- see the code Extra- Forward, Backward and Stepwise selection

# With caret we can't use the basic sequential best subset selection...! )

# Therefore, we will use the regsubsets() function from leaps package
library(leaps)

# regsubsets() performs best subset Selection by identifying the best model that 
# contains a given number of predictors, where (best is quantified using RSS). 

regfit.full <- regsubsets(Salary~.,Hitters, nvmax = 8) #The syntax is the same as for lm()glm()train()

summary(regfit.full) #summary() outputs the best set of variables for each model size
#up to the best 8-variable model (8 is the default)
#An asterisk- a given variable is included in the corresponding model. 
#For instance, this output indicates that
#the 2-variable model contains Hits and CRBI.  
#the 3-variable model contains Hits, CRBI and PutOuts.  

#If we want we can fit in this data up to a 19-variable model (and not 8) using the nvmax option.
regfit.full <- regsubsets(Salary~.,data=Hitters,nvmax=19)
summary(regfit.full)

#Let's get a closer look on the statistics of this output
reg.summary <- summary(regfit.full)
names(reg.summary) #The summary() function also returns R2 (rsq), RSS, adjusted R2 (adjr2), Cp, and BIC.
#We can examine these statistics to select the best overall model

reg.summary$rsq #For instance, we see that R2 increases from 32% for 1-variable model, 
#to almost 55%, for 19-variables model.
#As expected, the R2 increases monotonically as more variables are included!

reg.summary$adjr2  #Also as expected, this is not the case for adjusted R2!

#Plotting RSS,adj.R2,Cp and BIC for all of the models at once will help us decide which model to select

par(mfrow=c(2,2)) #setting a graphical parameter such that 
#Subsequent figures will be drawn in an 2-by-2 array
#plotting RSS,adjusted R2,Cp and BIC as a factor of Number of Variables in the model
#RSS
plot(reg.summary$rss,xlab="Number of Variables",ylab="RSS",type="l") 
#(the type=l  tells R to connect the plotted points with lines)
#adjusted R2
plot(reg.summary$adjr2,xlab="Number of Variables",ylab="Adjusted RSq",type="l")
which.max(reg.summary$adjr2)                             #find the max adjusted R2 point
points(11,reg.summary$adjr2[11], col="red",cex=2,pch=20) #mark it on the plot
#Cp
plot(reg.summary$cp,xlab="Number of Variables",ylab="Cp",type="l")
which.min(reg.summary$cp)                                #find the min Cp point
points(10,reg.summary$cp[10],col='red',cex=2,pch=20)     #mark it on the plot
#BIC
plot(reg.summary$bic,xlab="Number of Variables",ylab="BIC",type="l")
which.min(reg.summary$bic)                               #find the min BIC point
points(6,reg.summary$bic[6],col="red",cex=2,pch=20)      #mark it on the plot

#regsubsets() has a built-in plot() command for displaying the selected variables for 
#the best model with a given predictors num., ranked (top to bottom)
#according to the BIC, Cp, adjusted R2, or AIC
par(mfrow=c(1,1)) 
plot(regfit.full,scale="r2")
plot(regfit.full,scale="adjr2")
plot(regfit.full,scale="Cp")
plot(regfit.full,scale="bic")
#The top row of each plot contains a black square for each variable selected
#according to the optimal model associated with that statistic. 
#E.G. several models share a BIC close to -150. However, the model with the lowest BIC 
#(top row) is 6-variable model that contains AtBat, Hits, Walks, CRBI, DivisionW, and PutOuts.

#NOTE- for r2 the top row contains all predictors, for adjusted r2- 11 predictors,     
# for Cp - 10 predictors, and for BIC- 6 predictors.
# BIC places the heaviest penalty on models with many variables.


## Two difficulties with regsubsets() ----------------------------------------------

## Problem 1- no predict() for regsubsets()  No way to test the model
#                                              No, it is possible, but less elegant.

# [If data was good enough for train and test, we would have wanted to assess 
# performance of best subset selection on test data.] 

# To get over the first problem, here youv'e got an HOME-MADE predict method for regsubsets() analyses
# (you don't have to understand it, just run it!).

#THE HOME-MADE 'predict.regsubsets' FUNCTION 
predict.regsubsets <- function(object,newdata,id,...){
  form <- as.formula(object$call[[2]])
  mat <- model.matrix(form,newdata) 
  coefi <- coef(object,id=id)
  xvars <- names(coefi)
  mat[,xvars]%*%coefi
}
# (if you really want to understand it- BUT YOU CAN TOTALLY SKIP THIS
# object= the object created using regsubsets() function. In our case this was regfit.best.
# All objects got a component called a call that contains creation information. 
# The second value contains the formula.
# we create 'form' by extracting the formula used in the call to regsubsets().
# newdata= the test data, id= in our case this can be 1-19 indexes for different model size.

## Example

# Split
set.seed(09042022) 
Hitters.indxTrain <- createDataPartition(y = Hitters$Salary,
                                         p = 0.5, # let's use 0.5 because of the size of the data
                                         list = FALSE)
Hitters.train <- Hitters[Hitters.indxTrain,] 
Hitters.test <- Hitters[-Hitters.indxTrain,] 


# (1) FIT (with up  to a 19-variable model)
regfit.full2 <- regsubsets(Salary~.,data=Hitters.train,nvmax=19)
reg.summary2 <- summary(regfit.full2)

# Manually choose the best model by metric
which.max(reg.summary2$adjr2) #12
which.min(reg.summary$cp)  #10
which.min(reg.summary$bic) #6

# different metrics suggest different things...
# it may be (also) related to the fact that we have very small data.

# (2) Let's PREDICT on test data when we take the best 12 predictors model
predicted.Salary.12p <- predict.regsubsets(regfit.full2, #enter model
                                          Hitters.test, #enter test data
                                          id=12) #enter chosen variable number
# And also with 10
predicted.Salary.10p <- predict.regsubsets(regfit.full2,Hitters.test,id=10) 
# And 6
predicted.Salary.6p <- predict.regsubsets(regfit.full2,Hitters.test,id=6) 

# (3) Finally- assess performance

#(Reminder For regression problems- R-squared, MSE, RMSE, MAE...)

#defaultSummary() fun. can help
defaultSummary(data.frame(obs =Hitters.test$Salary, pred=predicted.Salary.12p))
defaultSummary(data.frame(obs =Hitters.test$Salary, pred=predicted.Salary.10p))
defaultSummary(data.frame(obs =Hitters.test$Salary, pred=predicted.Salary.6p))

# 10 predictors seems good!

## Problem 2- since there was no predict() for regsubsets(), it can't preform CV 
#             But if we prepared one... We can CV using a loop!
#             (even less elegant, though)

# Let's run CV on the FULL data (because, again, data is very small, and using CV will make us more confident
# in the model fitted)

# computing the validation error for the bestmodel of each model size for k=10

# (1) making folds create a vector that allocates each observation to one of k = 10 folds
k=10
set.seed(1)
folds <- sample(1:k,nrow(Hitters),replace=TRUE) 
# (2) create a matrix which will store the results
cv.errors <- matrix(NA,k,19, dimnames=list(NULL, paste(1:19)))
# (3) write a loop that performs k-folds CV 
for(j in 1:k){
  best.fit <- regsubsets(Salary~.,data=Hitters[folds!=j,],nvmax=19)
  #selection according to training set of the jth fold 
  #(i.e.elements of folds that are not equal to j)
  for(i in 1:19){
    pred <- predict(best.fit,Hitters[folds==j,],id=i) #predict for each model size in test set
                                                    #using our home-made predict() function!
    cv.errors[j,i] <- mean((Hitters$Salary[folds==j]-pred)^2) #compute the test MSE and store 
    #it in the errors matrix
  }
}
cv.errors #we got 10 by 19 matrix, of which the (i,j)th element corresponds
          #to the test MSE for the ith CV fold for the best j-variable model. 
mean.cv.errors <- apply(cv.errors,2,mean) # We use the apply() function to average over the columns 
                                        #to obtain a vector for which the jth element is the CV
                                        #error for the j-variable model.
plot(mean.cv.errors,type='b') #We see that CV selects an 10-variable model.
which.min(mean.cv.errors)
points(10,mean.cv.errors[10], col="red",cex=2,pch=20) #mark it on the plot

### PART B- Regularization Shrinkage Methods- Ridge Regression and the Lasso----------------------

# We will perform ridge regression and the lasso in order to predict Salary on the Hitters data.

# (A) FIT a ridge regression to the train data!
# we will continue with caret's train() (you can also use the glmnet() package),
# just put in method = glmnet for ridge lasso regression

# First, choose hyperparameters
# One hyperparameter will be the type of penalty- alpha argument determines what type of model is fit
# based on the penalty- alpha=0 for a ridge regression model, alpha=1 for lasso model,
# and 0alpha1 for net...we will start with ridge
# Another hyperparameter will be the tuning penalty lambda. Let's choose a range for lambda values 

tg <- expand.grid(alpha = 0, 
                 lambda = c(2 ^ seq(10, -10, length = 100)))

# Here we will implement it over a grid of 100 values ranging from 
# lambda=2^10 to lambda=2^-10, thus covering the full range of scenarios from 
# the null model containing only the intercept, to the least squares fit (lambda almost 0).   

# Fit ridge regression with 10-folds cv

set.seed(1)
rigreg_fit <- train(Salary ~ ., 
                    data = Hitters.train,
                    method = "glmnet",
                    preProcess = c("center", "scale"), # IMPORTANT! scale the variable pre-fitting
                    tuneGrid = tg,
                    trControl =  trainControl(method = "cv", number = 10)# 10-fold CV
)
rigreg_fit # we can see CV errors for each lambda- and the chosen best lambda
rigreg_fit$bestTune # gives the row number for best lambda
rigreg_fit$results[74,]

#lets see it
plot(rigreg_fit,xTrans = log)
plot(log(rigreg_fit$results$lambda),
     rigreg_fit$results$RMSE) # the plot shows us 
# x- log(lambda), if we won't use log() it will be very hard to see 
# y- RMSE 
bestlambda <- rigreg_fit$bestTune$lambda #the value of lambda with the smallest CV error is 231.013
log(bestlambda) #3.290699 seems reasonable, looking at the plot.

##we can extract the model according to lambda
# The coef of the chosen model
coef(rigreg_fit$finalModel, s = bestlambda)   
            # we don't have to use s = bestlambda
            # this is the default!

# or other lambdas...
# E.g. for lambda = 0.0000, (this result should be similar to OLS result)
coef(rigreg_fit$finalModel, s = 0) # s for shrinkage parameter
# Again, we don't have to use s = bestlambda, this is the default!
# this is the last tuning parameter in our grid
rigreg_fit$results[100,]

# the different models produced by different lambdas! 
# and the parameters gets smaller as lambda rises
#We can see that depending on the choice of tuning
#parameter, more coefficients will be exactly equal to zero
plot(coef(rigreg_fit$finalModel, s = 0))
plot(coef(rigreg_fit$finalModel, s = 100))
plot(coef(rigreg_fit$finalModel, s = 1000))

# (B) predict and estimate the test error the specific bestlambda models 
#     (we can also compare across the grid of lambdas but we will focus on the 
#      chosen one- best tuning lambda)

ridge.pred <- predict(rigreg_fit,
                    s=bestlambda,
                    newdata=Hitters.test)  #to get predictions for a test set.
# Again, we don't have to use s = bestlambda, this is the default!

# (C) EVALUATE the RMSE on the TEST set, associated with this value of lambda
RMSE(ridge.pred, Hitters.test$Salary)
#TEST RMSE= 379.7412
R2(ridge.pred, Hitters.test$Salary)
#0.2648186


# In Ridge regression we are left with 19 predictors. 
# The Ridge penalty shrink all coefficients, but doesn't set any of them exactly to zero. 
# - Ridge regression does not perform variable selection!

# This may not be a problem for prediction accuracy, but it can create a challenge 
# in model interpretation in settings in which the number of variables is large.               

# The lasso method overcomes this disadvantage...

### The Lasso

# As with ridge regression, the lasso shrinks the coefficient estimates towards zero. 
# However, Lasso's penalty also force some of the coefficient estimates to be exactly 
# equal to zero (when lambda is sufficiently large). Hence, performs variable selection.

# We once again use the train() function; however, this time we use the argument alpha=1.
# Other than that change, we proceed just as we did in fitting a ridge model.

# (A)FITTING a lasso model on the TRAINING set

tg <- expand.grid(alpha = 1, #switch to alpha=1 for lasso
                 lambda = c(2 ^ seq(10, -10, length = 100)))# SAME lambdas
set.seed(1)
lasso_fit <- train(Salary ~ ., data = Hitters.train,
                   method = "glmnet",
                   preProcess = c("center", "scale"),
                   tuneGrid = tg,
                   trControl = trainControl(method = "cv", number = 10))
lasso_fit

#We can see that depending on the choice of tuning
#parameter, more coefficients will be EXACTLY equal to zero
plot(coef(lasso_fit$finalModel, s = 0))
plot(coef(lasso_fit$finalModel, s = 50))
plot(coef(lasso_fit$finalModel, s = 100))
plot(coef(lasso_fit$finalModel, s = 1000))

# the best tuning parameter
plot(lasso_fit,xTrans = log)
bestlambda <- lasso_fit$bestTune$lambda
log(bestlambda)
bestlambda
# (B) PREDICTING for test data a(using best lambda).
lasso.pred <- predict(lasso_fit,s=bestlam,newdata=Hitters.test) 

# (C) EVALUATE the RMSE on the TEST set, associated with this value of lambda
RMSE(lasso.pred, Hitters.test$Salary)
# TEST RMSE= 377.9078
R2(lasso.pred, Hitters.test$Salary)
#0.2807987

# Lasso's model advantage over ridge regression-
# the resulting coefficient estimates are sparse
coef <- coef(lasso_fit$finalModel, s = bestlambda)
coef
sum(coef==0)# see that there are 3 estimates which are exactly 0!
plot(coef)

## Elastic Net----------------------------------------------------------------------------

# Elastic Net emerged as a result of critique on lasso, whose variable selection can be too
# dependent on data and thus unstable. The solution is to combine the penalties of ridge 
# regression and lasso to get the best of both worlds. 
# alpha is the mixing parameter between ridge (alpha=0) and lasso (alpha=1).
# That is, fo Elastic Net there are two parameters to tune lambda and alpha. 

# lets try 25 possible alpha values
tg <- expand.grid(alpha = c(seq(0, 1, length.out=25)), 
                 lambda = c(2 ^ seq(10, -10, length = 100)))# SAME lambdas
tg # 25 alphas 100 lambda = 2500 models (each one with 10-fold CV!)

# Train the model
elastic_fit <- train(Salary ~ .,
                     data = Hitters.train,
                     method = "glmnet",
                     preProcess = c("center", "scale"),
                     tuneGrid = tg,
                     trControl = trainControl(method = "cv", number = 10))
elastic_fit
# yes... that's quite long...
plot(elastic_fit,xTrans = log)


# the best tuning parameter
elastic_fit$bestTune

# Prediction and assesment are exactly the same as in Ridge and Lasso

## Exercise--------------------------------------------------------------

# Use the U.S. News and World Report’s College Data dataset ('College' in ISLR).
# this dataset contains 777 observations of US colleges with the following variables

head(College)

# Private - A factor with levels No and Yes indicating private or public university
# Apps- Number of applications received
# Accept- Number of applications accepted
# Enroll- Number of new students enrolled
# Top10perc- Pct. new students from top 10% of H.S. class
# Top25perc- Pct. new students from top 25% of H.S. class
# F.Undergrad- Number of fulltime undergraduates
# P.Undergrad- Number of parttime undergraduates
# Outstate- Out-of-state tuition
# Room.Board- Room and board costs
# Books- Estimated book costs
# Personal- Estimated personal spending
# PhD- Pct. of faculty with Ph.D.’s
# Terminal- Pct. of faculty with terminal degree
# S.F.Ratio- Studentfaculty ratio
# perc.alumni- Pct. alumni who donate
# Expend- Instructional expenditure per student

# Lets predict(linear prediction) from these 17 variables the outcome
# Grad.Rate- Graduation rate

# Split to train and test. use 0.7 for the train data

# Then, use each of the learned methods to answer this task. That is
# 1. Best Subset Selection 
#     Notes -No need to use the CV function for now,  just train on train data and predict
#            -Try all possible features combinations, with up  to a 17-variable model
#            -use AdjR2 to choose the model
# 2. Ridge regression
# 3. Lasso
# 4. Elastic net (use the alpha = c(seq(0, 1, length.out=25) )

# Notes for the last 3 methods 

#  choose the same lambda values. see that they are broad enough.
#  How plot change in RMSE and examine yourself. adjust the values when needed.
#  use 10-folds CV.
# 


# Did the method diverged from each other in their performance on test data (look at R2)
# Which one preformed best
