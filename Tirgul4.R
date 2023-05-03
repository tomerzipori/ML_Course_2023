### Tutorial 4: Decision Trees ###

library(ISLR)
library(caret)
library(dplyr)

# we will also use"MASS" and "randomForest" packages

## Fitting and Evaluating Basic *Classification* Trees-----------------------------------------------

# We will use classification trees to analyze the Carseats data set (from ISLR)-
# A simulated data set containing sales of child car seats at 400 different stores.
dim(Carseats)
head(Carseats)

## Preparing a binary response:

# We want to use Sales as the response.  
summary(Carseats$Sales)
# Since it is a continuous variable, we will re-code it as a binary variable: 
Carseats$HighSales<-ifelse(Carseats$Sales<=8,"No","Yes") #'High' is "Yes" if the Sales exceeds 8 and "No" otherwise.
Carseats$HighSales<-as.factor(Carseats$HighSales)
Carseats<-Carseats %>% select(!Sales) #removing Sales from the data as it won't make sense
# to predict HighSales from Sales
# Base rate:
table(Carseats$HighSales)/400

## First, split the data to train and test sets:
# (we will build the tree using the train set, and evaluate its performance on the test set...)

# Split:
set.seed(04262023) 
Carseats.indxTrain <- createDataPartition(y = Carseats$HighSales,
                                          p = 0.6, # let's use 0.6 because of the size of the data
                                          list = FALSE)
Carseats.train <- Carseats[Carseats.indxTrain,] 
Carseats.test <- Carseats[-Carseats.indxTrain,] 


## (A) Fitting a basic classification tree on the training data:

# train() function knows also how to do trees!
# a simple decision tree is fitted with method = "rpart"
# (same method will apply for numeric y, the function will detect outcome
#  type automatically)

# We will fit a tree that predict High using all variables (but Sales).

tc <- trainControl(method = "cv", number = 10) #using 10-folds CV
tg <- expand.grid(cp = 0)# cp is the complexity parameter. 
# If set to 0, no penalizing / pruning is done 
# (a simple tree)


fit.tree <- train(HighSales ~ ., 
                  data = Carseats.train,
                  method = "rpart",
                  tuneGrid = tg,
                  trControl = tc) 

fit.tree # fit indices based on the fitted model
# the CV error rate. in our case- 1-0.7548333   = 0.2451667 

## Let's explore the tree:
fit.tree$finalModel

# In each row in the output we see:
# 1. Node index number (where node 1 is the total sample)
# 2. the split criterion (e.g. ShelveLocGood< 0.5 and ShelveLocGood>=0.5 for
#                              the first split in nodes 2 and 3) 
# 3. num. of observations under that node 
# see how nodes (2) and (3) complete each other...
# (and also 4 and 5, or 6 and 7...)
# 4. the deviance- the number of obs. within the node that deviate from
#    the overall prediction for the node (because even trees are not perfect...)
# 5. the overall prediction for the node (Yes/ No)
# 6. in parenthesis- the fraction of observations in that node that take on 
#    values of No (first) or Yes (second) which actually can be calculated using n and deviance.
# * Branches that lead to terminal nodes are indicated using asterisks.


## PLOTTING:
# One of the most attractive properties of trees is that they can be graphically displayed:
plot(fit.tree$finalModel) #display the tree structure
text(fit.tree$finalModel, pretty = 0, cex = 0.6) # display the node labels. 
# pretty=0 tells R to include the category names
#          for any qualitative predictors.
# cex is for font size


summary(fit.tree) # more detailed results

## (B)+(C): predict for the test data and evaluate performance

# Predicting High sales for test data:
prd.tree<-predict(fit.tree,Carseats.test)

# Evaluating test error\ any other fit indices:
confusionMatrix(prd.tree, Carseats.test$HighSales, positive="Yes")

# This tree leads to correct predictions for 72.33% of the locations in the test data (test error rate= 27.67%)

## Tree Pruning-------------------------------------------------------------------------

# Next, we consider whether pruning the tree might lead to improved results.

# For finding the best size for the tree using CV change the possible alpha values for 
# complexity parameters (cps) in the tune grid.

tg <- expand.grid(cp = seq(0,0.25,length=100)) # check 100 alphas- from 0 to 0.3 

# Alpha specifies how the cost of a tree is penalized by the number of terminal nodes, 
# resulting in a regularized cost for each tress.
# Small alpha results in larger trees and potential over-fitting, large alpha - small trees and 
# potential under-fitting.
# (in other words, alpha controls a trade-off between the subtree?s complexity and its fit to the training data).

set.seed(1234)
fit.tree <- train(HighSales ~ ., 
                  data = Carseats.train,
                  method = "rpart",
                  tuneGrid = tg,
                  trControl = tc)
fit.tree

# we can see accuracy and kappa per cp.
# See the drop in accuracy when alpha gets bigger
plot(fit.tree)

# using CV train() determine the optimal level of tree complexity; 
# cost complexity pruning ("weakest link pruning") is used in order to select a 
# sequence of best subtrees for consideration, as a function of the tuning parameter alpha
# and here- accuracy
# we see that optimal model was found for cp = 0.02121212 (out of 100 alphas we looked at).
fit.tree$bestTune

fit.tree$finalModel
# In that case this final model don't seem so pruned (same number of nodes)..
#But....
fit.tree$finalModel$var # 9 variables where left

plot(fit.tree$finalModel) 
text(fit.tree$finalModel, pretty = 0, cex = 0.6)

## Evaluate the pruned tree:
prd.tree<-predict(fit.tree,Carseats.test)
confusionMatrix(prd.tree,Carseats.test$HighSales, positive = "Yes")

# Now 72.33% of the test observations are correctly classified, so not only has
# the pruning got rid of one variable, it also didn't 
# changed the classification accuracy.

## Fitting and Evaluating REGRESSION Trees--------------------------------------------------

# we also have regression problems!

## Boston Data: this data set is included in the MASS library.
library(MASS)
head(Boston)
dim(Boston)
# The data records medv (median house value) for 506 neighborhoods around Boston. 
# We will seek to predict medv using 13 predictors such as:
# rm= average number of rooms per house; age= average age of houses;
# and lstat=percent of households with low socioeconomic status.

# The processes fitting and Evaluation of a Regression Tree are essentially the same.

# splitting the data into a train and test sets
# Split
set.seed(26042023) 
Boston.indxTrain <- createDataPartition(y = Boston$medv,p = 0.7,list = FALSE)
Boston.train <- Boston[Boston.indxTrain,] 
Boston.test <- Boston[-Boston.indxTrain,] 


# (A) Fitting a regression tree on the training data:
#          *here we incorporate pruning during the initial fitting.

tg <- expand.grid(cp = seq(0,0.2,length=100))# check 100 alphas- from 0 to 0.2 

set.seed(100)
tree.boston <- train(medv ~ ., 
                     data = Boston.train,
                     method = "rpart",
                     tuneGrid = tg,
                     trControl = tc)
tree.boston$bestTune
tree.boston$results[2,]
# In the context of a regression tree,alpha is chosen based on the CV RMSE:
# (Here- alpha=0.002020202 and RMSE=4.482024)
plot(tree.boston) #RMSE gets bigger when cp rises

# The regression tree:
tree.boston$finalModel
plot(tree.boston$finalModel)
text(tree.boston$finalModel,pretty=0,cex=0.55)
# in the terminal nodes we see group means!


## (B) + (C) Evaluate the tree performance on test data
medv.pred<-predict(tree.boston,Boston.test)
RMSE(medv.pred,Boston.test$medv) #TEST RMSE is 3.698478
R2(medv.pred,Boston.test$medv) # 85.7% is the effect size


## Bagging-------------------------------------------------------------------------------------------

# We apply bagging and random forests to the Boston data.
# The motive- to reduce the variance when modelling a tree. 

# Bagging= training the method on B different bootstrapped training data sets.
#          and average across all the predictions (for regression) or take the 
#          majority vote\calculated probability (for classification) 

## Fitting a regression tree to the train data while applying bagging:
# use method= "rf" (as bagging in type of random forest)
# with maximal number of predictors.

tg <- expand.grid(mtry = 13)
# mtry=13 indicates that all 13 predictors should be considered
# for each split of the tree ? in other words, that bagging should be done.
# * this is a fixed hyper parameter for now...
set.seed(1)
fit.bag <- train(medv ~ ., 
                 data = Boston.train,
                 method = "rf",
                 tuneGrid = tg,
                 trControl = tc)

fit.bag$finalModel

# see that the default is 500 bootstraped trees. you can change it by 
# adding the argument: ntree= (num. of requested trees)
# CV MSE 13.70455, this is the mean error across all 500 trees.
# RMSE: sqrt(13.70455)= 3.701966

# see how it decreased with re-sampling:
plot(fit.bag$finalModel)
fit.bag

# We can't look on a specific tree but, we can asses variables'
# importance:

# varImp() will show us the mean decrease in node impurity
VarImp.Boston.Bagg<-varImp(fit.bag$finalModel)
VarImp.Boston.Bagg
# IncNodePurity= the total decrease in node impurity that results from
#                splits over that variable, averaged over all trees.
# * In the case of regression trees, the node impurity is measured by the training
#   RSS, and for classification trees by the deviance. 

# randomForest package has a built-in function to plot importance
randomForest::varImpPlot(fit.bag$finalModel)
# The results indicate that across all of the trees considered in the random
# forest, the wealth level of the community (lstat) and the house size (rm)
# are by far the two most important variables.

## Evaluating the tree performance on test data:
prd.bag <- predict(fit.bag, Boston.test)
RMSE(prd.bag, Boston.test$medv)
R2(prd.bag, Boston.test$medv)

# The test set RMSE, 2.590941, is better than the one
# obtained using an optimally-pruned single tree!
# also, we explained 93.7% of variance in test data, whereas we explained
# only 85.7% for the pruned tree...

## Random Forests-------------------------------------------------------------------------------------

# the same principle as for bagging with an improvement- when building the trees on the 
# bootstrapped training data, each time a split in a tree is considered, a random sample of
# m predictors is chosen as split candidates from the full set of p predictors-> 
# de-correlating the trees, thereby making the average of the resulting trees less variable
# and hence more reliable.

# Recall that bagging is simply a special case of a random forest with m = p. 
# Therefore, train() function with method="rf" is used for both analyses.. 
# Growing a random forest proceeds in exactly the same way, except that we use a smaller
# value of the mtry argument. A default, for mtry will be  p/3 variables for 
# regression trees, and square root of p variables for classification trees. 
# we can use CV to choose among different mtry. We have 13 predictors, so lets try:
# mtry=4, (closest p/3)
# and also mtry=2,7,10 for fun, as well as 13 (i.e. bagging) 

tg <- expand.grid(mtry = c(2,4,7,10,13))

set.seed(1)
fit.rf <- train(medv ~ ., 
                data = Boston.train,
                method = "rf",
                tuneGrid = tg,
                trControl = tc)
# for each mtry 500 trees where fitted! that is why it took so long 
plot(fit.rf)
fit.rf
#The final value used for the model was mtry = 4.
plot(fit.rf$finalModel) 

# Evaluate
prd.rf <- predict(fit.rf, Boston.test)
RMSE(prd.rf, Boston.test$medv)
R2(prd.rf, Boston.test$medv)

#random forests yielded an improvement over bagging in this case.

varImp(fit.rf$finalModel)
randomForest::varImpPlot(fit.rf$finalModel)

# Same conclusions with better fit!

## Boosting --------------------------------------------------------------------

# In Bagging\ Random forest, each tree is built on an independent bootstrap data.
# Boosting does not involve bootstrap sampling, and trees are grown sequentially: 
# each tree is grown using information from previously grown trees->
# each tree is fitted using the current residuals, rather than the outcome Y.

#Boosting has three tuning parameters:
# 1. The number of trees B
# 2. The shrinkage parameter lambda
# 3. The number d of splits in each tree, which controls 
#    the complexity of the boosted ensemble

## Fitting a regression tree to the train data while applying Boosting:
# we use method= "gbm"

tg<- expand.grid( interaction.depth = 4,  # limits the depth of each tree (d) to 4 splits
                  n.trees = 5000,  # indicates that we want 5000 trees (B)
                  shrinkage = 0.1, #alpha, learning rate
                  n.minobsinnode = 5 # n.minobsinnode restrict splitting of trees until the minimum of
                  # 5 obs. per node. The furthest you can go is n.minobsinnode=1. 
)

set.seed(100)
fit.boost <- train(medv ~ ., 
                   data = Boston.train,
                   method = "gbm",
                   tuneGrid = tg,
                   trControl = tc,
                   verbose = FALSE)

# A note regarding n.minobsinnode- What is the best value to use? It depends on the data set and whether you 
# are doing classification or regression. Since each trees' prediction is taken as the average of the
# dependent variable of all inputs in the terminal node, a value of 1 probably won't work so well for 
# regression(!) but may be suitable for classification.
# Higher values > smaller trees + algorithm run faster.
# Generally, results are not very sensitive to this parameter,
# The interaction depth, shrinkage and number of trees will all be much more significant!

fit.boost
summary(fit.boost)
# summary() produces a relative influence plot and outputs the relative influence statistics.
# We see that lstat and rm are by far the most important variables. 

# We can also produce partial dependence plots for these two variables. These plots
# illustrate the marginal effect of the selected variables on the response after
# integrating out the other variables: 
plot(fit.boost$finalModel,i="rm") # median house prices are increasing with rm 
plot(fit.boost$finalModel,i="lstat") # median house prices are decreasing with lstat 

## Evaluating the tree performance on test data:
prd.boost <- predict(fit.boost, Boston.test)
RMSE(prd.boost, Boston.test$medv)
R2(prd.boost, Boston.test$medv)


# Keep in mind- here we kept lambda and interaction.depth constant.
# we may want to use CV using different optional values for these hyper-paramaters.


# Exercise 4---------------------------------------------------------------

# Use the Hitters dataset from the former tutorial.
# Hitters Dataset: Baseball Data from the 1986 and 1987 seasons
# A data frame with 322 observations of major league players on the following 20 variables.

# A) Fit *regression* trees to predict Salary from the other variables:
# 1. use pruned tree (try different cps with CV). 
#    what is the best tune cp and related CV error? (examine the CV error plot and
#    change the cp range if needed.)
#    plot the tree. 
#    What is the test error and effect size (R2)?
# 2. use Random Forrest (try different mtry values with CV,
#    one value should include the bagging option for this model).
#    what was the chosen mtry and related cv error?
#    What is the test error and effect size (R2)?
#    which variable was the most important?
# 3. use Boosting (you may play with hyper-parameter values or just 
#    use the same as in the tutorial)
#    what was the cv error?What is the test error and effect size (R2)?

# B) Fit *classification* trees to predict Salary from the other variables.
#    Now, create a binary Salary variable called Salary_dic, where <550 is labeled
#    as "low", and >=550 is labeled as "high" (don't forget to remove the "old" salary
#    variable to use the model: Salary_dic~. Make Salary_dic a factor!
#     repeat tasks 1-3 as in A (look at one or more relavent fit indices).

# Notes: when preparing the data: 
# - remove cases with NA values.
# - split to train and test with p=0.5