## Extra- Forward/Backward/Stepwise selection -------------------------------------------------------

library(leaps)

# Besides Best- subset selection, we can use regsubsets() from leaps to perform forward\backward stepwise
# selection, using the argument 'method':
regfit.fwd<-regsubsets(Salary~.,data=Hitters,nvmax=19,method="forward")
regfit.bwd<-regsubsets(Salary~.,data=Hitters,nvmax=19,method="backward")

#using best-subset\ forward\ backward selection, can lead to different results.
#E.G., for this data, the best 7-variable models identified the three methods are different:
coef(regfit.full,7)
coef(regfit.fwd,7)
coef(regfit.bwd,7)

# With caret we can't use the exhaustive best subset selection but we can use 
# the step-wise, forward and backward method!

library(caret)

# First, forward:
caretfit.fwd <- train(Salary ~ ., data = Hitters,
                      method = "leapForward",
                      tuneGrid = data.frame(nvmax = 1:19)) # same as in regsubsets()
caretfit.fwd
plot(caretfit.fwd)
caretfit.fwd$bestTune # gives the row number
caretfit.fwd$results[9,] # Best model was with 9 predictors
coef(caretfit.fwd$finalModel, id = caretfit.fwd$bestTune$nvmax) # Coef for this 9 model
# we got the same coef for the 9 predictor model using regsubsets()
coef(regfit.fwd,9) #see it is the same?

# Same with backward:
caretfit.bwd <- train(Salary ~ ., data = Hitters,
                      method = "leapBackward",
                      tuneGrid = data.frame(nvmax = 1:19)) # same as in regsubsets()
plot(caretfit.bwd)

# leap() don't have stepwise method which combines fwd and bwd selection, but caret does!
# we can use "leapSeq"- which combines Backward and Forward selection:
set.seed(1) 
caretfit.sw <- train(Salary ~ ., data = Hitters,
                     method = "leapSeq", 
                     tuneGrid = data.frame(nvmax = 1:19)) # same as in regsubsets()
plot(caretfit.sw)
caretfit.sw$bestTune # gives the row number
caretfit.sw$results[11,] # Best model was with 11 predictors

# 1) Fitting selection methods on the train data using 10-folds CV.
Backward_fit <- train(Salary ~ ., data = Hitters.train,method = "leapBackward", 
                      tuneGrid = data.frame(nvmax = 1:19),
                      trControl = trainControl(method = "cv", number = 10))
Backward_fit
Forward_fit <- train(Salary ~ ., data = Hitters.train,method = "leapForward",
                     tuneGrid = data.frame(nvmax = 1:19),
                     trControl = trainControl(method = "cv", number = 10))
Forward_fit

set.seed(1)
stepwise_fit <- train(Salary ~ ., data = Hitters.train,
                      method = "leapSeq",tuneGrid = data.frame(nvmax = 1:19),
                      trControl = trainControl(method = "cv", number = 10))
stepwise_fit

# subset selection with CV is actually preforming subset selection WITHIN each of 
# the k training sets so the models are fitted based on the mean of the k fitted models!
# the best model is chosen based on the cv errors
Backward_fit$bestTune$nvmax # Best is 18 when using Backward
Forward_fit$bestTune$nvmax # And 14 when using Forward
stepwise_fit$bestTune$nvmax # And 15 for stepwise...

# *note- RMSE was used to select the optimal model using the smallest value.

# 2) Predict using test data.
pred.bwd<-predict(Backward_fit,Hitters.test)  
pred.fwd<-predict(Forward_fit,Hitters.test)  
pred.sw<-predict(stepwise_fit,Hitters.test)  

# 3) Assess the test error for each type of models:
defaultSummary(data.frame(obs =Hitters.test$Salary, pred=pred.fwd))
defaultSummary(data.frame(obs =Hitters.test$Salary, pred=pred.bwd))
defaultSummary(data.frame(obs =Hitters.test$Salary, pred=pred.sw))