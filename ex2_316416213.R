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