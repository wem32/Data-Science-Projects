# loading packages
library(modelr)
library(tidyverse)
library(ggplot2)
library(randomForest)
library(readxl)
library(caret)
library(e1071)
library(gbm)

### This is the code for the other larger datasets, which were unused for this project. 

# loading data
credit_c.d <- read_excel("data/Unprocessed/default of credit card clients_1.xlsx") %>%
  mutate(default_nm=as.factor(default_nm))

View(credit_c.d)
# creating a time series version of the data

table1 <- credit_c.d %>%
  gather(key = "pay_period", value = "pay_amt", starts_with("pay")) %>% 
  transmute(pay_period =pay_period, pay_amt = pay_amt)


table2 <- credit_c.d %>%
  gather(key = "bill_period", value = "bill_amt", starts_with("bill")) %>%
  transmute(bill_period = bill_period, bill_amt = bill_amt)

table3 <- credit_c.d %>%
  gather(key = "repay_period", value = "status", starts_with("repay")) %>%
  transmute(limit_bal = limit_bal, sex = sex, educ = educ, marriage = marriage, age = age, repay_period = repay_period, status = status, default_nm=default_nm)

credit_gather <- data.frame(table3, table1, table2)


# renaming file to credit default time series
credit_c.d_ts <- credit_gather %>% mutate(ID = seq.int(nrow(credit_gather)))


# checking for missing values
lapply(credit_c.d_ts, function(x) {sum(is.na(x))})

# creating the training data
set.seed(1)
credit_train <- credit_c.d_ts %>% sample_n(100000)

set.seed(2)
credit <- anti_join(credit_c.d_ts, credit_train, by = "ID") %>%
  mutate(sex=as.factor(sex), educ=as.factor(educ), marriage=as.factor(marriage), repay_period=as.factor(repay_period),
         status=factor(status,levels = c("no credit-use", "pay duly", "revolving credit","1M-delay","2M-delay", "3M-delay","4M-delay","5M-delay","6M-11M-delay")),
         pay_period=as.factor(pay_period), bill_period = as.factor(bill_period))

# creating a validation set
set.seed(3)
credit_val <- credit %>% sample_frac(1/2)


# creating a test set
set.seed(4)
credit_test <- anti_join(credit, credit_val, by = "ID")

# checking that limit_bal follows a gaussian distribtuion
credit_train %>%
  ggplot(aes(x =limit_bal)) +
  geom_freqpoly()

#The distribution is right skewed, but it is still standard normal because of the central limit theorem. 

# Making a feature plots for our continous variables
library(mlbench)
featurePlot(x = credit_train[,c(1,5,10,12)], 
            y = credit_train$default_nm, 
            plot = "strip",
            span = .5,
            layout = c(3, 1))


# checking the correlation of our variables 
cor(credit_train[c(1,10,12)])
# our quantitative variables seem to suggest low correlation, which is good.

# transforming categorical variables to factors
credit_train <- credit_train %>% 
  mutate(sex=as.factor(sex), educ=as.factor(educ), marriage=as.factor(marriage), repay_period=as.factor(repay_period),
         status=factor(status,levels = c("no credit-use", "pay duly", "revolving credit","1M-delay","2M-delay", "3M-delay","4M-delay","5M-delay","6M-11M-delay")),
         pay_period=as.factor(pay_period), bill_period = as.factor(bill_period))
# checking for missing values
lapply(credit_train, function(x) {sum(is.na(x))})




### Logistic Regression

credit_glm <- glm(default_nm~. , data =credit_train[,-13],family = "binomial" )
summary(credit_glm)

# The logistic regression seems to make some sense; however, there are some questionable variables that are significant
# Making prediction using on our validation set

set.seed(5)
credit_probs <- predict(credit_glm, type = "response")

# quick glace on our prediction
credit_probs[1:10]

# Predicting default
cred_df_pred <- rep("current", 100000)
cred_df_pred[credit_probs >.55] <- "default"

# viewing our training error
table(cred_df_pred, credit_train$default_nm)
train_error <- (matrix[2,1]+matrix[1,2]) /100000
train_error

# Our training error seems pretty good. 

# checking against the validation 
set.seed(6)
credit_probs <- predict(credit_glm, credit_val[,-8], type = "response" )
credit_probs[1:10]

cred_df_pred <-rep("current", 40000)
cred_df_pred[credit_probs >.55] <-"default"

# calculating the validation error
table(cred_df_pred, credit_val$default_nm)
val_error <- (matrix[2,1]+matrix[1,2]) /40000
val_error


# calculating the test error
set.seed(7)
credit_probs <- predict(credit_glm, credit_test[,-8], type = "response")
credit_probs[1:10]

cred_df_pred <- rep("current", 40000)
cred_df_pred[credit_probs >.55] <- "default"

table(cred_df_pred, credit_test$default_nm)

test_error <- (matrix[2,1]+matrix[1,2]) /40000
test_error


# Excluding bill and pay period because they are not significant. 

credit_glm <- glm(default_nm~. , data =credit_train[,-c(9, 11,13)],family = "binomial" )
summary(credit_glm)

# predicting class
set.seed(8)
credit_probs <- predict(credit_glm, type = "response")
cred_df_pred <- rep("current", 100000)
cred_df_pred[credit_probs >.55] <- "default"

matrix <- table(cred_df_pred, credit_train$default_nm)
train_error <- (matrix[2,1]+matrix[1,2]) /100000
train_error
# We have a higher training error rate

# measuring the validation error-rate
set.seed(9)
credit_probs_1 <- predict(credit_glm, newdata = credit_val[,-8], type = "response")
cred_df_pred_1 <-rep("current", 40000)
cred_df_pred_1[credit_probs_1 >.55] <-"default"
matrix <- table(cred_df_pred_1, credit_val$default_nm)
val_error <- (matrix[2,1]+matrix[1,2]) /40000
val_error

# measuring test error
set.seed(10)
credit_probs_2 <- predict(credit_glm, newdata = credit_test[,-8], type = "response")
cred_df_pred_1[credit_probs_2 >.55] <-"default"
matrix <- table(cred_df_pred_1, credit_test$default_nm)
test_error <- (matrix[2,1]+matrix[1,2]) /40000
test_error

# The errors on the data that the model has not seen were higher than that of the training error. 


# Running another logistic regression with some nonlinear terms based on theoretical anticipation. 

mod_credit_glm <- glm(default_nm~I(limit_bal^2)+sex+educ+marriage+I(1/age)+repay_period+status+pay_period+I(pay_amt^2)
                      +bill_period+I(bill_amt^2), data =credit_train[,-13],family = "binomial" )
summary(mod_credit_glm)

# making prediction
set.seed(11)
mod_credit_probs <- predict(mod_credit_glm, type = "response")

# glacing on our prediction
mod_credit_probs[1:10]

# Predicting default
mod_cred_pred <- rep("current", 100000)
mod_cred_pred[mod_credit_probs >.55] <- "default"

# viewing our training error
matrix <- table(mod_cred_pred, credit_train$default_nm)
train_error <- (matrix[2,1]+matrix[1,2])/100000
train_error


# measuring the validation error-rate
set.seed(12)
mod_credit_probs_1 <- predict(mod_credit_glm, credit_val[,-8], type = "response")

mod_cred_pred_1 <-rep("current", 40000)
mod_cred_pred_1[mod_credit_probs_1 >.55] <-"default"
matrix <- table(mod_cred_pred_1, credit_val$default_nm)
val_error <-(matrix[2,1]+matrix[1,2]) /40000
val_error

# measuring test error
set.seed(13)
mod_credit_probs_2 <- predict(mod_credit_glm, credit_test[,-8], type = "response")
mod_cred_pred_1[mod_credit_probs_2>.55] <- "default"
matrix <- table(mod_cred_pred_1, credit_test$default_nm)
test_error <- (matrix[2,1]+matrix[1,2]) /40000
test_error
# The error rate were higher under this modified approach.


##### RandomForest.

## randomforest using train function

rf_grid <- expand.grid(.mtry=seq(1, 5, by =2), .ntree = seq(1000, 8500, by= 1500))
rf_train <- train(default_nm~., credit_train[,-13], method = "rf", metric = "Accuracy", tuneGrid = rf_grid,)

### Performing randomforest using randomForest

credit_train_rf <- randomForest(default_nm~. , data = credit_train[,-13], mtyr= 3, ntree =500)
credit_train_rf
credit_predict <- predict(credit_train_rf, newdata = credit_val[, -8], type = "response")
matrix <- table(credit_predict, credit_val$default_nm)
val_error <-(matrix[2,1]+matrix[1,2])/40000
val_error

# visualization
importance(credit_train_rf)
varImpPlot(credit_train_rf)

# Testing on test set


# Another RandomForest

set.seed(8)
credit_train_rf <- randomForest(default_nm~., data = credit_train[,-13], mtyr = 3, ntree = 2000)
credit_train_rf

credit_predict <- predict(credit_train_rf, newdata = credit_val[, -8], type = "response")
matrix <- table(credit_predict, credit_val$default_nm)
val_error <-(matrix[2,1]+matrix[1,2])/40000
val_error


# testing on test set.


# Stochastic gradient Boosting

gbmGrid <- expand.grid(.interaction.depth = seq(1,5, by =2), .n.trees =seq(1000, 5500, by =1500), .shrinkage = c(0.01), .n.minobsinnode = 50)
set.seed(14)
gbm_fit <- train(default_nm ~., data= credit_train[,-13], method = "gbm", tuneGrid = gbmGrid, verbose = FALSE)
gbm_fit
summary(gbm_fit)
plot(gbm_fit)

## The accuracy from this training algorithm is perhaps not the best. However, if we expanded our grid, we would certainly
## improved our accuracy greatly.

## Another Stochastic gradient Boosting

gbmGrid_2 <- expand.grid(.interaction.depth = seq(1,9, by =2), .n.trees =seq(1000, 8500, by =1500), .shrinkage = c(0.01), .n.minobsinnode = c(10,50))
set.seed(15)
gbm_fit_2 <- train(default_nm ~., data= credit_train[,-13], method = "gbm", tuneGrid = gbmGrid, verbose = FALSE)
gbm_fit
summary(gbm_fit)
plot(gbm_fit)

## Stochastic gradient boosting using gbm function


credit_train_1 <- credit_train %>% mutate(default_nm = factor(default_nm))

credit_train_1$default_nm <- ifelse(credit_train_1$default_nm==0,0,1)
set.seed(16)
credit_train_bst <- gbm(default_nm~., data=credit_train_1[,-13], distribution = "bernoulli", n.trees = 2000, interaction.depth = 3, shrinkage =0.2, verbose = F )
summary(credit_train_bst)

# Visualizatiion
par(mfrow= c(2,2))
plot(credit_train_bst, i = "status")
plot(credit_train_bst, i = "bill_amt")
plot(credit_train_bst, i = "limit_bal")
plot(credit_train_bst, i = "pay_amt")

credit_pred_bst <- predict(credit_train_bst, newdata = credit_val[,-8], n.trees = , type = "response")
credit_pred_bst

gbm.class<-ifelse(credit_pred_bst<0.55,'no','yes')

matrix <- table(gbm.class, credit_val$default_nm)
matrix

val_error <- (matrix[2,1]+matrix[1,2])/40000
val_error

# Testing on Test data

credit_pred_bst <- predict(credit_train_bst, newdata = credit_test[,-8], n.trees =2000 , type = "response")
credit_pred_bst


gbm.class<-ifelse(credit_pred_bst<0.55,'no','yes')

matrix <- table(gbm.class, credit_test$default_nm)
matrix

# test  error
test_error <- (matrix[2,1]+matrix[1,2])/40000
test_error

# Performing another boosting using gbm

set.seed(17)
credit_train_bst <- gbm(default_nm~., data=credit_train_1[,-13], distribution = "bernoulli", n.trees = 5000, interaction.depth = 3, shrinkage =0.2, verbose = F )
summary(credit_train_bst)

# Visualizatiion
par(mfrow= c(2,2))
plot(credit_train_bst, i = "status")
plot(credit_train_bst, i = "bill_amt")
plot(credit_train_bst, i = "limit_bal")
plot(credit_train_bst, i = "pay_amt")

credit_pred_bst <- predict(credit_train_bst, newdata = credit_val[,-8], n.trees =5000 , type = "response")
credit_pred_bst


gbm.class<-ifelse(credit_pred_bst<0.55,'no','yes')

matrix <- table(gbm.class, credit_val$default_nm)
matrix

val_error <- (matrix[2,1]+matrix[1,2])/40000
val_error

# Testing on Test data

credit_pred_bst <- predict(credit_train_bst, newdata = credit_test[,-8], n.trees =5000 , type = "response")
credit_pred_bst


gbm.class<-ifelse(credit_pred_bst<0.55,'no','yes')

matrix <- table(gbm.class, credit_test$default_nm)
matrix

# test  error
test_error <- (matrix[2,1]+matrix[1,2])/40000
test_error