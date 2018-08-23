# loading packages
library(modelr)
library(tidyverse)
library(ggplot2)
library(randomForest)
library(readxl)
library(caret)
library(e1071)
library(gbm)

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

# Filtering data to include only graduate and college students since they mirror the closest our study group


student_univ <- credit_c.d_ts %>%
  filter(educ == c("grad school", "university"), age == c(22, 45))

# Looking at how the distribution of some variables changed (may or may not) because of our filtering

student_univ %>%
  ggplot(aes(x=limit_bal)) + geom_histogram()

student_univ %>%
  ggplot(aes(x = pay_amt)) + geom_histogram()

student_univ %>%
  ggplot(aes(x =bill_amt)) + geom_histogram()


# fitting randomForest models

#creating a model with default paramters
control <- trainControl(method = "repeatedcv", number = 5, repeats = 3)
set.seed(15)
metric <- "Accuracy"
mtry <- sqrt(ncol(student_univ[,-c(8,13)]))
tunegrid <- expand.grid(.mtry = mtry)
rf_default <- train(default_nm~., data = student_univ, method = "rf", metric = metric, tuneGrid = tunegrid, trControl = control)
print(rf_default)

control <- trainControl(method = "repeatedcv", number = 10, repeats = 3)
set.seed(16)
metric <- "Accuracy"
mtry <- sqrt(ncol(student_univ[,-c(8,13)]))
tunegrid <- expand.grid(.mtry = mtry)
rf_default <- train(default_nm~., data = student_univ, method = "rf", metric = metric, tuneGrid = tunegrid, trControl = control)
print(rf_default)


# Random search for mtry within a range
control <- trainControl(method="repeatedcv", number=5, repeats=3, search="random")
set.seed(17)
mtry <- sqrt(ncol(student_univ[,-c(8,13)]))
rf_random <- train(default_nm~., data=student_univ, method="rf", metric=metric, tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)

control <- trainControl(method="repeatedcv", number=10, repeats=3, search="random")
set.seed(18)
mtry <- sqrt(ncol(student_univ[,-c(8,13)]))
rf_random <- train(default_nm~., data=student_univ, method="rf", metric=metric, tuneLength=15, trControl=control)
print(rf_random)
plot(rf_random)

# linear grid search
control <- trainControl(method="repeatedcv", number=5, repeats=3, search="grid")
set.seed(19)
tunegrid <- expand.grid(.mtry=c(1:11))
rf_gridsearch <- train(default_nm~., data=student_univ, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
plot(rf_gridsearch)

control <- trainControl(method="repeatedcv", number=10, repeats=3, search="grid")
set.seed(20)
tunegrid <- expand.grid(.mtry=c(1:11))
rf_gridsearch <- train(default_nm~., data=student_univ, method="rf", metric=metric, tuneGrid=tunegrid, trControl=control)
print(rf_gridsearch)
summary(rf_gridsearch)
plot(rf_gridsearch)

# Manual search. Also we will stick to 10K-fold because they seem to provide a better estimate. 
control <- trainControl(method="repeatedcv", number=10, repeats = 3, search = "grid")
tunegrid <- expand.grid(.mtry=c(sqrt(ncol(student_univ[,-c(8,13)]))))
modellist <- list()
for(ntree in c(200, 500, 1000, 2000)) {
  set.seed(21)
  rf_manual <- train(default_nm~., data = student_univ, method = "rf", metric = metric, tunegrid = tunegrid, trControl= control, ntree = ntree)
  key <- toString(ntree)
  modellist[[key]] <- rf_manual
}

# compare results

results <- resamples(modellist)
print(results)
summary(results)
dotplot(results)

# tuning multiple parameters

x <- student_univ[,-c(8,13)]
y <- student_univ[,8]

customRF <- list(type = "Classification", library = "randomForest", loop = NULL)
customRF$parameters <- data.frame(parameter = c("mtry", "ntree"), class = rep("numeric", 2), label = c("mtry", "ntree"))
customRF$grid <- function(x, y, len = NULL, search = "grid") {}
customRF$fit <- function(x, y, wts, param, lev, last, weights, classProbs, ...) {
  randomForest(x, y, mtry = param$mtry, ntree=param$ntree, ...)
}
customRF$predict <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata)
customRF$prob <- function(modelFit, newdata, preProc = NULL, submodels = NULL)
  predict(modelFit, newdata, type = "prob")
customRF$sort <- function(x) x[order(x[,1]),]
customRF$levels <- function(x) x$classes

# training a model
control <- trainControl(method="repeatedcv", number=10, repeats=3)
tunegrid <- expand.grid(.mtry=c(1:11), .ntree=c(500, 1000, 1500, 2000))
set.seed(22)
custom <- train(default_nm~., data=student_univ, method=customRF, metric=metric, tuneGrid=tunegrid, trControl=control)
custom$results
summary(custom)
plot(custom)



# Performing Boosting

#Selecting the best boosting tuning paramter
fit_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
set.seed(23)
best_boost_1 <- train(default_nm~., data =student_univ[,-13] , method = "gbm", trControl = fit_control, verbose = FALSE)
best_boost_1
summary(best_boost_1)
plot(best_boost_1)


# random searching 
fit_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5, search = "random")
set.seed(24)
best_boost <- train(default_nm~., data =student_univ[,-13] , method = "gbm", trControl = fit_control, verbose = FALSE)
best_boost
summary(best_boost)
plot(best_boost)

fit_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, search = "random")
set.seed(25)
best_boost_2 <- train(default_nm~., data =student_univ[,-13] , method = "gbm", trControl = fit_control, verbose = FALSE)
best_boost_2
summary(best_boost_2)
plot(best_boost_2)


# Defining a tuning grid to include multiple parameters
fit_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, search = "grid")
gbmGrid <- expand.grid(.interaction.depth = seq(1,9, by =2), .n.trees =seq(100, 2000, by =50), .shrinkage = c(0.001, 0.01), .n.minobsinnode = 10)
set.seed(26)
best_boost_3 <- train(default_nm ~., data= student_univ[,-13], method = "gbm", trControl = fit_control, tuneGrid = gbmGrid, verbose = FALSE)
best_boost_3
summary(best_boost_3)
plot(best_boost_3)


# Adjusting the parameters since more trees under the 0.01 shrinkage parameters seems to increase accurary

fit_control <- trainControl(method = "repeatedcv", number = 10, repeats = 10, search = "grid")
gbmGrid <- expand.grid(.interaction.depth = seq(3,9, by =2), .n.trees =seq(1500, 3000, by =150), .shrinkage = c(0.01, 0.1), .n.minobsinnode = 10)
set.seed(27)
best_boost_4 <- train(default_nm ~., data= student_univ[,-13], method = "gbm", trControl = fit_control, tuneGrid = gbmGrid, verbose = FALSE)
View(best_boost_4$results)
best_boost_4
summary(best_boost_4)
plot(best_boost_4)

# Logistic Regression
fit_control <- trainControl(method = "repeatedcv", number = 10, repeats = 5)
seet.seed(28)
glm_fit <- train(default_nm~., data = student_univ[,-13], method = "glm", trControl = fit_control)
glm_fit
summary(glm_fit)

# adding a nonlinear variable
modified_data <- student_univ %>% 
  transmute(limit_bal=(limit_bal^2), sex = sex, educ= educ, marriage = marriage, age = I(1/age),
            repay_period = repay_period, status = status, default_nm = default_nm, pay_period = pay_period,
            pay_amt = I(pay_amt^2), bill_period = bill_period, bill_amt = I(bill_amt^2), ID = ID)

set.seed(29)
glm_fit_2 <- train(default_nm~., data = modified_data[,-13], method = "glm", trControl = fit_control)
glm_fit_2
summary(glm_fit_2)

glm_predict <- predict()


# number of default
student_univ %>%
  count(default_nm)

credit_c.d_ts %>%
  filter(educ == "others") %>% count(default_nm)

credit_c.d_ts %>%
  filter(educ == "high school") %>% count(default_nm)


