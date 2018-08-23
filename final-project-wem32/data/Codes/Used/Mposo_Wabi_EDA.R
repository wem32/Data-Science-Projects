# Performing an Exploratory Data analysis

# loading packages
library(modelr)
library(tidyverse)
library(ggplot2)
library(knitr)
library(readxl)


# loading data
credit_c.d <- read_excel("data/unprocessed/default of credit card clients.xls")

str(credit_c.d)

credit_c.d[2,]
# getting a quick glance of the data
View(head(credit_c.d, 5))

# five number summary
lapply(credit_c.d[, -c(1, 3:12)], summary)
lapply(credit_gather[, -c(2:4, 6:8, 10)], summary)

# Visualizing credit amount based on gender, education, and marriage status
credit_c.d %>%
  ggplot(aes(x= limit_bal)) +
  geom_freqpoly(aes(color= sex)) +
  facet_grid(educ~marriage)

# visualizing default
credit_c.d %>%
  ggplot(aes(x=default_pay))+
  geom_bar()

# There are many more credit card payment default for next month than payment on schedule. This is somewhat
# to be expected since this was a period approaching the financial crisis where there was easy access to credit.


credit_gather %>%
  ggplot(aes(x= repay_period, y = limit_bal)) +
  geom_boxplot()

# It seems that lower credit limit balance is associated with longer delayed time until repayment. 

credit_gather %>%
  group_by(educ) %>%
  summarise(avg_bal = mean(limit_bal), avg_pay = mean(pay_amt)) %>%
  ggplot(aes(x = avg_pay, y = avg_bal, color = educ)) +
  geom_point(size =5) 
# It is amazing to see the gap between the credit limit and payment capacity of those with university and grad school diploma.
# repayment status based on age


# It seems that payments to credit balance is definitely non- linear as a relates to age
credit_gather %>%
  group_by(age, marriage) %>%
  summarise(X1 = sum(pay_amt/1000), Z1 = sum(bill_amt)/1000) %>%
  ggplot(aes(x=Z1, y=X1 , color = marriage)) +
  geom_point() +
  geom_smooth(se = F)

# The relationship amount billed and payment made is linear, which is what we expect if people are paying
# what they fully owe.

## Performing principal component analysis

# PCA is about the variance;so, the mean should play no role. 


# We want to standardize the variable
pca_credit <- prcomp(credit_c.d[, -c(1, 3:12,25)], scale. = T)
pca_credit 
names(pca_credit)

library(ggfortify)
autoplot(pca_credit, loadings = T)
autoplot(pca_credit, data = credit_c.d, colour = 'educ', loadings = T)


library(cluster)
autoplot(clara(credit_c.d[, -c(1, 3:12,25)]))





# time series look at the data
dat %>%
  gather(key = "pay_period", value = "pay_amt", starts_with("pay_")) %>% View()

table1 <- credit_c.d %>%
  gather(key = "pay_period", value = "pay_amt", starts_with("pay")) %>% 
  transmute(pay_period =pay_period, pay_amt = pay_amt)

table2 <- credit_c.d %>%
  gather(key = "bill_period", value = "bill_amt", starts_with("bill")) %>%
  transmute(bill_period = bill_period, bill_amt = bill_amt)

table3 <- credit_c.d %>%
  gather(key = "repay_period", value = "status", starts_with("repay")) %>%
  transmute(limit_bal = limit_bal, sex = sex, educ = educ, marriage = marriage, age = age, repay_period = repay_period, status = status)

credit_gather <- data.frame(table3, table1, table2)

# Looking at the distribution of the data
credit_gather %>%
  ggplot(aes(x =status)) +
  geom_bar()

credit_gather %>%
  ggplot(aes(x = age)) +
  geom_histogram()

credit_gather %>%
  ggplot(aes(x= limit_bal)) +
  geom_histogram( bin = 20)

credit_gather %>%
  ggplot(aes(x = pay_amt)) +
  geom_histogram()

# Visualization  using scatterplots
credit_gather %>%
  ggplot(aes(x = limit_bal, y = pay_amt, color = sex)) +
  geom_point() +
  facet_grid(educ~marriage)

credit_gather %>%
  ggplot(aes(x = bill_amt, y = pay_amt, color = sex)) +
  geom_point() +
  facet_grid(educ~marriage)

# visualization with boxplot

credit_gather %>%
  ggplot(aes(x=status, y = limit_bal)) +
  geom_boxplot()


credit_gather %>%
  ggplot(aes(x=educ, y = limit_bal)) +
  geom_boxplot()


credit_gather %>%
  ggplot(aes(x=marriage, y = limit_bal)) +
  geom_boxplot()

credit_gather %>%
  ggplot(aes(x=sex, y = limit_bal)) +
  geom_boxplot()

# Visualizing two categorical variables
credit_gather %>%
  ggplot(aes(x=educ, y = status)) +
  geom_count()

credit_gather %>%
  ggplot(aes(x =marriage, y = status)) +
  geom_count()
