
install.packages("xgboost")

#install.packages("lattice")

#install.packages("ggplot2")
#install.packages("doSNOW")

install.packages("keras")


library(xgboost)
library(rpart)
library(tidyverse)
library(AER)
library(keras)



df1 <- read.csv("C:\\Users\\Nico\\Documents\\Data\\titanic_2.csv", sep = ",")

View(df1)

dim(df1)

# deleting unimportant vars

df1$X. <- NULL
df1$PassengerId <- NULL

table(df1$Emb_1)
table(df1$Survived)

summary(df1$Survived)
summary(df1$Sex)
summary(df1$Fare)

## Missing data 

df1 <- na.omit(df1)


## making factorial data-frame

df2 <- df1 %>%
  mutate(pclass =  1 * Pclass_1 + 2 * Pclass_2 + 3 * Pclass_3)

df2 <- df2 %>%
  mutate(title = 1 * Title_1  + 2 * Title_2 + 3 * Title_3 + 4 * Title_4)

View(df2)

df2 <- df2[-c(5:7, 9:12)]


## Visualisation


qplot( as.factor(df2$Survived), fill = I("blue") ) + 
  geom_bar() + 
  labs(x="Survived", y="Counts")

par(mfrow = c(3,1))

hist(df2$Survived[df2$pclass == 1])
hist(df2$Survived[df2$pclass == 2])
hist(df2$Survived[df2$pclass == 3])

# the higher the class, the lower the survival chance!


ggplot(df2, aes(x= Survived, y = Age)) +
  geom_point(size=2, shape=23)

cor(df2$Age, df2$Survived)

# no evident relationship

### Corrplot

install.packages("ggcorrplot")
library(ggcorrplot)

corr1 <- cor(df2)
ggcorrplot(corr1)


# high corr w/ Sex, Fare, Emb_1, Emb_3 , pclass

imp_vars <- c("Survived", "Sex", "Emb_1", "Emb_3" , "pclass", "Fare")


### Normalizing both dfs (df2 & df3)

to_scale <- scale(df2[, c("Age", "Fare")])

df2$Age <- NULL
df2$Fare <- NULL

to_scale <- data.frame(to_scale)

main_df <- cbind(df2, to_scale)

dim(main_df)
View(main_df)


## main df only3 w/ "important" vars

df_main_red <- main_df[, imp_vars]




######################
# Splitting the data

## for main_df

set.seed(3)
dt = sort(sample(nrow(main_df), nrow(main_df)*.7))
train<- df3[dt,]
test<-df3[-dt,]


prop.table(table(train$Survived))
prop.table(table(test$Survived))

## for reduced_df



set.seed(3)
dt = sort(sample(nrow(df_main_red), nrow(df_main_red)*.7))
train_red <- df3[dt,]
test_red <-df3[-dt,]

prop.table(table(train_red$Survived))
prop.table(table(test_red$Survived))



################
## Algorithms ##
################

# to compare the results
algos <- c()
score_normal_df <- c()
score_reduced_df <- c()

###############################
### LogReg

#append(vector, data, after)

model1 <- glm(Survived ~., family=binomial(link='logit'),data= train)
summary(model1)

anova(model1, test="Chisq")

pred1 <- predict(model1, test,type='response')
pred1 <- ifelse(pred1 > 0.5,1,0)

error1 <- mean(pred1 != test$Survived)
print(paste('Accuracy',1-error1))


## w/ reduced model

model2 <- glm(Survived ~., family=binomial(link='logit'),data= train_red)
summary(model1)

pred2 <- predict(model1, test_red,type='response')
pred2 <- ifelse(pred2 > 0.5,1,0)

error2 <- mean(pred2 != test_red$Survived)
print(paste('Accuracy',1-error2))

## appending vectors

algos <- c(algos, "LogReg")
score_normal_df <- c(score_normal_df, 1-error1)
score_reduced_df <- c(score_reduced_df, 1-error2)


######################################
## Decision trees


install.packages("rpart")
library(rpart)

## for normal df

tree1 <- rpart(Survived ~ ., 
               train, method="class")

tree1


pred_tree_1 <- predict(tree1, test)

pred_tree_1 <- ifelse(pred_tree_1 > 0.5,1,0)


error_tree_1 <- mean(pred_tree_1 != test$Survived)
print(paste('Accuracy',1-error_tree_1))


## for reduced df

tree2 <- rpart(Survived ~ ., 
               train_red, method="class")

tree2


pred_tree_2 <- predict(tree2, test_red)


pred_tree_2 <- ifelse(pred_tree_2 > 0.5,1,0)

error_tree_2 <- mean(pred_tree_2 != test$Survived)
print(paste('Accuracy',1-error_tree_2))

##
algos <- c(algos, "Dec Tree")
score_normal_df <- c(score_normal_df, 1-error_tree_1)
score_reduced_df <- c(score_reduced_df, 1-error_tree_2)



#### XGBoosted Trees

## for normal df

# automatic cross-v

xg_1 <- xgb.cv(data = as.matrix(df_main[,-1]), label = as.matrix(df_main[, 1]), nfold = 5,
               nrounds = 4, objective = "binary:logistic")

mean_test_error_xg1 <- xg_1$evaluation_log$test_error_mean[4]
print("Mean test error:"  + mean_test_error_xg1)


## for reduced tree

xg_2 <- xgb.cv(data = as.matrix(df_main_red[,-1]), label = as.matrix(df_main_red[, 1]), nfold = 5,
               nrounds = 4, objective = "binary:logistic")

mean_test_error_xg2 <- xg_2$evaluation_log$test_error_mean[4]
print("Mean test error:"  + mean_test_error_xg2)


##
algos <- c(algos, "X Gradient Boosting")
score_normal_df <- c(score_normal_df, 1-mean_test_error_xg1)
score_reduced_df <- c(score_reduced_df, 1-mean_test_error_xg2)


##############################
###### Neural networks

train <- data.frame(train)
test <- data.frame(test)

train_red <- data.frame(train_red)
test_red <- data.frame(test_red)

nn1 <- keras::keras_model_sequential()

nn1 <- keras_model_sequential() 

nn1 %>% 
  layer_dense(200, activation = "relu", input_shape = c(70)) %>% 
  layer_dropout(rate = 0.4) %>% 
  layer_dense(200, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(1)


summary(nn1)


nn1 %>% compile(
  loss = "mean_squared_error",
  optimizer = optimizer_rmsprop(),
  metrics = c("accuracy")
)


iter1 <- nn1 %>% fit(
  train[, -1], train[, 1], 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)


nn1 %>% evaluate(train[, -1], train[, 1],verbose = 0)

pred_nn1 <- nn1 %>% predict(test[, -1])
pred_nn1 <- ifelse(pred_nn1 > 0.5,1,0)

error_nn1 <- mean(pred_nn1 != test$Survived)
print(paste('Accuracy',1-error_nn1))


## w/ reduced DF

nn2 <- nn1

iter1 <- nn2 %>% fit(
  train_red[, -1], train_red[, 1], 
  epochs = 30, batch_size = 128, 
  validation_split = 0.2
)


nn2 %>% evaluate(train_red[, -1], train_red[, 1],verbose = 0)

pred_nn2 <- nn2 %>% predict(test_red[, -1])
pred_nn2 <- ifelse(pred_nn2 > 0.5,1,0)

error_nn2 <- mean(pred_nn2 != test$Survived)
print(paste('Accuracy',1-error_nn2))


algos <- c(algos, "Neural NW")
score_normal_df <- c(score_normal_df, 1-error_nn1)
score_reduced_df <- c(score_reduced_df, 1-error_nn2)


########################################
## comparing the algos

df_compare <- data.frame(algos, score_normal_df, score_reduced_df)
View(df_compare)






