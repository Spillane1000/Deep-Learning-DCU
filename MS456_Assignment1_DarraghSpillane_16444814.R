
###### loading in packages used, using install.packages to ensure most up to date version of each package

install.packages("rattle")
install.packages("e1071")
install.packages("rpart")
install.packages("dplyr")
install.packages("caret")
install.packages("randomForest")
install.packages("rpart.plot")
install.packages("ada")
library(caret)
library(randomForest)
library(rpart)
library(rpart.plot)
library(ada)
library(dplyr)
library(e1071)
library(RColorBrewer)
library(rattle)
##################

############ Task 0 data set up (train/test split) ######################## 
full_data = read.csv("ha1_data.csv")
full_data = full_data %>%
  mutate(CLAIMFLAG = factor(CLAIMFLAG,levels = c("No","Yes")),
         MARSTATUS = factor(MARSTATUS,levels = c("No","Yes" )),
         GENDER = factor(GENDER,levels = c("F","M")),
         CARUSE = factor(CARUSE,levels = c("Private","Commercial")),
         CARTYPE = factor(CARTYPE,levels = c( "SUV","Minivan","PanelTruck","Pickup","Van","SportsCar" )),
         EDUCATION = factor(EDUCATION,levels = c("Bachelors","HighSchool","Lower","Masters","PhD")))      # ordered?
glimpse(full_data)

create_train_test <- function(data, size = 0.8, train = TRUE) {
  n_row = nrow(data)
  total_row = size * n_row
  train_sample = c(1: total_row)
  if (train == TRUE) {
    return (data[train_sample, ])
  } else {
    return (data[-train_sample, ])
  }
}

data_train = create_train_test(full_data,0.8 ,train = TRUE)   # set up training set
data_test = create_train_test(full_data,0.8,train = FALSE)   # set up testing set

prop.table(table(data_train$CLAIMFLAG))    # tables to check whether proportion of claimflag values
prop.table(table(data_test$CLAIMFLAG))     # are consistent between training and testing sets


#       No       Yes 
#0.6906667 0.3093333

#       No       Yes 
#0.6966667 0.3033333



################# Task 1 ###################
par(mfrow = c(1,1))
#### task 1.1

default_tree = rpart(CLAIMFLAG~.,data = data_train,method = "class")  # trains default rpart decision tree


# function to return test training accuracy and kappa given a model
TrTe_Acc <- function(model,data_train,data_test,type = "class") {
  my_prediction <- predict(model, data_train,type)
  confmat<-confusionMatrix(my_prediction, data_train$CLAIMFLAG)
  c1 <- c(confmat$overall['Accuracy'],confmat$overall['Kappa'])
  my_prediction <- predict(model, data_test,type)
  confmat<-confusionMatrix(my_prediction, data_test$CLAIMFLAG)
  c2 <- c(confmat$overall['Accuracy'],confmat$overall['Kappa'])
  c12 <- c(c1,c2)
  names(c12) <- c('Train Acc', 'Train Kappa', 'Test Acc', 'Test Kappa')
  c12
}

TrTe_Acc(default_tree,data_train,data_test)

# Train Acc Train Kappa    Test Acc  Test Kappa 
#0.7273333   0.2514203   0.7286667   0.2429315



### task 1.2

rpart.plot(default_tree)
# conditions needed to be satisfied to be categorized in smallest leaf
# Claim frequency greater than one
# Education - lower or highschool
# use of car - private 
# income - greater than or equal to 45,000 Euro
# if all conditions met prediction for CLAIMFLAG is No



### task 1.3
control = rpart.control(cp = 0,minsplit = 60)

tuned_tree = rpart(CLAIMFLAG~.,data_train,method = "class", control = control)

TrTe_Acc(tuned_tree,data_train,data_test)

# Test/train Accuracy and kappa for tuned tree
#Train Acc Train Kappa    Test Acc  Test Kappa 
#0.7658333   0.3940197   0.7106667   0.2441395


# under the tuned decision tree we have an improved training accuracy of approximately 4% and the kappa value
# increased from approx. 0.25 to approx. 0.39 however with the testing data the accuracy actually dropped by about 1%
 # and kappa dropped from o.25 to 0.24 this is most likely due to overfitting of the model to the training data,
# this indicates for this particular tuned tree (which almost definitely is not the optimal tree) should not be 
# preferred to the default tree when applying it to unseen data


######################### Task 2 ####################################

### task 2.1

set.seed(123)
default_rf = randomForest(data_train[,-14],data_train$CLAIMFLAG,type = "class")



TrTe_Acc(default_rf,data_train,data_test)

# default random forest training/testing Accuracy and Kappa
#Train Acc Train Kappa    Test Acc  Test Kappa 
#0.9998333   0.9996099   0.7380000   0.3036195



### Task 2.2
varImpPlot(default_rf,sort = TRUE)

explan_importance = arrange(varImp(default_rf,scale = TRUE),Overall)
 # explan_importance = as.data.frame(default_rf$importance)
 # explan_importance = explan_importance["MeanDecreaseAccuracy"]
 # explan_importance = arrange(explan_importance,MeanDecreaseGini)
 # explan_importance = arrange(explan_importance,MeanDecreaseAccuracy)
explan_importance



 # table of explanatory variable importance in decreasing order by gini impurity
# high decrease in gini(/overall) indicates a high importance
#          Overall
#GENDER     41.81079
#MARSTATUS  53.02449
#CARUSE     68.03556
#KIDS       75.57988
#EDUCATION 134.09542
#CARTYPE   180.84152
#CLMFREQ   200.38921
#MVRPTS    212.49976
#CARAGE    214.55847
#HOMEVAL   295.52509
#TRAVTIME  318.23064
#AGE       330.19219
#INCOME    346.28910


### Task 2.3

ntrees_sizes = seq(50,600,by=50)

store_maxtrees = list()  # used for storing RF's with different numbers of trees

# loop trains and stores random forests with different number of trees in each
set.seed(123)
system.time(# runs the function AND also shows the computation time
  for (ntree in ntrees_sizes) {
    print(ntree)
    set.seed(123)
    rf_maxtrees = randomForest(data_train[,-14],data_train$CLAIMFLAG,
                         type = "class",
                         ntree = ntree)
    key = toString(ntree)
    store_maxtrees[[key]] = rf_maxtrees
  }
)



set.seed(123)
metrics =  matrix(0L,nrow = length(ntrees_sizes),ncol = 5) # used to store the metrics for each model with diff tree
                                                           
colnames(metrics) = c("ntrees","Train Accuracy","Train Kappa","Test Accuracy","Test Kappa") # metrics to be stored

# retrieves the train/test accuracies and Kappa's for each random forest with different number of trees
for (forest in 1:length(ntrees_sizes)) {
  metrics[forest,] = c(ntrees_sizes[forest],TrTe_Acc(store_maxtrees[[forest]],data_train,data_test))
}
metrics = as.data.frame(metrics)

# outputs test accuracy and kappa for each number of trees
test_metrics = as.data.frame(c(metrics["ntrees"],metrics["Test Accuracy"],metrics["Test Kappa"]))
test_metrics

# graphs of how random forest accuracy develops with respect to tree size 
par(mfrow = c(1,2)) # allows 2 graphs to show side by side 

plot(ntrees_sizes,metrics[,2],type = "l",main = "Training Accuracy",xlab = "Number of Trees",ylab = "Accuracy")
plot(ntrees_sizes,metrics[,4],type = "l",main = "Test Accuracy",xlab = "Number of Trees",ylab = "Accuracy")

#ntrees Test.Accuracy Test.Kappa
#1      50     0.7320000  0.2930915
#2     100     0.7373333  0.3100228
#3     150     0.7406667  0.3107074
#4     200     0.7400000  0.3084534
#5     250     0.7366667  0.3020215
#6     300     0.7393333  0.3090896
#7     350     0.7373333  0.3013606
#8     400     0.7360000  0.2978142
#9     450     0.7353333  0.2945650
#10    500     0.7386667  0.3049070
#11    550     0.7380000  0.3016727
#12    600     0.7386667  0.3029624

# from the table and graph the optimal number of trees to maximize the test accuracy is 150 which gives accuracy 
# 0.7406667 whereas the default random forest which has 500 trees gives accuracy of 0.7386667 which is
# very close to maximum value given by the 150 tree forest which suggests the default tree performs decently compared 
# to the optimal forest "size".it is likely that the forests with larger amount of trees overfit the training data which
#explains the decrease in their test accuracy




### task 2.4

set.seed(123)
opt_tree = 150  

ntree_tuned_rf = randomForest(data_train[,-14],data_train$CLAIMFLAG,ntree = opt_tree) # tuned RF with optimum tree no.

# Accuracy of tuned tree with full amount of explanatory variables
full_acc = as.numeric(TrTe_Acc(ntree_tuned_rf,data_train,data_test)["Test Acc"]) 


modified_acc = full_acc
explanatories = c(rownames(explan_importance)) # all explanatory varables taken from work in part 2.2
new_explan = explanatories


store_models = list() 
index = 1
accuracy_differences = matrix()

# while loop removes attributes used to train model until model is found with accuracy difference greater than 5%
# Note: importance of each variable will change after each consecutively least important variable is removed thus 
# order of importance may change for simplification it is assumed that the order of importance does not change as 
# explanatory variables are removed from the model

while(full_acc - modified_acc < 0.05){
  new_explan = new_explan[-1]       # removes the top explanatory variable which has least importance as list is ascending
  modified_rf = randomForest(data_train[new_explan],data_train$CLAIMFLAG,ntree = opt_tree)    # train rf with one less explanatory
  store_models[[index]] = modified_rf     
  modified_acc = as.numeric(TrTe_Acc(modified_rf,data_train,data_test)["Test Acc"])     # update the accuracy to continue through loop 
  
  accuracy_differences[index] = full_acc - modified_acc    # difference betweeen full model accuracy and modified 
  index = index + 1     # for indexing purposes 

}

# last model stored will have accuracy difference more than 5% so will be removed
# if statement to prevent incorrect model being removed in the case the code is run more than once
if(length(accuracy_differences) == length(store_models)){
  store_models = store_models[-length(store_models)]
}

# new last model will be the one with maximum variables removed while keeping accuracy within 5% of the full model 
modified_rf = store_models[[length(store_models)]]
modified_acc = as.numeric(TrTe_Acc(modified_rf,data_train,data_test)["Test Acc"])

modified_rf$importance
modified_acc
full_acc - modified_acc

#explanatory variables left after reducing the amount by importance and keeping accuracy within 5% 
# Note: there are 6 explanatory variables left compared to the original 13 so 7 can be removed while keeping 
# the accuracy loss below 5%

#MeanDecreaseGini
#MVRPTS           269.1399
#CARAGE           303.4166
#HOMEVAL          443.3201
#TRAVTIME         471.4950
#AGE              455.9233
#INCOME           547.2234

# accuracy of the revised model with max variables removed
# [1] 0.6973333

# difference between accuracy of full model and revised model
# [1] 0.04333333

###################### Task 3 ####################################

### task 3.1 
set.seed(123)
cv_folds = createFolds(data_train$CLAIMFLAG,k = 10,returnTrain = TRUE) # creates the folds for cross-validation 


### task 3.2

set.seed(123)
# defines the train control parameters for cross validation to be used in the ada boost model
train_control = trainControl(method = "cv",
                           number = 10, # number of folds K
                           search = "grid",
                           classProbs = TRUE,
                           index = cv_folds)
set.seed(123)
#trains a model using ada boost with grid search for the max. depth for each tree and training control parameters
# defined above
ada_trees = train(data_train[,-14],data_train[,14],method = "ada",
                  tuneGrid = expand.grid(nu = 0.1,iter = 10,maxdepth = seq(2,10,by = 1)),
                  trControl = train_control,
                  metric = "Accuracy")

ada_trees

#Boosted Classification Trees 

#6000 samples
#13 predictor
#2 classes: 'No', 'Yes' 

#No pre-processing
#Resampling: Cross-Validated (10 fold) 
#Summary of sample sizes: 5400, 5400, 5399, 5401, 5401, 5400, ... 
#Resampling results across tuning parameters:
  
#  maxdepth  Accuracy   Kappa    
#2        0.7199945  0.2121687
#3        0.7239945  0.2435792
#4        0.7266657  0.2590770
#5        0.7278251  0.2721205
#6        0.7298234  0.2825784
#7        0.7264970  0.2748034
#8        0.7321576  0.3009181
#9        0.7281662  0.2950176
#10        0.7286620  0.2975262

#Tuning parameter 'iter' was held constant at a value of 10
#Tuning parameter 'nu' was held constant at a value of 0.1
#Accuracy was used to select the optimal model using the largest value.
#The final values used for the model were iter = 10, maxdepth = 8 and nu = 0.1.


tune_grid = ada_trees$bestTune
set.seed(1234)
 # ada boosted model with tuned parameters found above 
boosted = train(data_train[,-14],data_train[,14],method = "ada",
                tuneGrid = tune_grid,
                trControl = train_control,
                metric = "Accuracy")

# small revision in TrTe_Acc function defined at start to prevent error caused when type = "class" is included
# in the predict function within the TrTe_Acc function itself
TrTe_Acc <- function(model,data_train,data_test) {
  my_prediction <- predict(model, data_train)
  confmat<-confusionMatrix(my_prediction, data_train$CLAIMFLAG)
  c1 <- c(confmat$overall['Accuracy'],confmat$overall['Kappa'])
  my_prediction <- predict(model, data_test)
  confmat<-confusionMatrix(my_prediction, data_test$CLAIMFLAG)
  c2 <- c(confmat$overall['Accuracy'],confmat$overall['Kappa'])
  c12 <- c(c1,c2)
  names(c12) <- c('Train Acc', 'Train Kappa', 'Test Acc', 'Test Kappa')
  c12
}
TrTe_Acc(boosted,data_train,data_test)
#Train Acc Train Kappa    Test Acc  Test Kappa 
#0.8060000   0.4937170   0.7213333   0.2535981 

# the best performer in terms of accuracy and in terms of kappa was with the maxdepth parameter of 8



### Task 3.3

ada_trees$bestTune$maxdepth
atts = c(rownames(explan_importance))
atts_removed = list()     

# creates list which contains attribute names with least important removed consecutively
for (i in 1:(length(explanatories))) {
  atts_removed[[i]] = atts
  atts = atts[-1]
}

att_removed_models = list(length(atts_removed))  #to store models with varied amount of explantory variables removed 
results = numeric(length(atts_removed)) # to store the training accuracies of each of the models




# loops through each amount of attributes removed and trains and stores the ada model for each while storing the
# training accuracy of each 
# Note: again for simplification as in task 2.4 it is assumed that the order of importance does not change as
# explanatory variables are removed (which may not be the case)

for(nAtts in 1:length(atts_removed)){
  set.seed(1234)
  ada = train(data_train[atts_removed[[nAtts]]],data_train[,14],method = "ada",
                     tuneGrid = expand.grid(nu = 0.1,iter = c(5,10,20),maxdepth = ada_trees$bestTune$maxdepth),
                     trControl = train_control,
                     metric = "Accuracy")
  att_removed_models[[nAtts]] = ada
  
  results[nAtts] = TrTe_Acc(ada$finalModel,data_train,data_test)["Train Acc"]
}


results = t(as.data.frame(results))
colnames(results) = c("13 Atts.","12 Atts.","11 Atts.","10 Atts.","9 Atts.","8 Atts.","7 Atts.","6 Atts.",
                      "5 Atts.","4 Atts.","3 Atts.","2 Atts.","1 Atts.")
boosted_acc = TrTe_Acc(boosted,data_train,data_test)["Train Acc"]
accuracy_diff = results - boosted_acc
accuracy_diff


#       13 Atts. 12 Atts. 11 Atts.     10 Atts.      9 Atts. 8 Atts. 7 Atts.     6 Atts.     5 Atts.     4 Atts.
#results   0.0095    0.006     0.01 -0.007833333 -0.006166667  -0.016  -0.025 -0.03966667 -0.06583333 -0.07183333
#3 Atts. 2 Atts. 1 Atts.
#results -0.07833333  -0.089 -0.1145

# it can be seen from the out put above that the minimum number off attributes required keeping a training accuracy 
# within 3% is 7 which has a decrease in train accuracy of 2.5%



####################### Task 4 #######################################

### task 4.1

# based on the results in 1.3 ,2.3 and 3.2 we can conclude that the most suited model for predicting the Claim flags 
 # for the insurance company is the random forest model containing 150 trees found in task 2.3 as this produced 
 # the highest test accuracy of approximately 74% compared to test accuracies of 71% and 72% found in tasks 1.3 and 3.2 
# respectively although as these accuracies are quite close it may be possible to find a more accurate model using 
# the decision tree or ada boost methods with further tuning of parameters however this would take longer and 
# and would require more computational energy and may not prove fruitful


### task 4.2
# yes i would recomend thje company to collect more data as although models with test accuracies of around 70% 
# will give good indications of claimflag it is not high enough to rely on these models to consistently correctly
# predict the claimflags because on average about 1 in 4 will be miss classified, collection of more data may
# allow the models capture better the characteristics of what causes a claim flag and give greater accuracy in 
# predicting future claim flags. on the other hand it may not be possible to achieve any greater accuracy by collecting 
# more data due to alot of randomness in claim flags arising that cannot be explained using any model however the 
# collection of more data of this nature should be relatively easy for this company as it is itself an insurance company
# and should have easy access to this kind of data and would be easy to include to try gain greater accuracies in predicting 
# claim flags 


















