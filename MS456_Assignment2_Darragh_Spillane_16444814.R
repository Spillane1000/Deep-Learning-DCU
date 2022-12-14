library(dplyr);library(e1071);library(ggplot2);library(ISLR);library(foreign);library(caret);library(nortest)
############################ Data Set Up #############################################
full_data = read.csv("ha2_data.csv")   

# function that creates train/test split
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

# check that proportion of clm are consistent between training and validation sets
# note proportions between training and testing data are off data should be reshuffled however later question refers 
# to this inbalance so it is left for the moment 
prop.table(table(data_train$clm))
prop.table(table(data_test$clm))

# quick overview of the data
glimpse(full_data)

# Training clm proportions
#    0         1 
#0.5360825 0.4639175 
#testing clm proportions
#        0         1 
#0.4650206 0.5349794 
################################ Task 1 ########################################
#### Task 1.1
# initialize weights, learning rates and epochs
m = 20
r = c(0.5,1,1.5)
w0 = 1
w1 = 1
w2 = 1

# remove features noty used in question ie. keep only veh_value , exposure and claim for training and validation sets
training = data_train %>%
  select(veh_value,exposure,clm)
validation = data_test %>%
  select(veh_value,exposure,clm)

# filter to find only rows which have a claim (clm = 1) and do not have a claim (clm = 0) (used in plotting)
clm_occured = filter(training,clm == 1)
clm_not_occured = filter(training,clm == 0)

# plot veh_value and exposure with instances where claim occured (clm = 1) in red and where no claim occured in blue
# Note: exposure on x-axis although correspond to w2 in models just to make the graph more clear 
plot(training$exposure,training$veh_value, type='n', xlab='Exposure', ylab='veh_value')
points(clm_occured$exposure,clm_occured$veh_value,col = "red")
points(clm_not_occured$exposure,clm_not_occured$veh_value,col = "blue")
legend(0.05,19,legend = c("clm = 1","clm = 0"),col = c("red","blue"),pch=1, cex=1)
# used to store the models for different learning rates
store_learnings = list(3)

# used to store the accuracies and weights of the perceptron for different learning rates
Lrates_comp = matrix(0L,nrow = length(r),ncol = 5)
colnames(Lrates_comp) = c("Learning_Rate","Accuracy","w0","w1","w2")

#loop through different learning rates
for (j in 1:length(r)) {
  acc_weights = matrix(0L,ncol = 5 , nrow = m)  # stores the accuracy and weights for each epoch
  colnames(acc_weights) = c("Epoch no.","Accuracy","w0","w1","w2")
  
  # re-initialize weights when new learning rate is used 
  w0 = 1
  w1 = 1
  w2 = 1
  # loop to learn perceptron for each epoch weights found in the previous epoch are used at the start of the next
  # allows percpetron to see the data more than once 
  set.seed(111)
  for (i in 1:m) {
    # We reshuffle the order of the data point for each epoch.
    index = 1:(length(training$clm))  
    index = sample(index)
    
    # loop for each data point update weights if a mistake is made
    for (k in index) {
      clm_k = w0 + w1*training$veh_value[k] + w2*training$exposure[k] 
      
      if(clm_k >= 0){
      pred_k = 1
      }
      else{
        pred_k = 0
      }
      # if no mistake is made term inside the brackets is 0 and no update is made 
      # if no clm is predicted but a claim did occur weights are promoted (ie. multiplying by 1)
      # if claim predicted but no claim occured weights are demoted (i.e multiplying by -1)
      w0 = w0 + r[j]*(training$clm[k]- pred_k)*1
      w1 = w1 + r[j]*(training$clm[k] - pred_k)*training$veh_value[k]
      w2 = w2 + r[j]*(training$clm[k] - pred_k)*training$exposure[k]
    }
    # change all predictions to binaray prediction (0 or 1) instead of the current contionuos values 
    clm_all = w0 + w1*training$veh_value + w2*training$exposure 
    clm_pred = clm_all
    clm_pred[clm_all >= 0] = 1
    clm_pred[clm_all < 0] = 0
    
    # accuracy for each epoch 
    accuracy = sum(clm_pred == training$clm)/length(training$clm)
    
    # option to print accuracy and weights for each epoch aswell as include plot for each classifier 
    #print(paste("epoch ",i," accuracy = ",accuracy, "w0 = ",w0," W1 = ",w1," w2 = ",w2))
    #abline(a = -1.0*w0/w1, b = -1.0*w2/w1, col='red', lwd=3, lty=2)
     acc_weights[i,] = c(i,accuracy,w0,w1,w2)
  }
  acc_weights = as.data.frame(acc_weights)
  max_acc = filter(acc_weights,acc_weights$Accuracy == max(acc_weights$Accuracy))  # extracts the weights that gave the maximum accuracy over all epochs
  store_learnings[[j]] = acc_weights
  Lrates_comp[j,] = c(r[j],max_acc$Accuracy,max_acc$w0,max_acc$w1,max_acc$w2)
}
Lrates_comp = as.data.frame(Lrates_comp)
Lrates_comp

#### Task 1.2
training_results = as.data.frame(Lrates_comp)

# for storing validation results
val_results = matrix(0L,nrow = length(r),ncol = 2)
colnames(val_results) = c("Learning_Rate","Validation_Accuracy")

# loop find validation accuracies for each learning rate
for (i in 1:length(r)) {
  w0 = training_results$w0[i]
  w1 = training_results$w1[i]
  w2 = training_results$w2[i]
  # plot the optimal perceptron classifier for each learning rate 
  # Note: exposure is on x-axis in plot but w2 is the corresponding weight so w1 and w2 have swapped places compared to the method in labs
  abline(a = -1.0*w0/w1, b = -1.0*w2/w1, col='red', lwd=3, lty=2)
  
  clm_all = w0 + w1*validation$veh_value + w2*validation$exposure 
  clm_pred = clm_all
  clm_pred[clm_all >= 0] = 1
  clm_pred[clm_all < 0] = 0
  accuracy = sum(clm_pred == validation$clm)/length(validation$clm)
  val_results[i,] = c(r[i],accuracy)
}
val_results = as.data.frame(val_results)
val_results

# Learning_Rate  Accuracy   w0     w1       w2
#1         0.5 0.6051546 -1 -0.03685 2.305955
#2         1.0 0.6030928 -2  0.44620 3.110883
#3         1.5 0.6000000 -2  0.29830 3.262834

#  Learning_Rate Validation_Accuracy
#1           0.5           0.6172840
#2           1.0           0.6296296
#3           1.5           0.6255144

# a learning rate of 1.0 appears to produce the best validation accuracy which is the best measure of performance,
# if you include the plots of each classifier for each epoch (remove comment on line 130 and run code again),
# they seem to fluctuate around the bottom of  the graph which indicates that there is no convergence which suggests that the data is not linearly seperable

#### perceptron weights with optimum validation accuracy
w0_perc = Lrates_comp$w0[2]
w1_perc = Lrates_comp$w1[2]
w2_perc = Lrates_comp$w2[2]

# optimal perceptron training and validation accuracies
perc_results = c(Lrates_comp$Accuracy[2],val_results$Validation_Accuracy[2])

#### Task 1.3
# everything here is the same as in the previous task except we are not looping for different 
# learning rate and we are using Winnow update rule
m = 20
r = 0.1   # learning rate found in previous task 
w0 = 1
w1 = 1
w2 = 1

set.seed(1234)
for (i in 1:m) {
  # We reshuffle the order of the datapoint for each epoch.
  index = 1:(length(training$clm))  # 2*
  index = sample(index)
  
  for (k in index) {
    clm_k = w0 + w1*training$veh_value[k] + w2*training$exposure[k]
    if(clm_k >= 0){
      pred_k = 1
    }
    else{
      pred_k = 0
    }
    # updates using winnow update rule 
    w0 = -1  # kept constant throughout
    w1 = w1*exp(r*(training$clm[k] - pred_k)*training$veh_value[k]) 
    w2 = w2*exp(r*(training$clm[k] - pred_k)*training$exposure[k])
  }
  clm_all = w0 + w1*training$veh_value + w2*training$exposure 
  clm_pred = clm_all
  clm_pred[clm_all >= 0] = 1
  clm_pred[clm_all < 0] = 0
  
  accuracy = sum(clm_pred == training$clm)/length(training$clm)
  # print(paste("epoch ",i," accuracy = ",accuracy, "w0 = ",w0," W1 = ",w1," w2 = ",w2))
  acc_weights[i,] = c(i,accuracy,w0,w1,w2)
}
# weights and accuracy for each epoch 
acc_weights
acc_weights = as.data.frame(acc_weights)
max_acc = filter(acc_weights,acc_weights$Accuracy == max(acc_weights$Accuracy))

# store optimal weights for winnow
w0_win = max_acc$w0
w1_win = max_acc$w1
w2_win = max_acc$w2

clm_all = w0_win + w1_win*validation$veh_value + w2_win*validation$exposure 
clm_pred = clm_all
clm_pred[clm_all >= 0] = 1
clm_pred[clm_all < 0] = 0
val_accuracy = sum(clm_pred == validation$clm)/length(validation$clm)
########## Task 1.4
# results from winnow model 
win_results = c(max_acc$Accuracy,val_accuracy)
win_results

# winnow training and validation accuracy
# train_acc val_acc
#0.5958763 0.5925926

# from the results of the Winnow and perceptron algorithms it appears the perceptron performs better 
# both classifiers are similar as they slope in the same direction. i beleive the perceptron performs better
# due to the fact the learning rate was tuned using the perceptron and the fact that the data does not seem 
# linearly seperable because of this the superior accuracies could just be down to chance. the winnow could 
# be improved by adding an update rule for w0 or by tuning for the learning rate

####################### Task 2 #################################
#### Task 2.1
training$clm = as.factor(training$clm) # change to factor to ensure svm function learns as classification instead of regression
set.seed(1234)
svm_linear = svm(training$clm~.,data = training,kernel = "linear", cost = 1)

#grid search for radial kernel svm for different cost and gamma values (gamma is the variable inside the radial kernel)
set.seed(678)
svm_radial_tune = tune.svm(clm ~ veh_value + exposure , data = training,kernel = "radial",
                          cost = c(0.01, 0.1, 0.5, 1, 10, 50),gamma = c(0.2, 0.5, 1, 2))
summary(svm_radial_tune)
svm_linear
# radial svm tuning output 
#Parameter tuning of 'svm':
#  - sampling method: 10-fold cross validation 
#- best parameters:
#  gamma cost
#1  0.1
#- best performance: 0.3969072 
#- Detailed performance results:
 # gamma  cost     error dispersion
#1    0.2  0.01 0.4639175 0.04506831
#2    0.5  0.01 0.4639175 0.04506831
#3    1.0  0.01 0.4639175 0.04506831
#4    2.0  0.01 0.4639175 0.04506831
#5    0.2  0.10 0.4103093 0.06007360
#6    0.5  0.10 0.4103093 0.04605324
#7    1.0  0.10 0.3969072 0.04792552
#8    2.0  0.10 0.4000000 0.04781451
#9    0.2  0.50 0.4072165 0.05194576
#10   0.5  0.50 0.4030928 0.04672781
#11   1.0  0.50 0.3989691 0.04125143
#12   2.0  0.50 0.3989691 0.04428859
#13   0.2  1.00 0.4113402 0.05014121
#14   0.5  1.00 0.4051546 0.04320887
#15   1.0  1.00 0.3979381 0.03770685
#16   2.0  1.00 0.4103093 0.04475277
#17   0.2 10.00 0.4041237 0.03394939
#18   0.5 10.00 0.4010309 0.03203442
#19   1.0 10.00 0.4061856 0.04129435
#20   2.0 10.00 0.4061856 0.04512068
#21   0.2 50.00 0.3979381 0.04564112
#22   0.5 50.00 0.4030928 0.03871131
#23   1.0 50.00 0.4061856 0.03770685
#24   2.0 50.00 0.3979381 0.03511214

# linear SVM details
#Call:
#  svm(formula = training$clm ~ ., data = training, kernel = "linear", cost = 1)
#Parameters:
#  SVM-Type:  C-classification 
#SVM-Kernel:  linear 
#cost:  1 
#Number of Support Vectors:  811

# take the optimal model find in the grid search
svm_radial = svm_radial_tune$best.model

# function to out put the accuracy given the model features and true values
accuracy = function(model,features,trueVal){
  prediction = predict(model,features)
  confusion = table(prediction,trueVal)
  acc = sum(diag(confusion))/sum(confusion)
  return(acc)
}
# extract the explanatory variables from the validation set to use in accuracy function 
val_features = validation %>%
  select("veh_value","exposure")
val_acc = accuracy(svm_radial,val_features,validation$clm)

# same as above but for training accuracy. prints results for the radial SVM
train_features = training %>%
  select("veh_value","exposure")
train_acc = accuracy(svm_radial,train_features,training$clm)
svm_results = c(train_acc,val_acc)

########## Task 2.2
# method of plotting the SVM classifier as in labs modified for task at hand 
make.grid <- function(x, n=100){
  grange <- apply(x, 2, range)
  x1 <- seq(from = grange[1,1], to = grange[2,1], length = n)
  x2 <- seq(from = grange[1,2], to = grange[2,2], length = n)
  expand.grid(x.1 = x1, x.2 = x2)
}
feature_grid = make.grid(training[,-3])
colnames(feature_grid) = c("veh_value","exposure")
clm_grid = predict(svm_radial,feature_grid)
plot(feature_grid$exposure,feature_grid$veh_value, col = c("blue","red")[as.numeric(clm_grid)],pch = 20, cex = .2)
abline(a = -1.0*w0_win/w1_win, b = -1*w2_win/w1_win, col='red', lwd=3, lty=2)  # Winnow classifier 
abline(a = -1.0*w0_perc/w1_perc, b = -1*w2_perc/w1_perc, col='blue', lwd=3, lty=2) # perceptron classifier
legend(0.05,19,legend = c("Perceptron","winnow"),col = c("blue", "red"),lty = 1, cex = 0.8)

# option to overlay the training data points in the plot 
# points(clm_occured$exposure,clm_occured$veh_value,col = "green")
# points(clm_not_occured$exposure,clm_not_occured$veh_value,col = "black")

########### Task 2.3
# outputs all results together
results = matrix(0L,ncol = 2,nrow = 3)
results[1,] = perc_results 
results[2,] = win_results
results[3,] = svm_results
results = as.data.frame(results)
colnames(results) = c("Training Accuracy","Validation Accuracy")
rownames(results) = c("perceptron","Winnow","SVM")
results

#          Training Accuracy Validation Accuracy
#perceptron         0.6030928           0.6296296
#Winnow             0.5958763           0.5925926
#SVM                0.6618557           0.5637860

# although the SVM with radial kernels has the best training performance it has the worst validation accuracy,
# this could be due to the fact that th proportion of claims in training and validation data differs (in training approx
# 54% of clm = 0 where in testing approx 46% of clm = 0) or it could be due to overfitting from the svm classifier
# as seen in the output above the perceptron has the highest validation accuracy which is relatively consistent with its training accuracy
############################ Task 3 #################################################
#### Task 3.1
#learn a naive Bayes model with laplace smoothing 
NB_class = naiveBayes(clm ~ veh_value + exposure,training,laplace = 1)
print(NB_class)

#Naive Bayes Classifier for Discrete Predictors
#Call:
#  naiveBayes.default(x = X, y = Y, laplace = laplace)
#A-priori probabilities:
# Y
#0         1 
#0.5360825 0.4639175 

#Conditional probabilities:
#  veh_value
#Y       [,1]     [,2]
#0 1.779963 1.510021
#1 1.850523 1.177548

#exposure
#Y        [,1]      [,2]
#0 0.4837835 0.2979168
#1 0.6343722 0.2512011

ad.test(full_data$veh_value) # Anderson-Darling normality test
ad.test(full_data$exposure)  #

# normality test veh_value
#Anderson-Darling normality test
#data:  full_data$veh_value
#A = 50.216, p-value < 2.2e-16

#Anderson-Darling normality test
#data:  full_data$exposure
#A = 12.265, p-value < 2.2e-1

# train and test accuracy for the Naive Bayes classifier 
test_acc_NB = accuracy(NB_class,val_features,validation$clm)
train_acc_NB = accuracy(NB_class,train_features,training$clm)
print(paste("Train Accuracy: ",train_acc_NB))
print(paste(" Test Accuracy: ", test_acc_NB))

#"Train Accuracy:  0.601030927835052"
#" Test Accuracy:  0.580246913580247

# very low p-values from the Anderson-Darling normality test suggest that the normality assumption is not valid 
# the Naive Bayes seems to perform similarly to the other models however NB is more appropriate for multi-class 
# problems such as text classification and assumes independence of features which may not suit this setting 

#### Task 3.2
# step 1 converting veh_value and exposure to factors with 4 levels
full_data$veh_value = cut( full_data$veh_value, breaks=c(quantile(full_data$veh_value, probs = seq(0, 1, by = 0.25))) )
full_data$exposure = cut( full_data$exposure, breaks=c(quantile(full_data$exposure, probs = seq(0, 1, by = 0.25))) )

# loop changes remainder of variables to factors
for (i in 3:ncol(full_data)) {
  full_data[,i] = as.factor(full_data[,i])
}
# remove N/A values
full_data = full_data %>%
  na.omit()
glimpse(full_data)
# create train test split
data_train = create_train_test(full_data,0.8 ,train = TRUE)   
data_test = create_train_test(full_data,0.8,train = FALSE)
nrow(full_data)
# check to see if number of entries is the same as mention in the task (1209)
#[1] 1209

########## Task 3.3
# extracting features after data preprocessing 
train_features = data_train %>%
  select("veh_value","exposure")
test_features = data_test %>%
  select("veh_value","exposure")

# learning new Naive Bayes classifier after preprocessing
NB_class = naiveBayes(clm ~ veh_value + exposure,data_train,laplace = 1)
print(NB_class)

# outputting train and test accuracies
test_acc = accuracy(NB_class,test_features,data_test$clm)
train_acc = accuracy(NB_class,train_features,data_train$clm)
print(paste("Train Accuracy: ",train_acc))
print(paste(" Test Accuracy: ", test_acc))

#[1] "Train Accuracy:  0.595656670113754"
#[1] " Test Accuracy:  0.59504132231405]
# using this method the test accuracy increased by 1% whereas the tranining decreased by 0.5% overall the increase in test suggests an improvent
############ Task 3.4 

# learning Naive Bayes classifier using all explanatories without smoothing
NB_all = naiveBayes(clm ~ .,data_train)
print(NB_all)

test_acc_NB_all = accuracy(NB_all,data_test[,-3],data_test$clm)
train_acc_NB_all = accuracy(NB_all,data_train[,-3],data_train$clm)
print(paste("Train Accuracy: ",train_acc_NB_all))
print(paste(" Test Accuracy: ", test_acc_NB_all))

#[1] "Train Accuracy:  0.619441571871768"
#[1] " Test Accuracy:  0.553719008264463"

# learning Naive Bayes classifier using all explanatories with smoothing
NB_all_lap = naiveBayes(clm ~ .,data_train,laplace = 1)

test_acc_NB_all_lap = accuracy(NB_all_lap,data_test[,-3],data_test$clm)
train_acc_NB_all_lap = accuracy(NB_all_lap,data_train[,-3],data_train$clm)
print(paste("Train Accuracy: ",train_acc_NB_all_lap))
print(paste(" Test Accuracy: ", test_acc_NB_all_lap))

#[1] "Train Accuracy:  0.616339193381593"
#[1] " Test Accuracy:  0.553719008264463"

# in both cases training accuracy increases slightly and test accuracy decreases significantly 
# smoothing seems to have no effect on the results this may be because there is sufficient data in every category 
# or there in no probability of zero dominating the estimate anywhere

###################### Task 4 ##########################
full_data2 = read.csv("ha2_data2.csv") #load in unlabelled data set

# preprocessing of unlabeled data set
full_data2$veh_value = cut( full_data2$veh_value, breaks=c(quantile(full_data2$veh_value, probs = seq(0, 1, by = 0.25))) )
full_data2$exposure = cut( full_data2$exposure, breaks=c(quantile(full_data2$exposure, probs = seq(0, 1, by = 0.25))) )
for (i in 3:ncol(full_data2)) {
  full_data2[,i] = as.factor(full_data2[,i])
}
full_data2 = full_data2 %>%
  na.omit()
glimpse(full_data2)

new_data = full_data2 
# adds predictions with posterior probability > 0.7 to training data and retrains NB classifier until no more data added
while(nrow(new_data) > 0){
  NB_model = naiveBayes(clm ~. , data_train)
  predictions = predict(NB_model,full_data2)  # predict for unlabeled data 
  probs = predict(NB_model,full_data2,type = "raw") # type = "raw" gives same predictions but outputs the probabilities 
  probs = as.data.frame(probs)
  max_prob = numeric()
  for (i in 1:nrow(probs)) {
    max_prob[i] = max(probs[i,1],probs[i,2])  #stores the probabilities used to make each prediction 
  }
  data2_wPrediction = full_data2 %>%
    mutate("clm" = predictions,"probability" = max_prob)  # add the predictions and probabilities to unlabeled data
  
  new_data = data2_wPrediction %>%
    filter(probability > 0.7)  # selects all instances with a prediction probability >0.7
  
  new_data = new_data[,-ncol(new_data)]  # removes probability column so it can be joined with training set 
  data_train = rbind(data_train,new_data)  # joins the new data with predicted labels to labeled training data
  
  # new unlabelled data with instances that had probability >0.7 removed 
  new_unlabelled = data2_wPrediction %>%
    filter(probability <= 0.7)
  full_data2 = new_unlabelled[,-((ncol(new_unlabelled)-1):ncol(new_unlabelled))]
}
semi_train_acc = accuracy(NB_model,data_train[,-3],data_train$clm)
semi_test_acc = accuracy(NB_model,data_test[,-3],data_test$clm)
print(paste("Train Accuracy: ",semi_train_acc,"Test Accuracy: ",semi_test_acc))

#[1] "Train Accuracy:  0.937037037037037 Test Accuracy:  0.56198347107438"
########### Task 4.2
# although the training accuracy drastically increased with this model the test accuracy only improved by about 1% compared to the other
# NB methods which may not justify the increased computational power needed to run it,increasing the threshold would decrease the amount 
# of data being added to the training set from the unlabelled and would likely decrease the training accuracy.the higher threshold would not
# have a big impact on  the validation accuracy 

########### Task 4.3
# semi-supervised learning would be appropriate when the data is separable, there is less unlabelled than labelled data otherwise the unlabelled
# data will dominate the learning of the model or when the unlabelled data can be shown to have roughly the same structure/properties as the labelled































