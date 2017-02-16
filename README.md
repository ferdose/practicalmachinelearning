# practicalmachinelearning
Practical Machine Learning Course Project

Ferdose

Summary

In this project, the goal is to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants to predict different fashions of the exercise Unilateral Dumbbell Biceps Curl in 5 different fashions: exactly according to the specification (Class A), throwing the elbows to the front (Class B), lifting the dumbbell only halfway (Class C), lowering the dumbbell only halfway (Class D) and throwing the hips to the front (Class E).

More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (Weight Lifting Exercise Dataset).

Loading data

The training data for this project are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv

The test data are available here: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

To load the datasets we execute:

training <- read.csv('pml-training.csv', sep = ",")
testing <-read.csv('pml-testing.csv', sep = ",")
Basic Exploration and Data Cleaning

With a simple str command (not shown due to size) on the training set we can see that we have a few problems with information being empty (‘’) or’#DIV/0!’.

So we adjust this problem as following:

training[training == '#DIV/0!'] <- NA
training[training == ''] <- NA
We also see a few columns that are not useful for prediction: * X (sequential number) * user_name * raw_timestamp_part_1 * raw_timestamp_part_2 * cvtd_timestamp * new_window * num_window

So we remove these columns from our training dataset:

training <- training[,-c(1:7)]
Also from the str command, we see that there are a few columns that have numbers but were brought as factors. Therefore we have to also adjust their types:

for(i in 1:(ncol(training)-1)){
  if(class(training[, i]) == 'factor'){    
    training[, i] <- as.numeric(as.character(training[, i]))    
  }
}
Feature Selection

We already removed 7 columns that were not useful, but now we explore columns that could be useful but aren’t.

Since we have a lot of data from accelerometers, maybe a few columns have near zero variance or even zero variance, making them not useful for prediction.

To check for near zero variance and remove any columns found, we do:

library(caret)
nzv <- nearZeroVar(training, saveMetrics = T)
removed.cols <- names(training)[nzv$nzv]
training <- training[,!(nzv$nzv)]
This way we removed the following columns from the training dataset:

cat(cat('COLUMNS REMOVED: '), cat(removed.cols, sep=', '), sep=' ')
## COLUMNS REMOVED: kurtosis_yaw_belt, skewness_yaw_belt, amplitude_yaw_belt, avg_roll_arm, stddev_roll_arm, var_roll_arm, avg_pitch_arm, stddev_pitch_arm, var_pitch_arm, avg_yaw_arm, stddev_yaw_arm, var_yaw_arm, max_roll_arm, min_roll_arm, min_pitch_arm, amplitude_roll_arm, amplitude_pitch_arm, kurtosis_yaw_dumbbell, skewness_yaw_dumbbell, amplitude_yaw_dumbbell, kurtosis_yaw_forearm, skewness_yaw_forearm, max_roll_forearm, min_roll_forearm, amplitude_roll_forearm, amplitude_yaw_forearm, avg_roll_forearm, stddev_roll_forearm, var_roll_forearm, avg_pitch_forearm, stddev_pitch_forearm, var_pitch_forearm, avg_yaw_forearm, stddev_yaw_forearm, var_yaw_forearm
Exploration Data Analysis

A useful approach is to analyse correlation of features, calculated as following:

cormax <- cor(training[,-118], use="pair")
Plotting a few variables we have:

library(corrplot)
corrplot(cormax[1:10,1:10])


We can see that we have variables with very high correlation. Plotting two of them colored by our outcome (classe), we have:

library(ggplot2)
qplot(roll_belt, total_accel_belt, data=training, color=classe, main='Plot of roll_belt by total_accel_belt per classe')


Although we can see high correlation, we chose not to remove any correlated variables, mainly because we intend to use Random Forest as our machine learning algorithm, since it deals fine with these variables.

Imputation of missing data

Also from the str command we saw that we have a lot of missing data (NA values), which might be expected since we’re dealing with lots of accelerometers.

To deal with these NA values we chose to imputate the median for each column, by executing:

trainingF <- training
preObj <- preProcess(trainingF[,-ncol(trainingF)], method="medianImpute")
trainingF <- predict(preObj, trainingF)
The preObj object has to be saved to be used later to impute data for the testing dataset.

As a final note, we see that we don’t have any major imbalance between the classes:

table(trainingF$classe)
## 
##    A    B    C    D    E 
## 5580 3797 3422 3216 3607
Machine Learning

As stated before we chose Random Forest as the algorithm, because it can deal with lots of highly correlated features and generally is a bagging algorithm with great results. Before running caret’s train function we reserve 20 cores to be used in our server, since we already antecipate a lot of processing. We also creat a trainControl object to specify 10-Fold Cross Validation one time and pass it to the train function. Then our code to create the model is:

library(doParallel)
registerDoParallel(cores=20)

cvCtrl <- trainControl('cv', 10, savePred=T)
set.seed(111)
model <- train(classe ~ ., data = trainingF, method = 'rf', trControl = cvCtrl)
The model results are:

model
## Random Forest 
## 
## 19622 samples
##   117 predictor
##     5 classes: 'A', 'B', 'C', 'D', 'E' 
## 
## No pre-processing
## Resampling: Cross-Validated (10 fold) 
## Summary of sample sizes: 17659, 17660, 17659, 17658, 17661, 17659, ... 
## Resampling results across tuning parameters:
## 
##   mtry  Accuracy   Kappa      Accuracy SD  Kappa SD   
##     2   0.9443983  0.9295828  0.004459038  0.005647919
##    59   0.9951579  0.9938749  0.002114660  0.002675397
##   117   0.9901637  0.9875570  0.003040770  0.003847427
## 
## Accuracy was used to select the optimal model using  the largest value.
## The final value used for the model was mtry = 59.
plot(model)


From the results we see that the final model has 59 randomly selected predictors of the 117 submitted, and a accuracy of 99.51579% from our 10-Fold Cross Validation.

Since these results are from 10-Fold Cross Validation, we expect the out of sample error to be 1 - 99.51579, which is 0.48421%.

Testing the final model

Finally we use our final model to evaluate the testing dataset.

First we filter the testing columns to keep only those that will be used. Right after we use the preObj object created for imputation of missing data from the training dataset - this object is used to now imputate missing values on the testing dataset but using values from the training dataset. Finally we run the model on the testing dataset to get the predicted classes.

testing <- testing[,names(testing) %in% names(trainingF)]
testingF <- predict(preObj, testing)
predict(model, newdata=testingF)
## Loading required package: randomForest
## randomForest 4.6-12
## Type rfNews() to see new features/changes/bug fixes.
## 
## Attaching package: 'randomForest'
## The following object is masked from 'package:ggplot2':
## 
##     margin
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
Observing the results given in the Course Project Prediction Quiz, we got 100% accuracy for the testing dataset.
