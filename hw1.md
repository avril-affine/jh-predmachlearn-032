---
title: Modelling Exercises Based on Body Movement
author: Kenneth Kihara
---

#Summary
The purpose of this project is to predict the way in which a person 
performed an exercise based off data taken from devices that measure 
various types of body movements. The model built predicts this with 
96% accuracy based on a 3-fold cross validation making the out of 
sample error 4%.

#Analysis
First, the dataset and appropriate packages need to be loaded.

```r
library(caret)
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")

x_train <- training[,-160]
y_train <- training[,160]
x_test <- testing[,-160]
```

Decided to remove the features that contained NA values, since those 
features are mostly NA.

```r
x_train[x_train=="" | x_train=="#DIV/0!"] <- NA

comp <- apply(x_train, 2, function(z) all(complete.cases(z)))
x_train <- x_train[,comp]
x_train <- x_train[,-c(1:6)]
x_test <- x_test[,comp]
x_test <- x_test[,-c(1:6)]
```

Split the data to use a portion for cross validation.

```r
set.seed(0)
inTraining <- createDataPartition(y_train, p=.75, list=FALSE)
x_cv <- x_train[-inTraining,]
x_train <- x_train[inTraining,]
y_cv <- y_train[-inTraining]
y_train <- y_train[inTraining]
```

Train the model using the k nearest neighbor method with k = 5, since 
there are 5 classes to predict. Also, used 3-fold cross validation to 
check accuracy. The data was preprocessed to choose features using the 
PCA method and center and scale to normalize the data.

```r
ctrl <- trainControl(method="repeatedcv",
                     number=3,
                     repeats=5,
                     classProbs=TRUE)
tuneknn <- expand.grid(.k=5)
knnfit <- train(x=x_train, y=y_train, method="knn",
                preProcess=c("pca","center","scale"),
                trControl=ctrl, tuneGrid=tuneknn)
```

Used the partition dataset that was set aside previously to confirm 
accuracy of the model.

```r
knntable <- table(predict(knnfit, x_cv), y_cv)
mat <- confusionMatrix(knntable)
mat$overall
```

```
##       Accuracy          Kappa  AccuracyLower  AccuracyUpper   AccuracyNull 
##     0.96227569     0.95228331     0.95655884     0.96743336     0.28446166 
## AccuracyPValue  McnemarPValue 
##     0.00000000     0.05032451
```

#Results
Accuracy was determined to be 96% making the out of sample error 4%.

Finally the test dataset was put through the model to give predictions 
on the test dataset.

```r
pred <- as.character(predict(knnfit, x_test))
pred
```

```
##  [1] "B" "A" "A" "A" "A" "E" "D" "B" "A" "A" "B" "C" "B" "A" "E" "E" "A"
## [18] "B" "B" "B"
```
