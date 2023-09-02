---
  title: "Assignment_ML_coursera"
output:
  pdf_document: default
html_document: default
date: "2023-08-29"
---
  
  ```{r}
library(ggplot2)
library(dplyr)
library(tidytext)
library(caret)
library(rattle)
library(rpart)
library(rpart.plot)
library(randomForest)
library(repmis)
```

#1 Data
## The trainning data for the project is available in: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv
## The test data is available in: https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv

#2 Data Cleaning & Cleaning Data

```{r}
trainData <- read.csv("pml-training.csv",
                      na.strings = c("NA", "#DIV/0!", ""))
testData <- read.csv("pml-testing.csv",
                     na.strings = c("NA", "#DIV/0!", ""))

trainData <- trainData[,colSums(is.na(trainData))==0] #delete colums with all mising values
testData <- testData[,colSums(is.na(testData))==0] #delete colums with all mising values

trainData <- trainData[,-c(1:7)] #remove unnecesary data
testData <- testData[,-c(1:7)] #remove unnecesary data
```

## Trainning, Testing & validation data
#### trainning part (60%), testing part(20%) and validation (20%)

```{r}
set.seed(123)
inBuild <- createDataPartition(y = trainData$classe, p = 0.8, list = F)
buildData <- trainData[inBuild,]
validation <- trainData[-inBuild,]
inTrain <- createDataPartition(y = buildData$classe, p = 0.75, list = F)
training <- buildData[inTrain,]
testing <- buildData[-inTrain,]
```
## Exploration Data
```{r}
qplot(classe, data = training, main = "Distribution of classes")
```

```{r}
highCorr <- findCorrelation(cor(training[,-53]), cutoff = 0.8)
names(training)[highCorr] #find correlation
```

```{r}
names(training[,53]) #final predictors
```

#MODEL BUILDING

```{r}
fitCtrl <- trainControl(method = "cv", number = 7, verboseIter = F,
                        preProcOptions = c("pca"),
                        allowParallel = T)
```

##MODEL SELECTION
### Evaluating multiple models

```{r}
preDf <- data.frame(run = 0, time = 0, gbm = 0, rf = 0, svmr =0,
                    svml = 0, nn = 0, lb = 0) #dataframe over multiple prediction
start.time.all = Sys.time() 
```

```{r}
for (i in 1:10){
  inTrain <- createDataPartition(y = buildData$classe, p = 0.75, list = F)
  training <- buildData[inTrain,]
  testing <- buildData[-inTrain,]
  dim(validation)
  dim(training)
  dim(testing)
  #building model
  start.time = Sys.time()
  mod.gbm <- train(classe ~ . , data = training , method = "gbm", trControl = fitCtrl, verbose = F)
  mod.rf <- train(classe ~ . , data = training , method = "rf", trControl = fitCtrl, verbose = F)
  mod.svmr <- train(classe ~ . , data = training , method = "svmRadial", trControl = fitCtrl, verbose = F)
  mod.svml <- train(classe ~ . , data = training , method = "svmLinear", trControl = fitCtrl, verbose = F)
  mod.nn <- train(classe ~ . , data = training , method = "nnet", trControl = fitCtrl, verbose = F)
  mod.lb <- train(classe ~ . , data = training , method = "LogitBoost", trControl = fitCtrl, verbose = F)
  stop.time = Sys.time()
}
```

```{r}
#Predictions
pred_val <- c( i, (stop.time - start.time), 
               unname(confusionMatrix(predict(mod.gbm, testing), testing$classe)$overall[1]),
               unname(confusionMatrix(predict(mod.rf, testing), testing$classe)$overall[1]),
               unname(confusionMatrix(predict(mod.svmr, testing), testing$classe)$overall[1]),
               unname(confusionMatrix(predict(mod.svml, testing), testing$classe)$overall[1]),
               unname(confusionMatrix(predict(mod.nn, testing), testing$classe)$overall[1]),
               unname(confusionMatrix(predict(mod.lb, testing), testing$classe)$overall[1]))

predDf <- rbind(predDf, pred_val) 
rownames(predDf)
kable(predDf[, -c(2), digits = 3])

```


##Selection final set of Models & out of sample accuracy

```{r}
validAccuracy <- data.frame(Accuracy = c(
  confusionMatrix(predict(mod.rf, validation), validation$classe)$overall[1],
  confusionMatrix(predict(mod.gbm, validation), validation$classe)$overall[1],
  confusionMatrix(predict(mod.svmr, validation), validation$classe)$overall[1]))
rowname(validAccuracy) <- c("rf", "gbm", "svmr")
kable(t(validAccuracy), digits = 3)

```

#Final Model

```{r}
finMod.rf <- train(classe ~ ., data = trainData, method = "rf",
                   trControl = fitCtrl, verbose = F) #Rf is best

```

#Model agreement accuracy

```{r}
# gbm for agreement accuracy
finMod.gbm <- train(classe ~ . , data = trainData, method = "gbm",
                    trControl = fitCtrl, verbose = F)
#svmr for agreement accuracy
finMod.svmr <- train(classe ~ . , data = trainData, method = "svmRadial")

#svmr for  agreement accuracy
finMod.svmr <- train(classe ~ . , data = trainData, method = "svmRadial", trControl = fitCtrl, verbose = F)
```


```{r}
#predict from 3 different best model
predFin.rf <- predict(finMod.rf, testData)
predFin.gbm <- predict(finMod.gbm, testData)
predFin.svmr <- predict(finMod.svmr, testData)

#check for agreement accuracy
modAgreementAccuracy <- data.frame(Agreement.Accuracy = c(
  confusionMatrix(predFin.gbm, predFin.rf)$overall[1],
  confusionMatrix(predFin.svmr, predFin.rf)$overall[1]))
rownames(modAgreementAccuracy) <- c("gbm vs. rf", "svmr vr.rf")
kable(t(modAgreementAccuracy), digits =3)
```
#FINAL PREDICTION

```{r}
#Final Prediction
finPred <- data.frame(prediction = predFin.rf)
row.names(finPred) <- 1:length(predFin.rf)
kable(t(finPred))
```