# setup working directory
setwd("/Users/tyagraj/desktop/Project2")

# load the following libraries
library(data.table)
library(Matrix)
library(xgboost)
library(caret)
library(stringr)
library(tm)
library(syuzhet) 


# LabelCount Encoding function
labelCountEncoding <- function(column){
  return(match(column,levels(column)[order(summary(column,maxsum=nlevels(column)))]))
}

# Load Text files
train_text <- do.call(rbind,strsplit(readLines('training_text'),'||',fixed=T))
train_text <- as.data.table(train_text)
train_text <- train_text[-1,]
colnames(train_text) <- c("ID", "Text")
train_text$ID <- as.numeric(train_text$ID)

test_text <- do.call(rbind,strsplit(readLines('test_text'),'||',fixed=T))
test_text <- as.data.table(test_text)
test_text <- test_text[-1,]
colnames(test_text) <- c("ID", "Text")
test_text$ID <- as.numeric(test_text$ID)

# Load Variant Files
train <- fread("training_variants", sep=",", stringsAsFactors = T)
test  <- fread("test_variants", sep=",", stringsAsFactors = T)

# Merging Vairant and Text Files
train <- merge(train,train_text,by="ID")
test  <- merge(test,test_text,by="ID")

# Removing the files that are not required
rm(test_text,train_text);gc()

# Adding a dummy Class to test variable
test$Class <- -1

# Binding train and test 
data <- rbind(train,test)

# Removing the files that are not required
rm(train,test);gc()

# Basic text features
data$nchar <- as.numeric(nchar(data$Text))
data$nwords <- as.numeric(str_count(data$Text, "\\S+"))

# TF-IDF
txt <- Corpus(VectorSource(data$Text))
txt <- tm_map(txt, content_transformer(tolower))
txt <- tm_map(txt, removePunctuation)
txt <- tm_map(txt, removeWords, stopwords("english"))
txt <- tm_map(txt, stemDocument, language="english")
txt <- tm_map(txt, removeNumbers)
txt <- tm_map(txt, stripWhitespace)

dtm <- DocumentTermMatrix(txt, control = list(weighting = weightTfIdf))
dtm <- removeSparseTerms(dtm, 0.95)
data <- cbind(data, as.matrix(dtm))

# LabelCount Encoding for Gene and Variation
data$Gene <- labelCountEncoding(data$Gene)
data$Variation <- labelCountEncoding(data$Variation)

# Sentiment analysis
sentiment <- get_nrc_sentiment(data$Text) 
data <- cbind(data,sentiment) 

# Set seed
set.seed(1012)
cvFoldsList <- createFolds(data$Class[data$Class > -1], k=5, list=TRUE, returnTrain=FALSE)

# To sparse matrix
varnames <- setdiff(colnames(data), c("ID", "Class", "Text"))
train_sparse <- Matrix(as.matrix(sapply(data[Class > -1, varnames, with=FALSE],as.numeric)), sparse=TRUE)
test_sparse <- Matrix(as.matrix(sapply(data[Class == -1, varnames, with=FALSE],as.numeric)), sparse=TRUE)
y_train <- data[Class > -1,Class]-1
test_ids <- data[Class == -1,ID]

dtrain <- xgb.DMatrix(data=train_sparse, label=y_train)
dtest <- xgb.DMatrix(data=test_sparse)

# Params for xgboost
param <- list(booster = "gbtree",
              objective = "multi:softprob",
              eval_metric = "mlogloss",
              num_class = 9,
              eta = .2,
              gamma = 1,
              max_depth = 5,
              min_child_weight = 1,
              subsample = .7,
              colsample_bytree = .7
)

# Cross validation - for determining CV scores & optimal amount of rounds
xgb_cv <- xgb.cv(data = dtrain,
                 params = param,
                 nrounds = 1000,
                 maximize = FALSE,
                 prediction = TRUE,
                 folds = cvFoldsList,
                 print_every_n = 5,
                 early_stopping_round = 100)

rounds <- which.min(xgb_cv$evaluation_log[, test_mlogloss_mean])

# Train model
xgb_model <- xgb.train(data = dtrain,
                       params = param,
                       watchlist = list(train = dtrain),
                       nrounds = rounds,
                       verbose = 1,
                       print_every_n = 5
)

# Feature importance
names <- dimnames(train_sparse)[[2]]
importance_matrix <- xgb.importance(names,model=xgb_model)

xgb.plot.importance(importance_matrix[1:30,],20)

# Predict and output csv
preds <- as.data.table(t(matrix(predict(xgb_model, dtest), nrow=9, ncol=nrow(dtest))))

colnames(preds) <- c("class1","class2","class3","class4","class5","class6","class7","class8","class9")
write.table(data.table(ID=test_ids, preds), "submission.csv", sep=",", dec=".", quote=FALSE, row.names=FALSE)
