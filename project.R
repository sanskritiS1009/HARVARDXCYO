###########################################
########## 1. Data preparation ############
###########################################

#The following commands will install the necessary libraries, if needed, and load them.

if(!require(dplyr)) install.packages("dplyr", repos = "http://cran.us.r-project.org")
if(!require(tm)) install.packages("tm", repos = "http://cran.us.r-project.org")
if(!require(kernlab)) install.packages("kernlab", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(splitstackshape)) install.packages("splitstackshape", repos = "http://cran.us.r-project.org")
if(!require(e1071)) install.packages("e1071", repos = "http://cran.us.r-project.org")

### 1.1 Ingest your training data and clean it. ###

# train data stored in this file  
path_train <- "C:\\Users\\sansr\\Desktop\\harvardX\\CYO\\Training"

# The VCorpus function from the tm package will create a volatile corpus from the train data.
train <- VCorpus(DirSource(path_train, encoding = "UTF-8"), readerControl=list(language="English"))

# Explore the object 'train':
inspect(train[1:10])
class(train)

# Let's clean up the data by collapsing extra whitespace to a single blank:
train <- tm_map(train, content_transformer(stripWhitespace))

# Convert all text to lower case
train <- tm_map(train, content_transformer(tolower))

# Remove numbers and punctuation
train <- tm_map(train, content_transformer(removeNumbers))
train <- tm_map(train, content_transformer(removePunctuation))

# This will remove English stopwords, i.e. English common words like 'the'
train <- tm_map(train, removeWords, stopwords("english"))

### 1.2 Create your document term matrices for the training data. ###

# First, let's create a document-term matrix, i.e. a mathematical matrix that describes the frequency of terms that occur in a collection of documents. 
# In a document-term matrix, rows correspond to words in the collection and columns correspond to documents.

train_dtm <- as.matrix(DocumentTermMatrix(train, control=list(wordLengths=c(1,Inf))))

# Let's have a look at the structure of the matrix:
str(train_dtm)


### 1.3 Repeat steps 1 & 2 above for the Test set. ###

# folder in which test data has been saved
path_test <- "C:\\Users\\sansr\\Desktop\\harvardX\\CYO\\Test"

# Clean test data
test <- VCorpus(DirSource(path_test, encoding = "UTF-8"), readerControl=list(language="English"))
test <- tm_map(test, content_transformer(stripWhitespace))
test <- tm_map(test, content_transformer(tolower))
test <- tm_map(test, content_transformer(removeNumbers))
test <- tm_map(test, content_transformer(removePunctuation))
test <- tm_map(train, removeWords, stopwords("english"))

# Create a document-term matrix  for the test data:
test_dtm <- as.matrix(DocumentTermMatrix(test, control=list(wordLengths=c(1,Inf))))


### 1.4 We need to make sure that the  test and train matrices are of identical length. To do so, we will find the intersection. ###
train_df <- data.frame(train_dtm[,intersect(colnames(train_dtm), colnames(test_dtm))])
test_df <- data.frame(test_dtm[,intersect(colnames(test_dtm), colnames(train_dtm))])

# Here is the result:

str(train_df, list.len = 10)
str(test_df, list.len = 10)

### 1.5 We need now to retrieve the correct labels for training data and put dummy values for testing data. ###
label_df <- data.frame(row.names(train_df))
colnames(label_df) <- c("filenames")
label_df<- cSplit(label_df, 'filenames', sep="_", type.convert=FALSE)
train_df$corpus<- label_df$filenames_1
test_df$corpus <- c("Neg")


###########################################
########## 2. Model building###############
###########################################

##We are now ready to create our classifier. We will create folds of our data, train our model and run it once to inspect results. ###

# We will start by using the training dataframes for both training and testing our model:
df_train <- train_df
df_test <- train_df
df_test$corpus <- as.factor(df_test$corpus)

# We will use this kernel for training our algorithm. 
df_model<-ksvm(corpus~., data= df_train, kernel="rbfdot")
# Predict:
df_pred<-predict(df_model, df_test)

# This is the confusion matrix of the result:
con_matrix<-confusionMatrix(df_pred, df_test$corpus)
print(con_matrix)


### Now we can Run the final prediction on the test data and re-attach the file names. ### 
df_test <- test_df
df_pred<-predict(df_model, df_test)
results <- as.data.frame(df_pred)
rownames(results) <- rownames(test_df)
head(results,30) %>% knitr::kable()