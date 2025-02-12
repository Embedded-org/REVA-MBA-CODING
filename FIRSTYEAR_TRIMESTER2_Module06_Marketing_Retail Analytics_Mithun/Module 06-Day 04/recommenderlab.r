library("recommenderlab")
data("MovieLense")
str(MovieLense)
slotNames(MovieLense)
str(as(MovieLense, "data.frame"))
943*1664
image(MovieLense [1:25, 1:25])
head(as(MovieLense, "data.frame"))
hist(rowCounts(MovieLense))
hist(colCounts(MovieLense))
hist(rowMeans(MovieLense),col=2)
nusers=dim(MovieLense)[1]
nusers
nmovies=dim(MovieLense)[2]
nmovies
summary(rowCounts(MovieLense))
image(MovieLense[sample(nusers,25),sample(nmovies,25)])
vector_ratings <- as.vector(MovieLense@data)
unique(vector_ratings)
table_ratings <- table(vector_ratings)
table_ratings
barplot(table_ratings)
vector_ratings2 <- vector_ratings[vector_ratings != 0]
table_ratings2 <- table(vector_ratings2)
table_ratings2
barplot(table_ratings2)
recommenderRegistry$get_entries(dataType = "realRatingMatrix")
similarity_users <- similarity(MovieLense[1:4, ], method = "cosine", which = "users") 
as.matrix(similarity_users)
similarity_items <- similarity(MovieLense[, 1:4], method = "cosine", which = "items")
as.matrix(similarity_items)
recommenderRegistry$get_entries(dataType = "realRatingMatrix")
similarity_users <- similarity(MovieLense[1:4, ], method = "jaccard", which = "users") 
as.matrix(similarity_users)
similarity_items <- similarity(MovieLense[, 1:4], method = "jaccard", which = "items")
as.matrix(similarity_items)
#Create an evaluation scheme by splitting the data and specifying other parameters
evlS <- evaluationScheme(MovieLense, method="split", train=0.9,given=12); 
evlS
trg <- getData(evlS, "train"); trg
test_known <- getData(evlS, "known"); 
test_known
test_unknown <- getData(evlS, "unknown"); 
test_unknown
#Create UBCF recommender model with the training data
rcmnd_ub <- Recommender(trg, "UBCF")
#Create predictions for the test users
pred_ub <- predict(rcmnd_ub, test_known, type="ratings"); pred_ub
#Evaluate model accuracy for the unknown set of test users
acc_ub <- calcPredictionAccuracy(pred_ub, test_unknown)
as(acc_ub,"matrix")
#Compare the results
as(test_unknown, "matrix")[1:8,1:5]
as(pred_ub, "matrix")[1:8,1:5]
#Repeat the process with a IBCF model
rcmnd_ib <- Recommender(trg, "IBCF")
pred_ib <- predict(rcmnd_ib, test_known, type="ratings")
acc_ib <- calcPredictionAccuracy(pred_ib, test_unknown)
acc <- rbind(UBCF = acc_ub, IBCF = acc_ib); acc
#Get the top recommendations
pred_ub_top <- predict(rcmnd_ub, test_known); pred_ub
movies <-as(pred_ub_top, "list")
movies[1]
