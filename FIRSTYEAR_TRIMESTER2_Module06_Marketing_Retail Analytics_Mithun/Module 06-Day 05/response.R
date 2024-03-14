# Data
getwd()
data <- read.csv("E:\\response.csv", header = TRUE)
abc.df <- read.csv("E:\\response.csv", header = TRUE)
View(data)
head(abc.df[,])
str(data)
data$Response=as.factor(data$Response)
str(data)
# Min-Max Normalization
data$income <- (data$income - min(data$income))/(max(data$income) - min(data$income))
data$debtinc <- (data$debtinc - min(data$debtinc))/(max(data$debtinc) - min(data$debtinc))
data$carloans <- (data$carloans - min(data$carloans))/(max(data$carloans)-min(data$carloans))
str(data)
#histogram
hist(data$income,col=2)
hist(data$debtinc,col=2)
hist(data$carloans ,col=2)
# Data Partition
set.seed(222)
ind <- sample(2, nrow(data), replace = TRUE, prob = c(0.7, 0.3))
training <- data[ind==1,]
testing <- data[ind==2,]
View(training) 
View(testing)
# Neural Networks
library(neuralnet)
set.seed(333)
n <- neuralnet(Response~income+debtinc+carloans,data = training,hidden = 1,err.fct = "ce",linear.output = FALSE)
plot(n)
plot(n,col.hidden = 'darkgreen',col.hidden.synapse = 'darkgreen',show.weights = T,information = T,fill = 'lightblue')
# Prediction
output <- compute(n, training[,-1])
head(output$net.result)
head(training[1,])
#########################optional part over here
# Node Output Calculations with Sigmoid Activation Function
in4 <- 0.0455 + (0.82344*0.7586206897) + (1.35186*0.8103448276) + (-0.87435*0.6666666667)
out4 <- 1/(1+exp(-in4))
in5 <- -7.06125 +(8.5741*out4)
out5 <- 1/(1+exp(-in5))
#########################optional part over here

# Confusion Matrix & Misclassification Error - training data
###################################################
output <- compute(n, training[,-1])
output
p1 <- output$net.result
pred1 <- ifelse(p1>=0.5, 1.0, 0.0)
tab1 <- table(pred1[1:281], training$Response)
length(pred1)
length(training$Response)
str(training)
1-sum(diag(tab1))/sum(tab1)
#######################################################################
# Confusion Matrix & Misclassification Error - testing data
output <- compute(n, testing[,-1])
output
p2 <- output$net.result
pred2 <- ifelse(p2>0.5, 1, 0)
tab2 <- table(pred2[1:119], testing$Response)
tab2
length(pred2)
length(testing$Response)
str(testing)
1-sum(diag(tab2))/sum(tab2)
#############################################################################
############model fine tuning to be done now
 #5 hidden layers
  n <- neuralnet(Response~income+debtinc+carloans,
                 data = training,
                 hidden = 5,
                 err.fct = "ce",
                 linear.output = FALSE)
plot(n)
# Confusion Matrix & Misclassification Error - training data
###################################################
output <- compute(n, training[,-1])
output
p1 <- output$net.result
pred1 <- ifelse(p1>=0.5, 1.0, 0.0)
tab1 <- table(pred1[1:281], training$Response)
length(pred1)
length(training$Response)
str(training)
1-sum(diag(tab1))/sum(tab1)
#######################################################################
# Confusion Matrix & Misclassification Error - testing data
output <- compute(n, testing[,-1])
output
p2 <- output$net.result
pred2 <- ifelse(p2>0.5, 1, 0)
tab2 <- table(pred2[1:119], testing$Response)
tab2
length(pred2)
length(testing$Response)
str(testing)
1-sum(diag(tab2))/sum(tab2)
#############################################################################
#Neural network with 2 hiden layers
n <- neuralnet(Response~income+debtinc+carloans,
               data = training,
               hidden = c(2,3),
               err.fct = "ce",
               linear.output = FALSE)
n <- neuralnet(Response~income+debtinc+carloans,
               data = training,
               hidden = c(2,1),
               err.fct = "ce",
               linear.output = FALSE)
plot(n)
# Confusion Matrix & Misclassification Error - training data
###################################################
output <- compute(n, training[,-1])
output
p1 <- output$net.result
pred1 <- ifelse(p1>=0.5, 1.0, 0.0)
tab1 <- table(pred1[1:281], training$Response)
tab1
length(pred1)
length(training$Response)
str(training)
1-sum(diag(tab1))/sum(tab1)
#######################################################################
# Confusion Matrix & Misclassification Error - testing data
output <- compute(n, testing[,-1])
output
p2 <- output$net.result
pred2 <- ifelse(p2>0.5, 1, 0)
tab2 <- table(pred2[1:119], testing$Response)
tab2
length(pred2)
length(testing$Response)
str(testing)
1-sum(diag(tab2))/sum(tab2)
#Neural network with repeat calculations--this gives error
####################################################################
n <- neuralnet(Response~income+debtinc+carloans,
               data = training,
               hidden = 5,
               err.fct = "ce",
               linear.output = FALSE,
               lifesign="full",
               rep=5)
plot(n,rep=4)
#########################################################################
################Neural network with repeat calculations---this does not give error
n <- neuralnet(Response~income+debtinc+carloans,
               data = training,
               hidden = 1,
               err.fct = "ce",
               linear.output = FALSE,
               lifesign="full",
               rep=5)
plot(n,rep=1)
plot(n,rep=2)
plot(n,rep=3)
plot(n,rep=4)
plot(n,rep=5)
plot(n)
# Confusion Matrix & Misclassification Error - training data
###################################################
output <- compute(n, training[,-1])
output
p1 <- output$net.result
pred1 <- ifelse(p1>=0.5, 1.0, 0.0)
tab1 <- table(pred1[1:281], training$Response)
tab1
length(pred1)
length(training$Response)
str(training)
1-sum(diag(tab1))/sum(tab1)
#######################################################################
# Confusion Matrix & Misclassification Error - testing data
output <- compute(n, testing[,-1])
output
p2 <- output$net.result
pred2 <- ifelse(p2>0.5, 1, 0)
tab2 <- table(pred2[1:119], testing$Response)
tab2
length(pred2)
length(testing$Response)
str(testing)
1-sum(diag(tab2))/sum(tab2)
#Change algorithm
rprop+: Resilient Backpropagation with weight backtracking
  
n <- neuralnet(Response~income+debtinc+carloans,
               data = training,
               hidden = 5,
               err.fct = "ce",
               linear.output = FALSE,
               algorithm="rprop+",
               stepmax=100000)
plot(n)
# Confusion Matrix & Misclassification Error - training data
###################################################
output <- compute(n, training[,-1])
output
p1 <- output$net.result
pred1 <- ifelse(p1>=0.5, 1.0, 0.0)
tab1 <- table(pred1[1:281], training$Response)
tab1
length(pred1)
length(training$Response)
str(training)
1-sum(diag(tab1))/sum(tab1)
#######################################################################
# Confusion Matrix & Misclassification Error - testing data
output <- compute(n, testing[,-1])
output
p2 <- output$net.result
pred2 <- ifelse(p2>0.5, 1, 0)
tab2 <- table(pred2[1:119], testing$Response)
tab2
length(pred2)
length(testing$Response)
str(testing)
1-sum(diag(tab2))/sum(tab2)
#########################################################################
n <- neuralnet(Response~income+debtinc+carloans,
               data = training,
               hidden = 5,
               err.fct = "ce",
               linear.output = FALSE,learningrate=0.4,
               algorithm="rprop+",
               stepmax=100000)
plot(n)


