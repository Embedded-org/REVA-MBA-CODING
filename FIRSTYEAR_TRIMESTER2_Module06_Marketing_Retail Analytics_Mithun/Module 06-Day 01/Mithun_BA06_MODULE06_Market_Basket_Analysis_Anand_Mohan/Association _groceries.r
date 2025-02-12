#install.packages("arulesViz",dependency=T)
#install.packages("colorspace",dependency=T)
#install.packages("tm",dependency=T)
library(arules)
library(arulesViz) 
library(colorspace)
library(datasets)
data(Groceries)
transactions <- Groceries
summary(transactions )
nrow(transactions)
itemFrequencyPlot(transactions, support=0.1, cex.names=0.8,col=2)
itemFrequencyPlot(transactions, support=0.05, cex.names=0.8,col=2)
itemFrequencyPlot(transactions, support=0.05, cex.names=0.8,col=2,horiz=TRUE)
itemFrequencyPlot(transactions, topN=20)
freq.itemsets <- eclat(transactions, parameter=list(supp=0.075, maxlen=15))
inspect(freq.itemsets)
# Apriori Algorithm
rules <- apriori(Groceries, parameter = list(support = 0.009, confidence = 0.25, minlen = 2))
summary(rules)
inspect(head(sort(rules, by ="lift"),5))
# Let us see rules that have high support and high confidence.
inspect(sort(sort(rules, by ="support"),by ="confidence")[1:5])
milk.rules <- sort(subset(rules, subset = rhs %in% "whole milk"), by = "confidence") 
summary(milk.rules)
inspect(milk.rules)
is.significant(milk.rules, transactions)
is.maximal(milk.rules)
is.redundant(milk.rules)
plot(milk.rules, measure=c("support", "confidence"), shading="lift")
coke.rules <- sort(subset(rules, subset = rhs %in% "soda"), by = "confidence")
summary(coke.rules)
inspect(coke.rules)
is.significant(coke.rules, transactions)
plot(coke.rules, measure=c("support", "confidence"), shading="lift")
meat.rules <- sort(subset(rules, subset = lhs %in% "beef"|lhs %in% "sausage" |lhs %in% "chicken"), by = "confidence") 
summary(meat.rules)
inspect(meat.rules)
is.significant(meat.rules, transactions)
yog.rules <- sort(subset(rules, subset = lhs %in% "yogurt"), by = "confidence") 
summary(yog.rules)
inspect(yog.rules)
is.significant(yog.rules, transactions)
plot(meat.rules,method="graph",shading="lift") 
plot(meat.rules,method="graph",engine='interactive' ,shading="lift")
plot(milk.rules,method="graph",shading="lift") 
plot(yog.rules,method="graph",shading="lift") 
plot(coke.rules,method="graph",shading="lift") 
trans.sel<-transactions[,itemFrequency(transactions)>0.1] # selected transactions
dissimilarity(trans.sel, which="items") 
plot(meat.rules, method="matrix", measure=c("support","confidence"), control=list(col=sequential_hcl(100)))
plot(meat.rules, method="grouped", measure="support", control=list(col=sequential_hcl(100)))
plot(meat.rules, method="paracoord", control=list(reorder=TRUE))
plot(yog.rules, method="matrix", measure=c("support","confidence"), control=list(col=sequential_hcl(100)))
plot(yog.rules, method="grouped", measure="support", control=list(col=sequential_hcl(100)))
plot(yog.rules, method="paracoord", control=list(reorder=TRUE))
plot(coke.rules, method="grouped", measure="support", control=list(col=sequential_hcl(100)))
plot(coke.rules, method="paracoord", control=list(reorder=TRUE))
#remove.packages("magrittr")
#install.packages("magrittr",dependency=T) # package installations are only needed the first time you use it
#install.packages("dplyr",dependency=T)    # alternative installation of the %>%
library(magrittr) # needs to be run every time you start R and want to use %>%
library(dplyr)    # alternatively, this also loads %>%
## Installing the package and calling the package in R## 
#install.packages("treemap",dependency=T) 
library(treemap)
occur1 <- transactions@itemInfo %>% group_by(level1) %>% summarize(n=n())
occur2 <- transactions@itemInfo %>% group_by(level1, level2) %>% summarize(n=n())
occur3 <- transactions@itemInfo %>% group_by(level1, level2, labels) %>% summarize(n=n())
treemap(occur1,index=c("level1"),vSize="n",title="",palette="Dark2",border.col="#FFFFFF")
treemap(occur2,index=c("level1", "level2"),vSize="n",title="",palette="Dark2",border.col="#FFFFFF")
treemap(occur3,index=c("level1", "labels"),vSize="n",title="",palette="Dark2",border.col="#FFFFFF")
inspect(tail(sort(rules, by = "lift")))