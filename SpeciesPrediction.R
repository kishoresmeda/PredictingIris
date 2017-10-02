#import datasets package. We'll be using iris dataset present in this package
library(datasets)

#Load iris dataset
data(iris)

#Take a sneak peak into the dataset
summary(iris)

#Set a seed for generating random numbers. For achieving repeatability
set.seed(42)

#generate 100 numbers randomly between 1 and 150. Use this as training dataset
indexes <- sample(x=1:150,size=100)

#Take a look at numbers in the indexes
print(indexes)

#Create a training dataset from indexes
train <- iris[indexes,]

#Create a test dataset by excluding numbers from indexes
test <- iris[-indexes,]

#Load decision tree classifier into memory
install.packages('tree') #Use this line to download "tree" if not installed already
library(tree)

#Training the decision tree model
model <- tree(formula = Species ~ ., data = train)

#Inspect the model
summary(model)

#Visualize the decision tree model
plot(model)
text(model)

#Optional: Let's visualize the model using a scatter plot
library(RColorBrewer)

#Create a color palette
palette <- brewer.pal(3,"Set1")

#Create a scatter plot colored by species
plot(x=iris$Petal.Length,
     y=iris$Petal.Width,
     pch=19,
     col=palette[as.numeric(iris$Species)],
     main = "Iris Petal length versus Petal Width",
     xlab = "Petal length in cm",
     ylab = "Petal width in cm")

#Let's plot the decision boundaries with labels
partition.tree(model, label="Species",add=TRUE)

#Predict the species for test data set
predictions <- predict(object = model,
                       newdata = test,
                       type = "class")

#Create a confusion matrix to find prediction accuracy
table(x=predictions,y=test$Species)

#Use caret(classification and regression training) package for measuring prediction accuracy
install.packages("caret")
library(caret)

#Use confusionMatrix() in caret package
confusionMatrix(predictions,test$Species)

#Save the trained model as an R object to a file
save(model,file = "IrisTree.Rdata")
