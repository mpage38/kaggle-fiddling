setwd("/github/kaggle-fiddling/titanic/R")

library(ggplot2)

```

Let us first define a generic function for reading CSV files which handles data types and missing values

```{r, message=F, warning=F}

# file.path: path of CSV file to be read
# column.types:  type of each column
# missing.types: string corresponding to missing data
readCsvData <- function(file.path, column.types, missing.types) {
  read.csv(file.path, colClasses=column.types, na.strings=missing.types)
}

missing.types <- c("NA", "")
train.column.types <- c('integer',   # PassengerId
                        'factor',    # Survived 
                        'factor',    # Pclass
                        'character', # Name
                        'factor',    # Sex
                        'numeric',   # Age
                        'integer',   # SibSp
                        'integer',   # Parch
                        'character', # Ticket
                        'numeric',   # Fare
                        'character', # Cabin
                        'factor'     # Embarked
)
train <- readCsvData("../data/train.csv", train.column.types, missing.types)


barplot(prop.table(train$Survived),
        names.arg = c("Perished", "Survived"),
        main="Survived (passenger fate)", col="black")