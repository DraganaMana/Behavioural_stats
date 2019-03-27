cats <- data.frame(coat = c("calico", "black", "tabby"), 
                   weight = c(2.1, 5.0,3.2), 
                   likes_string = c(1, 0, 1))
write.csv(x = cats, file = "data/feline-data.csv", row.names = FALSE)

# Load the csv file in R
cats <- read.csv(file = "data/feline-data.csv")

# Read the 'weight' column
cats$weight

cats$coat

## Say we discovered that the scale weighs two Kg light:
cats$weight + 2 # this doesn't change the original column

paste("My cat is", cats$coat)

# gives the type
typeof(cats$weight)


# The L suffix forces the number to be an integer, since by default R uses float numbers
typeof(1L)

# creating a vector by default sets it to logical
my_vector <- vector(length = 3)

another_vector <- vector(mode='character', length=3)

# check if sth is a vector
str(another_vector)

# coerce data into logical
cats$likes_string <- as.logical(cats$likes_string)


# Factors
coats <- c('tabby', 'tortoiseshell', 'tortoiseshell', 'black', 'tabby')
str(coats)
CATegories <- factor(coats)
class(CATegories)

# how not to get factors but strings
cats <- read.csv(file="data/feline-data.csv", stringsAsFactors=FALSE)
str(cats$coat)

# There are several subtly different ways to call variables, observations and elements from data.frames:
  
cats[1]
cats[[1]]
cats$coat
cats["coat"]
cats[1, 1]
cats[, 1]
cats[1, ]

#
x <- matrix(1:50, ncol=5, nrow=10)
x <- matrix(1:50, ncol=5, nrow=10, byrow = TRUE) # to fill by row
