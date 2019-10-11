#Install and load packages for loading the data

if(!require("pacman")) install.packages("pacman")

#Load contributed packages with pacman
pacman::p_load(pacman,party,rio,tidyverse)
p_load(psych)


#Importing a CSV file with tidyverse

(df <- read_csv("Train.csv"))
(hf <- read_csv("promoted.csv"))
 
y <- object.size(hf)

?hist

#create a bar chart of all variables
hist(df$`awards_won?`,
     main ="Awards Won",
     col = "salmon1"
     )
  

# summarize Dataframe
summary(hf)

hf %>%
  select(department) %>%
  table()

df %>%
  select(department) %>%
  table()

describe(hf)
describe(df)
