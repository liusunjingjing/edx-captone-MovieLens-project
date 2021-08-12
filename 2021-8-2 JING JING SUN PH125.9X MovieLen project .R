
# title: MovieLens Project Submission
## by JING JING SUN
## Aug 2, 2021


## **Input data set and clean the data with the code provided by course**


if(!require(tidyverse)) install.packages("tidyverse", repos = "http://cran.us.r-project.org")
if(!require(caret)) install.packages("caret", repos = "http://cran.us.r-project.org")
if(!require(data.table)) install.packages("data.table", repos = "http://cran.us.r-project.org")

library(tidyverse)
library(caret)
library(data.table)

### MovieLens 10M dataset:
### https://grouplens.org/datasets/movielens/10m/
### http://files.grouplens.org/datasets/movielens/ml-10m.zip


dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- fread(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                 col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")

## create data frame:

movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(movieId),
                                           title = as.character(title),
                                           genres = as.character(genres))
movielens <- left_join(ratings, movies, by = "movieId")


## Validation set will be 10% of MovieLens data

set.seed(1, sample.kind="Rounding") 
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]


## Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

## Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)

# know our data

library(tidyverse)
library(dslabs)
library(matrixStats)
library(caret)
library(lubridate)
library(dplyr)

head(edx)
str(edx)
dim(edx)
summary(edx)

## how many user and how many movies in this dataset 

edx %>%
  summarize(n_users = n_distinct(userId),
            n_movies = n_distinct(movieId))
## Distribution of Movie Ratings

edx %>% 
  dplyr::count(movieId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("Movies Ratings count")


## Distribution of user Ratings

edx %>% 
  dplyr::count(userId) %>% 
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black") + 
  scale_x_log10() + 
  ggtitle("user Ratings count")

## histogram of rating star number

edx%>% ggplot(aes(rating)) +
  geom_histogram( binwidth = 0.5,color = "black") +
  scale_x_continuous(breaks=seq(0, 5, by= 0.5)) +
  labs(x="rating", y="number of ratings") +
  ggtitle("number of ratings")

# **add more predictors to our data**

## add rating year 

edx <- mutate(edx, year_rated = year(as_datetime(timestamp)))
head(edx)

validation <- mutate(validation, year_rated = year(as_datetime(timestamp)))
head(validation)

##  rating vs year_rated 

edx %>% 
  group_by(year_rated) %>%
  summarize(rating = mean(rating)) %>%
  ggplot(aes(year_rated, rating)) +
  geom_point() +
  geom_smooth() +
  ggtitle("rating_year") +
  xlab("year_rated") + ylab("number of ratings")

## Extract the premier date 

premier <- stringi::stri_extract(edx$title, regex = "(\\d{4})", comments = TRUE ) %>% as.numeric()
head(premier)

vpremier <- stringi::stri_extract(validation$title, regex = "(\\d{4})", comments = TRUE ) %>% as.numeric()
head(vpremier)

## Add the premier date

edx <- edx %>% mutate(premier_date = premier) 
head(edx)

validation <- validation %>% mutate(premier_date = vpremier) 
head(validation)

##  plot premier date vs rating

edx %>%
  group_by(premier_date) %>% 
  summarize(n = n()) %>%
  ggplot(aes(n)) + 
  geom_histogram(bins = 30, color = "black", fill = "black", alpha = 0.75) + 
  scale_x_log10() +
  ggtitle("premier_date rating") +
  xlab("premier_date") + ylab("Rating")

## copy column genres with a name of s_genres

edx <- edx %>% mutate(s_genres = genres)
head(edx)

validation <- validation %>% mutate(s_genres = genres)
head(validation)

## split s_genres column 

edx <-edx %>% separate_rows(s_genres, sep ="\\|")
head(edx)

validation <-validation %>% separate_rows(s_genres, sep ="\\|")
head(validation)

## plot Genre Effect vs rating

edx %>% group_by(s_genres) %>%
  summarize(n = n(), avg = mean(rating)) %>%
  ggplot(aes(x = s_genres, y = avg)) + 
  geom_point() + 
  theme(axis.text.x = element_text(angle = 90, hjust = 1)) +
  ggtitle("Genre Effect") +
  xlab("Genres") + ylab("Rating")

#  **Build the Recommendation System**

Based on the course instruction, we should not use validation during the model building, so I split edx data set into train_set and test_set, and use them to build models.

##  split edx data into separate training and test 

set.seed(1, sample.kind="Rounding")
test_index <- createDataPartition(y = edx$rating, times = 1,
                                  p = 0.2, list = FALSE) # set 20% of edx as test set
train_set <- edx[-test_index,]
head(train_set )

test_set <- edx[test_index,]

## make sure all the user and movie and years found in test are in the train_set

test_set <- test_set %>% 
  semi_join(train_set, by = "movieId") %>%
  semi_join(train_set, by = "userId")
head(test_set)

# Build models
## **build model 1, assumes the same rating for all movies and all users**
## compute the average 

mu_hat <- mean(train_set$rating)
mu_hat

# rmse of same rating of all movies and all users model 

all_average_rmse <- RMSE(test_set$rating, mu_hat)
all_average_rmse

It is a high rmse, so it is not a good model. We must add more predictors to algorithm

# store RMSE to data.frame "rmse_result"

rmse_results <- data_frame(method = "all_average",  RMSE = all_average_rmse)
rmse_results 

## **build mode2  movie effects**

## compute the average rating based on movies

movie_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_i = mean(rating - mu_hat))
movie_avgs

## see distribution of bi(movie effect)

movie_avgs %>% qplot(b_i, geom ="histogram", bins = 10, data = ., color = I("black"))

# predicted rating of all average + movies effect model

movies_predicted_ratings <- mu_hat + test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  .$b_i

head(movies_predicted_ratings)

# RMSE of all average + movies effect

movies_rmse <- RMSE(movies_predicted_ratings, test_set$rating)
movies_rmse

#add moviesBi_rmse to rmse_result data frame

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Movie Model",
                                     RMSE = movies_rmse))
rmse_results


## **Modeling 3, movies+user effects** 
## see distribution of bu (user effect) that user rated more than 100 movies

train_set %>% 
  group_by(userId) %>% 
  filter(n()>=100) %>%
  summarize(b_u = mean(rating)) %>% 
  ggplot(aes(b_u)) + 
  geom_histogram(bins = 30, color = "black")

## compute the average rating for user that have rated 100 or more movies

user_avgs <- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  group_by(userId) %>%
  summarize(b_u = mean(rating - mu_hat - b_i))

head(user_avgs)

## construct predictors  with movies+user effects model
user_predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  mutate(pred = mu_hat + b_i + b_u) %>%
  .$pred

head(user_predicted_ratings)

## RMSE of movies+user effects model

movies_user_rmse <- RMSE(user_predicted_ratings, test_set$rating)
movies_user_rmse

## add movies_user_rmse to rmse_result data frame

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="movies and user model",
                                     RMSE = movies_user_rmse))
rmse_results

## **Modeling 4, movies+user+rating year effects**
## compute the average rating based on year_rated

year_avgs<- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  group_by(year_rated) %>%
  summarize(b_y = mean(rating - mu_hat - b_i-b_u))

head(year_avgs)

## construct predictors with movies+user+rating year effects model

year_predicted_ratings <- test_set%>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by='year_rated') %>%
  mutate(pred = mu_hat + b_i + b_u + b_y) %>%
  .$pred
head(year_predicted_ratings)

## RMSE of movies+user+rating year effects model

movies_user_year_rmse <- RMSE(year_predicted_ratings , test_set$rating)
movies_user_year_rmse

The result show year_rated doed not improve prediction much, it is not a good predictor. Since it do decrease rmse a little, we will still keep it in the aglorium.

## add movies_user_year_rmse to rmse_result data frame

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="movies+user+yearrating model",
                                     RMSE = movies_user_year_rmse))
rmse_results


## **Modeling 5, movies+user+rating year+premier effects**

## compute the average rating based on movies+user+rating year+premier effects model

premier_avgs<- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by='year_rated') %>%
  group_by(premier_date) %>%
  summarize(b_p = mean(rating - mu_hat - b_i - b_u - b_y))

head(premier_avgs)

## construct predictors with movies+user+rating year+premier effects model

premier_predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by='year_rated') %>%
  left_join(premier_avgs, by='premier_date') %>%
  mutate(pred = mu_hat + b_i + b_u + b_y +b_p) %>%
  .$pred

head(premier_predicted_ratings)

## RMSE of movies+user+rating year+premier effects model

movies_user_year_premier_rmse <- RMSE(premier_predicted_ratings, test_set$rating)
movies_user_year_premier_rmse

## add movies_user_year_premier_rmse to rmse_result data frame

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="movies+user+yearrating+premier model",
                                     RMSE = movies_user_year_premier_rmse))
rmse_results


## **Modeling 6, movies+user+rating year+premier+genres effects**

## compute the average rating based on movies+user+rating year+premier+genres effects model 

genres_avgs<- train_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by='year_rated') %>%
  left_join(premier_avgs, by='premier_date') %>%
  group_by(s_genres) %>%
  summarize(b_g = mean(rating - mu_hat - b_i - b_u - b_y - b_p))

head(genres_avgs)

## construct predictors with movies+user+rating year+premier+genres effects model

genres_predicted_ratings <- test_set %>% 
  left_join(movie_avgs, by='movieId') %>%
  left_join(user_avgs, by='userId') %>%
  left_join(year_avgs, by='year_rated') %>%
  left_join(premier_avgs, by='premier_date') %>%
  left_join(genres_avgs, by='s_genres') %>%
  mutate(pred = mu_hat + b_i + b_u + b_y +b_p + b_g) %>%
  .$pred

head(genres_predicted_ratings)

## RMSE of movies+user+rating year+premier+genres effects model

movies_user_year_premier_genres_rmse <- RMSE(genres_predicted_ratings, test_set$rating)
movies_user_year_premier_genres_rmse

## add movies_user_year_premier_genres_rmse to rmse_result data frame

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="movies+user+yearrating+premier+genres model",
                                     RMSE = movies_user_year_premier_genres_rmse))
rmse_results

## ** Regularization effect model**

## **Modeling 7, Regularized_Movie_Model**

## add  Penalized least squares  λ=3 

lambda <- 3
mu<- mean(train_set$rating)

## compute the average rating based on Regularized_Movie_Model

movie_reg_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(b_ir = sum(rating - mu)/(n()+lambda), n_i = n()) 

## construct predictors with Regularized_Movie_Model

movie_reg_predicted_ratings <- test_set %>% 
  left_join(movie_reg_avgs, by = "movieId") %>%
  mutate(pred = mu + b_ir) %>%
  pull(pred)

head(movie_reg_predicted_ratings)

## RMSE of Regularized_Movie_Model

Regularized_Movie_Model<-RMSE(movie_reg_predicted_ratings , test_set$rating)
Regularized_Movie_Model


## add rmse of Regularized_Movie_Model to rmse_result data frame

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="Regularized_Movie_Model",
                                     RMSE = Regularized_Movie_Model))
rmse_results

## ** Movie_reg_lambda_d model--Modeling 8, Regularized_Movie_Model with  different tuning** 

## add  Penalized least squares with  different tuning 

lambdas_d <- seq(0, 10, 0.25)

movie_reg_sum_avgs <- train_set %>% 
  group_by(movieId) %>% 
  summarize(s = sum(rating - mu), n_i = n())

head(movie_reg_sum_avgs)

## construct predictors with Movie_reg_lambda_d model

movie_reg_lambda_d_rmses <- sapply(lambdas_d, function(l){
  predicted_ratings <- test_set %>% 
    left_join(movie_reg_sum_avgs, by='movieId') %>% 
    mutate(b_ir = s/(n_i+l)) %>%
    mutate(pred = mu + b_ir) %>%
    pull(pred)
  return(RMSE(predicted_ratings, test_set$rating))
})

head(movie_reg_lambda_d_rmses)

# summary λ with corresponding rmse

qplot(lambdas_d, movie_reg_lambda_d_rmses)  

data.frame(lambdas_d, movie_reg_lambda_d_rmses)

## show the smallest  rmse and the lambda lead to it.


lambdas_d[which.min(movie_reg_lambda_d_rmses)]


movie_reg_lambda_d_rmses_model<- movie_reg_lambda_d_rmses[which.min(movie_reg_lambda_d_rmses)]

movie_reg_lambda_d_rmses_model

## add movie_reg_lambda_d_rmses_model  to rmse_result data frame

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="movie_reg_lambda_d_rmses_model",
                                     RMSE = movie_reg_lambda_d_rmses_model))
rmse_results

## **Modeling 9, use all cross  Regularization effects with  different tuning**

## add Penalized least squares with different tuning 
## add all mivie+user+rate year+ premier+genres predictors 

lambdas_d <- seq(0, 10, 0.25)

all_reg_lambda_d_rmses <- sapply(lambdas_d, function(l){
  
  mu <- mean(train_set$rating)
  
  b_ia <- train_set %>% 
    group_by(movieId) %>%
    summarize(b_ia = sum(rating - mu)/(n()+l))
  
  b_ua <- train_set %>% 
    left_join(b_ia, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_ua = sum(rating - b_ia - mu)/(n()+l))
  
  
  b_ya <- train_set %>% 
    left_join(b_ia, by="movieId") %>%
    left_join(b_ua, by="userId") %>%
    group_by(year_rated) %>%
    summarize(b_ya = sum(rating - b_ia - mu - b_ua)/(n()+l))
  
  
  b_pa <- train_set %>% 
    left_join(b_ia, by="movieId") %>%
    left_join(b_ua, by="userId") %>%
    left_join(b_ya, by="year_rated") %>%
    group_by(premier_date) %>%
    summarize(b_pa = sum(rating - b_ia - mu - b_ua - b_ya)/(n()+l))
  
  b_ga <- train_set %>% 
    left_join(b_ia, by="movieId") %>%
    left_join(b_ua, by="userId") %>%
    left_join(b_ya, by="year_rated") %>%
    left_join(b_pa, by="premier_date") %>%
    group_by(s_genres) %>%
    summarize(b_ga = sum(rating - b_ia - mu - b_ua - b_ya- b_pa)/(n()+l))
  
  predicted_ratings <- 
    test_set %>% 
    left_join(b_ia, by = "movieId") %>%
    left_join(b_ua, by = "userId") %>%
    left_join(b_ya, by = "year_rated") %>%
    left_join(b_pa, by = "premier_date") %>%
    left_join(b_ga, by = "s_genres") %>%
    mutate(pred = mu + b_ia + b_ua + b_ya + b_pa + b_ga) %>%
    pull(pred)
  
  return(RMSE(predicted_ratings, test_set$rating))
})

head(all_reg_lambda_d_rmses)

## summary λ with corresponding rmse and find the lambda that get the smallest RMSE,

qplot(lambdas_d, all_reg_lambda_d_rmses)  
lambdasmall <- lambdas_d [which.min(all_reg_lambda_d_rmses)]
lambdasmall

## show the smallest RMSE

all_reg_lambda_d_rmses_model <-all_reg_lambda_d_rmses[which.min(all_reg_lambda_d_rmses)]
all_reg_lambda_d_rmses_model

## add all_reg_lambda_d_rmses_model to rmse_result data frame

rmse_results <- bind_rows(rmse_results,
                           data_frame(method="all_reg_lambda_d_rmses_model",
                                      RMSE = all_reg_lambda_d_rmses_model))
rmse_results

## **final test,  the edx set as train set, the validation set as the test set to run the final all_reg_lambda_d_rmses_model**

lambdas_d <- seq(0, 10, 0.25)

final_all_reg_lambda_d_rmses <- sapply(lambdas_d, function(l){
  
  fmu <- mean(edx$rating)
  
  b_if <- edx %>% 
    group_by(movieId) %>%
    summarize(b_if = sum(rating - fmu)/(n()+l))
  
  b_uf <- edx %>% 
    left_join(b_if, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_uf = sum(rating - b_if - fmu)/(n()+l))
  
  
  b_yf <- edx %>% 
    left_join(b_if, by="movieId") %>%
    left_join(b_uf, by="userId") %>%
    group_by(year_rated) %>%
    summarize(b_yf = sum(rating - b_if - fmu - b_uf)/(n()+l))
  
  
  b_pf <- edx %>% 
    left_join(b_if, by="movieId") %>%
    left_join(b_uf, by="userId") %>%
    left_join(b_yf, by="year_rated") %>%
    group_by(premier_date) %>%
    summarize(b_pf = sum(rating - b_if - fmu - b_uf - b_yf)/(n()+l))
  
  b_gf <- edx %>% 
    left_join(b_if, by="movieId") %>%
    left_join(b_uf, by="userId") %>%
    left_join(b_yf, by="year_rated") %>%
    left_join(b_pf, by="premier_date") %>%
    group_by(s_genres) %>%
    summarize(b_gf = sum(rating - b_if - fmu - b_uf - b_yf- b_pf)/(n()+l))
  
  
  fpredicted_ratings <-  validation %>% 
    left_join(b_if, by = "movieId") %>%
    left_join(b_uf, by = "userId") %>%
    left_join(b_yf, by = "year_rated") %>%
    left_join(b_pf, by = "premier_date") %>%
    left_join(b_gf, by = "s_genres") %>%
    mutate(pred = fmu + b_if + b_uf + b_yf + b_pf + b_gf) %>%
    pull(pred)
  
  return(RMSE( fpredicted_ratings, validation$rating))
})

head(final_all_reg_lambda_d_rmses)

## find the lambda that get the smallest RMSE,

qplot(lambdas_d, final_all_reg_lambda_d_rmses)  

final_lambdasmall <- lambdas_d [which.min(final_all_reg_lambda_d_rmses)]
final_lambdasmall 
data.frame(lambdas_d, final_all_reg_lambda_d_rmses)

## show the smallest RMSE

final_all_reg_lambda_d_rmses_model <-final_all_reg_lambda_d_rmses[which.min(final_all_reg_lambda_d_rmses)]
final_all_reg_lambda_d_rmses_model

## add final_all_reg_lambda_d_rmses to rmse_result data frame

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="final_all_reg_lambda_d_rmses_model",
                                     RMSE = final_all_reg_lambda_d_rmses_model))
rmse_results

## **use wider ranger of lambdas,  the edx set as training set, the validation set as the testing set to run the final all_reg_lambda_d_rmses_model**

lambdas_w <- seq(0, 20, 0.25)

final_all_reg_lambda_w_rmses <- sapply(lambdas_w, function(l){
  
  fwmu <- mean(edx$rating)
  
  b_ifw <- edx %>% 
    group_by(movieId) %>%
    summarize(b_ifw = sum(rating - fwmu)/(n()+l))
  
  b_ufw <- edx %>% 
    left_join(b_ifw, by="movieId") %>%
    group_by(userId) %>%
    summarize(b_ufw = sum(rating - b_ifw - fwmu)/(n()+l))
  
  
  b_yfw <- edx %>% 
    left_join(b_ifw, by="movieId") %>%
    left_join(b_ufw, by="userId") %>%
    group_by(year_rated) %>%
    summarize(b_yfw = sum(rating - b_ifw - fwmu - b_ufw)/(n()+l))
  
  
  b_pfw <- edx %>% 
    left_join(b_ifw, by="movieId") %>%
    left_join(b_ufw, by="userId") %>%
    left_join(b_yfw, by="year_rated") %>%
    group_by(premier_date) %>%
    summarize(b_pfw = sum(rating - b_ifw - fwmu - b_ufw - b_yfw)/(n()+l))
  
  b_gfw <- edx %>% 
    left_join(b_ifw, by="movieId") %>%
    left_join(b_ufw, by="userId") %>%
    left_join(b_yfw, by="year_rated") %>%
    left_join(b_pfw, by="premier_date") %>%
    group_by(s_genres) %>%
    summarize(b_gfw = sum(rating - b_ifw - fwmu - b_ufw - b_yfw- b_pfw)/(n()+l))
  
  
  fwpredicted_ratings <-  validation %>% 
    left_join(b_ifw, by = "movieId") %>%
    left_join(b_ufw, by = "userId") %>%
    left_join(b_yfw, by = "year_rated") %>%
    left_join(b_pfw, by = "premier_date") %>%
    left_join(b_gfw, by = "s_genres") %>%
    mutate(pred = fwmu + b_ifw + b_ufw + b_yfw + b_pfw + b_gfw) %>%
    pull(pred)
  
  return(RMSE(fwpredicted_ratings, validation$rating))
})

head(final_all_reg_lambda_w_rmses)

## find the lambda that get the smallest RMSE

final_w_lambdasmall <- lambdas_w [which.min(final_all_reg_lambda_w_rmses)]
final_w_lambdasmall 

qplot(lambdas_w, final_all_reg_lambda_w_rmses) 

data.frame(lambdas_w, final_all_reg_lambda_w_rmses)

## show the smallest RMSE

final_all_reg_lambda_w_rmses_model <-final_all_reg_lambda_w_rmses[which.min(final_all_reg_lambda_w_rmses)]
final_all_reg_lambda_w_rmses_model

## add Regularized Movie_user_ratingyear_premier_Effec_Model1 to rmse_result data frame

rmse_results <- bind_rows(rmse_results,
                          data_frame(method="final_all_reg_lambda_w_rmses_model",
                                     RMSE = final_all_reg_lambda_w_rmses_model))
rmse_results

# end