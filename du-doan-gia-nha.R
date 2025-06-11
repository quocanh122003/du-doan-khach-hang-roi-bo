# Install necessary libraries (if not already installed)
install.packages(c("dplyr", "ggplot2", "caret", "xgboost", "data.table"))

# Load libraries
library(dplyr)
library(ggplot2)
library(caret)
library(xgboost)
library(DMwR)
library(data.table)
#xem dữ liệu
customer_data <- read.csv("D:/Năm 4/R/train.csv")
str(customer_data)
summary(customer_data)

# Trực quan hóa dữ liệu
# check giá trị trống
colSums(is.na(customer_data))

# Tải thư viện ggplot2 nếu chưa có
library(ggplot2)

# Xác định các cột phân loại
object_cols <- names(customer_data)[sapply(customer_data, is.factor) | sapply(customer_data, is.character)]

# Vẽ đồ thị cho các cột phân loại
for (col in object_cols) {
  p <- ggplot(customer_data, aes_string(x = col)) +
    geom_bar() +
    ggtitle(col) +
    theme_minimal() +
    theme(axis.text.x = element_text(angle = 45, hjust = 1))
  
  print(p)
}


# Tải các thư viện cần thiết
install.packages("gridExtra")

library(ggplot2)
library(gridExtra)  # Để sắp xếp nhiều biểu đồ trong một lưới

# Xác định các cột số
numeric_cols <- names(customer_data)[sapply(customer_data, is.numeric)]

# In ra các cột số
print(numeric_cols)

# Vẽ biểu đồ histogram cho các cột số
for (col in numeric_cols) {
  p <- ggplot(customer_data, aes_string(x = col)) +
    geom_histogram(binwidth = 1, fill = "blue", color = "black", alpha = 0.7) +
    ggtitle(paste("Histogram of", col)) +
    theme_minimal()
  
  print(p)
}

# Vẽ biểu đồ boxplot để kiểm tra giá trị ngoại lai
for (col in numeric_cols) {
  p_box <- ggplot(customer_data, aes_string(y = col)) +
    geom_boxplot(fill = "orange", outlier.colour = "red") +
    ggtitle(paste("Boxplot of", col)) +
    theme_minimal()
  
  print(p_box)
}




#####
# Tải các thư viện cần thiết
library(dplyr)
library(caret)  # Để sử dụng dummyVars
library(corrplot)  # Để trực quan hóa ma trận tương quan

# Sao chép tập dữ liệu
dataset <- customer_data

# Chuyển đổi cột 'state' thành yếu tố (factor) và tạo các biến giả cho các cột phân loại
dataset$state <- as.factor(dataset$state)

# Sử dụng dummyVars để tạo biến giả cho các cột phân loại
dummies <- dummyVars(~ ., data = dataset)
dataset_hash_dummy <- predict(dummies, newdata = dataset)
dataset_hash_dummy <- as.data.frame(dataset_hash_dummy)

# Kiểm tra ma trận tương quan
corr <- cor(dataset_hash_dummy)

# Trực quan hóa ma trận tương quan
corrplot(corr, method = "color", addCoef.col = "black", tl.col = "black", tl.srt = 45)

# In ra ma trận tương quan
print(corr)

###
# Tải các thư viện cần thiết
install.packages("DMwR")
install.packages("smotefamily")
library(dplyr)
library(caret)  # Để sử dụng dummyVars và tạo partition
library(corrplot)  # Để trực quan hóa ma trận tương quan
library(DMwR)  # Để sử dụng SMOTE
library(scales)  # Để chuẩn hóa dữ liệu

# Giả sử rằng dataset_hash_dummy đã được tạo ra trước đó

# Đặt độ chính xác với format
corr <- cor(dataset_hash_dummy)
corr <- round(corr, 2)  # Làm tròn ma trận tương quan

# Trực quan hóa ma trận tương quan
corrplot(corr, method = "color", addCoef.col = "black", tl.col = "black", tl.srt = 45)

# Xóa những cột không hợp lý
dataset_hash_dummy_drop_corr <- dataset_hash_dummy %>%
  select(-c(voice_mail_plan, total_day_charge, total_eve_charge, total_night_charge, total_intl_charge))

#Xây dựng mô hình
X <- dataset_hash_dummy_drop_corr %>% select(-churn_yes)
y <- dataset_hash_dummy_drop_corr$churn_yes

#Chia train, test
set.seed(42)  # Để tái tạo kết quả
train_index <- createDataPartition(y, p = 0.8, list = FALSE)
X_train <- X[train_index, ]
X_test <- X[-train_index, ]
y_train <- y[train_index]
y_test <- y[-train_index]

# Upsampling = SMOTE
X_train_resample <- SMOTE(y_train ~ ., data = data.frame(X_train, y_train), k = 5)
y_train_resample <- X_train_resample$y_train
X_train_resample <- X_train_resample[, -ncol(X_train_resample)]  # Loại bỏ cột y_train

# Scale
scale_columns <- c('account_length', 'number_vmail_messages', 'total_day_minutes',
                   'total_day_calls', 'total_eve_minutes', 'total_eve_calls',
                   'total_night_minutes', 'total_night_calls', 'total_intl_minutes',
                   'total_intl_calls', 'number_customer_service_calls')

# Chuẩn hóa dữ liệu
X_train_resample[scale_columns] <- scale(X_train_resample[scale_columns])
X_test[scale_columns] <- scale(X_test[scale_columns], center = attr(X_train_resample[scale_columns], "scaled:center"),
                               scale = attr(X_train_resample[scale_columns], "scaled:scale"))

# Bây giờ, bạn có thể sử dụng X_train_resample và X_test để xây dựng mô hình.



# mô hình XGboost
# Install required packages (if not already installed)
if (!require(xgboost)) install.packages("xgboost")

# Load các thư viện
library(xgboost)
library(caret)  

set.seed(42)

# các tham số
model_params <- list(
  objective = "binary:logistic",  # Assuming binary classification
  nrounds = 200,                  # Number of trees
  random.seed = 42               # Set random seed
)

model_xgb <- xgboost(data = X_train, label = y_train, params = model_params)

# làm dự đoán với file test 
test <- read.csv("D:/Năm 4/R/test.csv")
id_submit <- test$id
test <- test[,-1]  # Remove "id" column
# chuẩn bị data
test_hash_state <-
test_dummy <- caret::model.matrix(formula = ~ ., data = test_hash_state, drop = TRUE)
test_dummy_drop_corr <- test_dummy[,-c(which(colnames(test_dummy) %in% c("voice_mail_plan_yes", "total_day_charge", "total_eve_charge", "total_night_charge", "total_intl_charge")))]

# dự đoán
y_pred_submit <- predict(model_xgb, newdata = test_dummy_drop_corr)

# gán với id chuyển từ dạng 0,1 sang dạng yes no
submit_result <- data.frame(id = id_submit, churn = ifelse(y_pred_submit == 0, "no", "yes"))

# in kết quả và in ra file csv
print(submit_result)
write.csv(data, file = "dubaoroibo.csv", row.names = FALSE)