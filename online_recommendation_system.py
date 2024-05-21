import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

class RecommendationModel:
    def __init__(self):
        self.model = LinearRegression()
    
    def train_model(self,X,y):
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Training the model using the training data
        self.model.fit(X_train, y_train)

        # Predicting the target variable for the test data
        y_pred = self.model.predict(X_test)

        # Evaluating the model's performance
        mae = mean_absolute_error(y_test, y_pred)
        print("Mean Absolute Error:", mae)

    def get_recommended_data(self, user_input):
        predict_value = self.model.predict(user_input)
        print(f'predict_value : {predict_value}')
        return predict_value[0]
    
class CourseraCoursesData:
    def __init__(self):
        self.file_name = 'coursera_dataset.csv'
        self.read_data_from_file()
        self.do_preprocessing()

    def course_difficulty_modifier(self, x):
        if x == "Beginner":
            return "0"
        elif x == "Intermediate":
            return "1"
        elif x == "Mixed":
            return "2"
        elif x == "Advanced":
            return "3"
        else:
            return "0"
    
    def course_time_modifier(self, x):
        if x=="3 - 6 Months":
            return "3"
        elif x=="1 - 3 Months":
            return "2"
        elif x=="1 - 4 Weeks":
            return "1"
        elif x=="Less Than 2 Hours":
            return "0"
        else:
            return "0"
    
    def course_certificate_modifier(self, x):
        if x=="Specialization":
            return "3"
        elif x=="Professional Certificate":
            return "2"
        elif x=="Course":
            return "1"
        elif x=="Guided Project":
            return "0"
        else:
            return "0"
        
    def read_data_from_file(self):
        self.courses = pd.read_csv(self.file_name)
    
    def do_preprocessing(self):
        self.courses.drop(['course_reviews_num', 'course_url'], axis=1, inplace=True)
        self.courses.drop(['course_skills', 'course_summary', 'course_description'], axis=1, inplace=True)
        self.courses.dropna(subset = ['course_rating', 'course_students_enrolled'], inplace=True)
        self.courses['course_students_enrolled'] = self.courses.course_students_enrolled.str.replace(',', '').astype(float)
        self.courses.drop(self.courses[self.courses.course_students_enrolled > 1000000].index, inplace=True)

        self.courses['course_difficulty_modified']=self.courses['course_difficulty'].apply(self.course_difficulty_modifier)
        self.courses['course_difficulty_modified']=self.courses['course_difficulty_modified'].apply(pd.to_numeric)
        # self.courses = self.courses.drop(['course_difficulty'],axis=1)

        self.courses['course_certificate_type_modified']=self.courses['course_certificate_type'].apply(self.course_certificate_modifier)
        self.courses['course_certificate_type_modified']=self.courses['course_certificate_type_modified'].apply(pd.to_numeric)
        # self.courses =self.courses.drop(['course_certificate_type'],axis=1)

        self.courses['course_time_modified']= self.courses['course_time'].apply(self.course_time_modifier)
        self.courses['course_time_modified']= self.courses['course_time_modified'].apply(pd.to_numeric)
        # self.courses =self.courses.drop(['course_time'],axis=1)

        self.courses['overall_rating']=(self.courses['course_students_enrolled']/self.courses['course_students_enrolled'].max())*3+(self.courses['course_rating']/self.courses['course_rating'].max())*7

    def get_features_for_model(self):
        X = self.courses[['course_rating', 'course_students_enrolled', 'course_difficulty_modified','course_time_modified']]
        y = self.courses['overall_rating']
        return (X,y)
    
    def get_dataFrame(self):
        return self.courses
    
    def getMinimumRate(self):
        return self.courses['course_rating'].min()
    
    def getMaximumRate(self):
        return self.courses['course_rating'].max()
    
    def getMinimumStudent(self):
        return self.courses['course_students_enrolled'].min()
    
    def getMaximumStudent(self):
        return self.courses['course_students_enrolled'].max()

    def get_data_by_predict_value(self, rating):
        data = self.courses[(self.courses['overall_rating'] >= rating)]
        sorted_data = data.sort_values(by='overall_rating', ascending=True)
        print(f'number of rows : {sorted_data.shape[0]}')
        return sorted_data.head(4)
