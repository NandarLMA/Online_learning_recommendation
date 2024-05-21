import sys
from PyQt6.QtWidgets import (QMainWindow, QApplication, QWidget, QLabel, QSlider, QPushButton, 
                             QFormLayout, QVBoxLayout, QComboBox, QRadioButton, QButtonGroup, QDial)
from PyQt6.QtCore import Qt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from online_recommendation_system import RecommendationModel, CourseraCoursesData
from random import randint

class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.setGeometry(100, 100, 900, 1000)
        self.setWindowTitle("Online Course Recommendation System")
        
        recommendation = RecommendationSystemGUI()
        self.w = None  # No external window yet.
        self.button = QPushButton("Show Data Analysis")
        self.button.clicked.connect(self.show_new_window)

        layout = QVBoxLayout()
        layout.addWidget(recommendation)
        layout.addWidget(self.button)

        # Create a central widget to hold the layout
        central_widget = QWidget()
        central_widget.setLayout(layout)

        self.setCentralWidget(central_widget)


    def show_new_window(self, checked):
        if self.w is None:
            self.w = HistogramWindow()
        self.w.show()

class HistogramWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Data Analysis Window")
        self.course_data = CourseraCoursesData()

        # Create four subplots
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        bins = [4.2, 4.4, 4.6, 4.8, 5]
        rating_intervals = pd.cut(self.course_data.get_dataFrame().course_rating, bins=bins, include_lowest=True)

        # Add color to each figure
        rating = sns.countplot(x=rating_intervals, ax=axs[0, 0], palette='Blues')
        rating.set_xticklabels(rating.get_xticklabels(), fontsize=8)

        certificate = sns.countplot(x='course_certificate_type', data=self.course_data.get_dataFrame(), ax=axs[0, 1], palette='Greens')
        certificate.set_xticklabels(certificate.get_xticklabels(), fontsize=5)

        course = sns.countplot(x='course_difficulty', data=self.course_data.get_dataFrame(), ax=axs[1, 0], palette='Oranges')
        course.set_xticklabels(course.get_xticklabels(), fontsize=6)

        time = sns.countplot(x='course_time', data=self.course_data.get_dataFrame(), ax=axs[1, 1], palette='Purples')
        time.set_xticklabels(time.get_xticklabels(), fontsize=6)

        # Create a layout to organize the plot figures
        layout = QVBoxLayout()
        canvas = FigureCanvas(fig)
        layout.addWidget(canvas)
        self.setLayout(layout)

class RecommendationSystemGUI(QWidget):
    def __init__(self):
        super().__init__()
        self.model = RecommendationModel()
        self.course_data = CourseraCoursesData()
        self.init_ui()
        
    def updateStudentLabel(self):
        val = self.student_slider.value() * 1000
        self.student_label.setText(' Minimum Number of Enrolled Student : ' + str(val))
        print(val)

    def updateRatingLabel(self):
        val = self.rating_input.value()
        self.rating_label.setText(' Course Rating: ' + str(val))
        print(val)

    def init_ui(self):
        self.setWindowTitle('Online Course Recommendation System')
        self.setGeometry(100, 100, 900, 1000)
        
        #course_students_enrolled
        self.student_label = QLabel(self)
        self.student_slider = QSlider(Qt.Orientation.Horizontal)
        self.student_slider.setMinimum(self.course_data.getMinimumStudent()/10000)
        self.student_slider.setMaximum(self.course_data.getMaximumStudent()/10000)
        self.student_slider.setValue(self.course_data.getMinimumStudent()/10000)
        self.student_slider.setTickInterval(10000)
        self.student_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.student_slider.valueChanged.connect(
                self.updateStudentLabel)
        
        self.student_label.setText(
                ' Minimum Number of Enrolled Student: ' + str(self.student_slider.value()*1000))

         # Course Rating
        self.rating_label = QLabel(self)
        self.rating_input = QDial()
        self.rating_input.setMinimumSize(100, 100) 
        self.rating_input.setMinimum(self.course_data.getMinimumRate())
        self.rating_input.setMaximum(self.course_data.getMaximumRate())
        self.rating_input.setValue(self.course_data.getMinimumRate())
        self.rating_input.valueChanged.connect(
                self.updateRatingLabel)
        
        self.rating_label.setText(
                ' Course Rating: ' + str(self.rating_input.value()))

        # Course Difficulty 
        self.difficulty_label = QLabel('Choose the course difficulty level :')
        self.difficulty_input = QComboBox()
        self.difficulty_input.addItems(['Beginner', 'Intermediate', 'Mixed', 'Advanced'])

        # Duration of Course 
        self.duration_label = QLabel('Choose the course duration :')
        self.duration_3_6_months = QRadioButton('3-6 months')
        self.duration_1_3_months = QRadioButton('1-3 Months')
        self.duration_1_4_weeks = QRadioButton('1-4 Weeks')
        self.duration_less_than_2_hours = QRadioButton('Less than 2 hours')

        # Group for exclusive selection
        duration_group = QButtonGroup(self)
        duration_group.addButton(self.duration_3_6_months)
        duration_group.addButton(self.duration_1_3_months)
        duration_group.addButton(self.duration_1_4_weeks)
        duration_group.addButton(self.duration_less_than_2_hours)

        # Submit Button
        self.submit_button = QPushButton('Submit')
        self.submit_button.clicked.connect(self.submit_clicked)

        # Layout
        self.figure_canvases = []
        self.layout = QFormLayout()
        self.layout.addRow(self.student_slider)
        self.layout.addRow(self.student_label)
        self.layout.addRow(self.difficulty_label, self.difficulty_input)
        self.layout.addRow(self.duration_label)
        self.layout.addRow(self.duration_3_6_months)
        self.layout.addRow(self.duration_1_3_months)
        self.layout.addRow(self.duration_1_4_weeks)
        self.layout.addRow(self.duration_less_than_2_hours)
        self.layout.addRow(self.rating_label, self.rating_input)
        
        self.layout.addRow(self.submit_button)

        self.visual_label = QLabel("Your data visualization : ")
        self.visual_label.setVisible(False)

        self.predicted_data_display = QLabel(self)
        self.layout.addRow(self.predicted_data_display)
        self.layout.addRow(self.visual_label)
        self.setLayout(self.layout)
        self.show()

        # Variable to store the selected course duration
        self.selected_duration = None

        # Connect radio buttons to a function that updates the variable
        self.duration_3_6_months.clicked.connect(self.update_selected_duration)
        self.duration_1_3_months.clicked.connect(self.update_selected_duration)
        self.duration_1_4_weeks.clicked.connect(self.update_selected_duration)
        self.duration_less_than_2_hours.clicked.connect(self.update_selected_duration)

    def update_selected_duration(self):
        sender = self.sender()  # Get the sender of the signal
        if sender.isChecked():
            self.selected_duration = sender.text()

    def show_matplotlib_figure(self, figure, title):
        # Create FigureCanvas
        canvas = FigureCanvas(figure)
        canvas.setWindowTitle(title)

        # Add the canvas reference to the list
        self.figure_canvases.append(canvas)
        self.layout.addWidget(canvas)
        
    def close_figures(self):
        # Close previously shown figures
        for canvas in self.figure_canvases:
            canvas.close()

        self.figure_canvases.clear()
    
    def split_title(self, title, max_length=25):
    # Break the title into two lines if it exceeds the max length
        if len(title) > max_length:
            split_index1 = title.rfind(' ', 0, max_length)  # Find the last space within the max length
            if split_index1 != -1 :
                return title[:split_index1] + '\n' + title[split_index1+1:]
        return title

    def show_visualization(self):
        course_titles = self.predicted_data.course_title
        enrolled_students = self.predicted_data.course_students_enrolled
        overall_rating = self.predicted_data.overall_rating

        fig1, ax1 = plt.subplots(figsize=(7, 2.4))  # Adjust the width and height as needed
        fig2, ax2 = plt.subplots(figsize=(7, 2.4))  # Adjust the width and height as needed

        bar_colors = ['blue', 'salmon', 'green', 'gold', 'red']

        ax1.bar(course_titles, enrolled_students, width = 0.5, color = bar_colors)
        ax1.set_title('Enrolled students Vs Course title', fontsize=10)  # Set title font size
        ax1.set_xticklabels([self.split_title(title) for title in course_titles], ha='right', fontsize=5)  # Split titles and rotate x-axis labels
        ax1.set_ylabel('Enrolled students (persons)', fontsize=8)

        # Visualization for Overall rating Vs Course title
        ax2.plot(course_titles, overall_rating, marker='o', linestyle='-', markersize=8)
        ax2.set_title('Overall rating Vs Course title', fontsize=10)  # Set title font size
        ax2.set_xticklabels([self.split_title(title) for title in course_titles], ha='right', fontsize=5)  # Split titles and rotate x-axis labels
        ax2.set_ylabel('Overall rating', fontsize=8)

        # Adjust layout to prevent clipping of labels
        plt.tight_layout()

        self.show_matplotlib_figure(fig1, 'Enrolled students Vs Course title')
        self.show_matplotlib_figure(fig2, 'Overall rating Vs Course title')
        
    def submit_clicked(self):
        self.visual_label.setVisible(True)
        
        student_slider = self.student_slider.value()
        course_rating = self.rating_input.value()
        difficulty_level = self.difficulty_input.currentIndex()
        course_duration = int(self.course_data.course_certificate_modifier(self.selected_duration)) 

        print(f'No.of Enrolled Student: {student_slider}')
        print(f'Difficulty Level: {difficulty_level}')
        print(f'Course Rating: {course_rating}')
        print(f'Course Duration: {course_duration}')

        user_input = np.array([course_rating, student_slider, difficulty_level, course_duration]).reshape(1, -1)
        self.close_figures()
        (X,y) = self.course_data.get_features_for_model()
        self.model.train_model(X,y)
        predict_value = self.model.get_recommended_data(user_input)
        self.predicted_data = self.course_data.get_data_by_predict_value(predict_value)
        my_string = ' , '.join(self.predicted_data.course_title)
        self.predicted_data_display.setText(
                'Recommendation Courses are : ' + my_string )
        self.show_visualization()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())

