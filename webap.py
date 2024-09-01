import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

import streamlit as st
import numpy as np
import pandas as pd
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
from PIL import Image

# ________________________________________________________(SideBar [ET,OT & IT Data])

with st.sidebar:
    selected=option_menu(
        menu_title="Main Menu",
        options=["ET_Data","OT_Data","IT_Data"]
    )

# ---------------------------------------------------------(ET DATA)

if selected == "ET_Data":
   import streamlit as st
   st.image('ETData.jpg')

# ----------------------------------------------------------(OT DATA)

if selected == "OT_Data":
     #power
     import streamlit as st
     import numpy as np
     import pandas as pd
     
     st.set_option('deprecation.showPyplotGlobalUse', False)

# -----------------------------------------------------------([POWER])
     
     st.markdown("<h1 style='text-align: center;'>OPERATIONAL TECHNOLOGY DATA</h1>", unsafe_allow_html=True)
     st.header("Graph of Power of the Machine")

     data=pd.read_csv('accelerometers.csv')
     x=data[["speed(RPM)"]]
     y1=data[["power (HP)"]]

     from sklearn.model_selection import train_test_split
     xtrain , xtest ,ytrain , ytest = train_test_split(x,y1,test_size=0.2,random_state=0)

     from sklearn.linear_model import LinearRegression
     lr = LinearRegression()
     if ytrain.ndim==1 :
         ytrain=ytrain.shape(-1,1)
     if xtrain.ndim==1 :
         xtrain=xtrain.shape(-1,1)
     
     lr.fit(xtrain,ytrain)

     c=lr.intercept_
     
     m=lr.coef_
     
     ytrain_predict=lr.predict(xtrain)
     # ytrain_predict
     from sklearn.metrics import r2_score
     r2_score(ytrain,ytrain_predict)

     import matplotlib.pyplot as plt

     plt.scatter(xtrain,ytrain,color='blue', label = 'Actual')
     plt.plot(xtrain,ytrain_predict,color='red', label = 'Predicted')
     plt.xlabel("Speed (RPM)")
     plt.ylabel("Power (HP)")
     plt.title("Linear Regression - Actual vs. Predicted")
     plt.legend()
     plt.show()
     st.pyplot(plt.show())

# ------------------------------------------------------------([TORQUE])

     #torque
     st.header("Graph of Torque of Machine")
     import streamlit as st
     import numpy as np
     import pandas as pd
     from sklearn.model_selection import train_test_split
     from sklearn.linear_model import LinearRegression, Ridge
     from sklearn.preprocessing import PolynomialFeatures
     from sklearn.metrics import r2_score, mean_squared_error
     import matplotlib.pyplot as plt

     st.set_option('deprecation.showPyplotGlobalUse', False)

# Load and explore the data
     data = pd.read_csv('accelerometers.csv')
     x = data[["speed(RPM)"]]
     y = data[["torque (lb.in)"]]

# Split data into training and testing sets
     x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)

# Experiment with different polynomial degrees and regularization
     degrees = [2, 3, 4]  # Adjust as needed
     regularization_strengths = [0.01, 0.1, 1.0]

     best_score = float('-inf')  # Initialize to negative infinity
     best_model = None

     for degree in degrees:
      for reg_strength in regularization_strengths:
        # Create polynomial features (avoid multicollinearity)
         poly = PolynomialFeatures(degree=degree, include_bias=False)
         x_train_poly = poly.fit_transform(x_train)

        # Create and fit a regression model (e.g., Ridge regression)
         model = Ridge(alpha=reg_strength)
         model.fit(x_train_poly, y_train)

        # Calculate R^2 score (or RMSE/MAE if more relevant)
         r2_score_train = r2_score(y_train, model.predict(x_train_poly))

        # Track the best model based on R^2 score
         if r2_score_train > best_score:
            best_score = r2_score_train
            best_model = model

# Evaluate the best model on the test set
            y_test_predict = best_model.predict(poly.fit_transform(x_test))
            r2_score_test = r2_score(y_test, y_test_predict)

            print("Best R^2 score on training set:", best_score)
            print("R^2 score on test set:", r2_score_test)

     plt.scatter(x_test, y_test, color='blue', label='Actual Torque')
     plt.scatter(x_test, y_test_predict, color='lightgreen', label='Predicted Torque')
     plt.xlabel("Speed (RPM)")
     plt.ylabel("Torque (lb.in)")
     plt.title("Polynomial Regression - Actual vs. Predicted")
     plt.legend()
     plt.show()
     st.pyplot(plt.show())

# -----------------------------------------------------------([STITCH])

     #stitch
     st.header("Graph of Stitches of the Machine")
     st.set_option('deprecation.showPyplotGlobalUse', False) 
     data=pd.read_csv('accelerometers.csv')
     x=data[["speed(RPM)"]]
     y1=data[["stiches"]]

     from sklearn.model_selection import train_test_split
     xtrain , xtest ,ytrain , ytest = train_test_split(x,y1,test_size=0.2,random_state=0)

     from sklearn.linear_model import LinearRegression
     
     lr = LinearRegression()

     if ytrain.ndim==1 :
         ytrain=ytrain.shape(-1,1)
     if xtrain.ndim==1 :
         xtrain=xtrain.shape(-1,1)
     
     lr.fit(xtrain,ytrain)

     c=lr.intercept_
     
     m=lr.coef_
     
     ytrain_predict=lr.predict(xtrain)

     from sklearn.metrics import r2_score
     r2_score(ytrain,ytrain_predict)

     import matplotlib.pyplot as plt
     
     plt.scatter(xtrain,ytrain,color='blue',label = 'Actual')
     plt.plot(xtrain,ytrain_predict,color='red',label = 'Predicted')
     plt.xlabel("Speed [RPM]")
     plt.ylabel("Stitch")
     plt.title("Linear Regression - Actual vs. Predicted")
     plt.legend()
     plt.show()
     st.pyplot(plt.show())

# ------------------------------------------------------([VIBRATION])

     #xyz
     st.header("Graph of Vibration of Machine")
     from sklearn.model_selection import train_test_split
     from sklearn.linear_model import Ridge
     from sklearn.preprocessing import PolynomialFeatures
     from sklearn.metrics import r2_score, mean_squared_error
     import matplotlib.pyplot as plt
     from sklearn.pipeline import Pipeline
     from sklearn.model_selection import GridSearchCV

     st.set_option('deprecation.showPyplotGlobalUse', False)

# Load the dataset
     try:
         data = pd.read_csv('accelerometers.csv')
     except FileNotFoundError:
         print("Error: 'accelerometers.csv' file not found. Please check the file path.")
         exit()

# Select features and target columns
     features = data[["speed(RPM)"]]
     target_columns = ["x", "y", "z"]

# Create empty lists to store R^2 and MSE scores for each target column
     r2_scores_poly = {col: [] for col in target_columns}
     mse_scores_poly = {col: [] for col in target_columns}

# Iterate through target columns 
     for col in target_columns: 
         y = data[[col]]

    # Split data into training and testing sets
         x_train, x_test, y_train, y_test = train_test_split(features, y, test_size=0.2, random_state=0)

    # Experiment with different polynomial degrees and regularization (optional)
         degrees = [2, 3, 4]  # Adjust as needed
         regularization_strengths = [0.01, 0.1, 1.0]

         for degree in degrees:
             for reg_strength in regularization_strengths:
            # Create polynomial feature transformer
                 poly = PolynomialFeatures(degree=degree, include_bias=False)  # Avoid multicollinearity
                 x_train_poly = poly.fit_transform(x_train)

            # Create a Ridge regression model
                 ridge_model = Ridge(alpha=reg_strength)

            # Create a pipeline combining polynomial features and Ridge model
                 pipe = Pipeline([
                     ('poly', poly),
                     ('ridge', ridge_model)
                 ])

            # Fit the pipeline on training data
                 pipe.fit(x_train, y_train)

            # Predict on training data
                 y_train_predict_poly = pipe.predict(x_train)

            # Calculate R^2 score
                 r2_score_poly = r2_score(y_train, y_train_predict_poly)
                 r2_scores_poly[col].append(r2_score_poly)

            # Calculate mean squared error (MSE)
                 mse_score_poly = mean_squared_error(y_train, y_train_predict_poly)
                 mse_scores_poly[col].append(mse_score_poly)

# Plotting the bar graph
     for col in target_columns:
      plt.bar([f"{col} (R^2)", f"{col} (MSE)"], [np.mean(r2_scores_poly[col]), np.mean(mse_scores_poly[col])], alpha=0.7, label=col)

      plt.xlabel('Metrics')
      plt.ylabel('Scores')
      plt.title('Average R^2 and MSE Scores for Different Target Columns')
      plt.legend()
      plt.show()
      st.pyplot(plt.show())
      
# ______________________________________________________________________(PREDICTION)

st.set_option('deprecation.showPyplotGlobalUse', False)

# Load data
data = pd.read_csv('accelerometers.csv')
x = data[["speed(RPM)"]]
z = data[["power (HP)"]]
y = data[["torque (lb.in)"]]
option = st.sidebar.selectbox('Select an option', ("Power", "Torque"))

# ----------------------------------------------------------------------([POWER])

if selected == "IT_Data":
    if option == "Power":

       from sklearn.model_selection import train_test_split
       xtrain , xtest ,ytrain , ytest = train_test_split(x,z,test_size=0.2,random_state=0)

       from sklearn.linear_model import LinearRegression
       lr = LinearRegression()
       if ytrain.ndim==1 :
           ytrain=ytrain.shape(-1,1)
       if xtrain.ndim==1 :
           xtrain=xtrain.shape(-1,1)
       lr.fit(xtrain,ytrain)

       c=lr.intercept_
       
       m=lr.coef_
       
       ytrain_predict=lr.predict(xtrain)

       from sklearn.metrics import r2_score
       r2_score(ytrain,ytrain_predict)

    # Define coefficients
       c = 10
       m = 0.05

       def predict_power(speed):
           """Predicts power for a given speed."""
           try:
               speed = float(speed)
               predicted_power = c + m * speed
               return predicted_power
           except ValueError:
               return None

       # Streamlit app header
       st.title("Power Prediction App")
    
    # User input for speed
       speed = st.number_input("Enter speed in RPM:", min_value=0, max_value=5000)
    
    # Calculate and display predicted power
       predicted_power = predict_power(speed)
       if predicted_power is not None:
           st.write(f"Predicted power for speed {speed} RPM: {predicted_power:.2f} HP") 
            # Plot graph
           plt.scatter(xtrain,ytrain, color='blue', label='Actual Speed')
           plt.plot(xtrain,ytrain_predict, color='red', label='Predicted Power')
           plt.scatter(speed, predicted_power, color='green', label='User Input Prediction')
           plt.xlabel("Speed (RPM)")
           plt.ylabel("Power (HP)")
           plt.title("Actual Speed vs. Predicted Power")
           plt.legend()
           st.pyplot()
       else:
           st.error("Invalid speed entered. Please enter a numerical value.")

# ------------------------------------------------------------------------([TORQUE])

    if option == "Torque":
        st.title("Torque Prediction App")
    # Split data into training and testing sets
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.1, random_state=0)
    
    # Experiment with different polynomial degrees and regularization
        degrees = [2, 3, 4]  # Adjust as needed
        regularization_strengths = [0.01, 0.1, 1.0]

        best_score = float('-inf')  # Initialize to negative infinity
        best_model = None

        for degree in degrees:
            for reg_strength in regularization_strengths:
            # Create polynomial features (avoid multicollinearity)
                poly = PolynomialFeatures(degree=degree, include_bias=False)
                x_train_poly = poly.fit_transform(x_train)

            # Create and fit a regression model (e.g., Ridge regression)
                model = Ridge(alpha=reg_strength)
                model.fit(x_train_poly, y_train)

            # Calculate R^2 score
                r2_score_train = r2_score(y_train, model.predict(x_train_poly))

            # Track the best model based on R^2 score
                if r2_score_train > best_score:
                    best_score = r2_score_train
                    best_model = model

    # Evaluate the best model on the test set
        y_test_predict = best_model.predict(poly.transform(x_test))
        r2_score_test = r2_score(y_test, y_test_predict)

    # Streamlit app
        input_number = 0
        while True:
            try:
            # Get user input for speed
                input_number += 1
                speed = st.number_input(f"Enter speed {input_number} in RPM:", key=f"speed_{input_number}", min_value=0, max_value=5000)

            # Predict torque
                speed_poly = poly.transform(np.array([[speed]]))
                predicted_torque = best_model.predict(speed_poly)[0][0]

            # Print prediction and show graph
                st.write(f"Predicted torque for speed {speed} RPM: {predicted_torque}")

            # Visualization
                plt.scatter(x_test, y_test, color='blue', label='Actual Torque')
                plt.scatter(x_test, y_test_predict, color='red', label='Predicted Torque')
                plt.scatter(speed, predicted_torque, color='green', marker='o', label='User Input')
                plt.xlabel("Speed (RPM)")
                plt.ylabel("Torque (lb.in)")
                plt.title("Polynomial Regression - Predictions")
                plt.legend()
                plt.grid(True)
                st.pyplot(plt.gcf())
                break

            except ValueError:
                st.error("Invalid input. Please enter a numerical value for speed.")

# -------------------------------------------------------------------END-------------------------------------------------------------------------------------------
