import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix, accuracy_score, f1_score
import seaborn as sns
import cv2
from keras.models import load_model
from PIL import Image
import os
from fpdf import FPDF
import mainTrain

# Load the trained model
model = load_model('BrainTumor10Epochs.h5')

# Directory containing test images
pred_folder = 'pred/'
report_folder = 'reports/'

# Check if report folder exists, if not, create one
if not os.path.exists(report_folder):
    os.makedirs(report_folder)

# Initialize PDF
pdf = FPDF()
pdf.set_auto_page_break(auto=True, margin=15)
pdf.add_page()
pdf.set_font("Arial", size=12)

# Define maximum width for each section
max_width_left_section = 120  # Left section for images and analysis
max_width_right_section = 80  # Right section for graphs

# Function to add an image, its prediction, and visualizations to the PDF
def add_image_and_visualizations_to_pdf(pdf, original_img_path, processed_img_path, prediction, image_title):
    # Add a page for the current image
    pdf.add_page()
    
    # Add left section for images and analysis
    add_left_section(pdf, original_img_path, processed_img_path, prediction)
    
    # Add right section for graphs
    add_right_section(pdf)

# Function to add left section (images and analysis) to the PDF
def add_left_section(pdf, original_img_path, processed_img_path, prediction):
    # Add original image
    pdf.cell(max_width_left_section, 10, txt="Original Image:", ln=True)
    original_img_y = pdf.get_y()  # get the current y-coordinate
    pdf.image(original_img_path, x=10, y=original_img_y, w=max_width_left_section/2)
    
    # Leave some space after the image
    image_height = max_width_left_section/2  # Assuming square images here
    pdf.set_y(original_img_y + image_height + 10)  # Adjust 10 or as needed

    # Add processed image
    processed_img_y = pdf.get_y()  # get the current y-coordinate
    pdf.image(processed_img_path, x=10, y=processed_img_y, w=max_width_left_section/2)
    
    # Leave some space after the image
    pdf.set_y(processed_img_y + image_height + 10)  # Adjust 10 or as needed
    
    # Convert prediction to string
    prediction_str = "Brain tumor is present" if prediction == 1 else "Brain tumor is absent"
    
    # Add prediction
    pdf.cell(max_width_left_section, 10, txt="Prediction: " + prediction_str, ln=True)

# Function to add right section (graphs) to the PDF
def add_right_section(pdf):
    # Add histogram
    pdf.cell(max_width_right_section, 10, txt="Histogram of Pixel Intensities:", ln=True)
    pdf.image('histogram.png', x=max_width_left_section + 10, y=20, w=max_width_right_section)
    
    # Add heatmap
    pdf.cell(max_width_right_section, 10, txt="Heatmap:", ln=True)
    pdf.image('heatmap.png', x=max_width_left_section + 10, y=120, w=max_width_right_section)
    
    # Add ROC curve
    pdf.cell(max_width_right_section, 10, txt="ROC Curve:", ln=True)
    pdf.image('roc_curve.png', x=max_width_left_section + 10, y=220, w=max_width_right_section)
    
    # Add Precision-Recall curve
    pdf.cell(max_width_right_section, 10, txt="Precision-Recall Curve:", ln=True)
    pdf.image('precision_recall_curve.png', x=max_width_left_section + 10, y=320, w=max_width_right_section)
    
    # Add Confusion Matrix
    pdf.cell(max_width_right_section, 10, txt="Confusion Matrix:", ln=True)
    pdf.image('confusion_matrix.png', x=max_width_left_section + 10, y=420, w=max_width_right_section)

# Get the list of selected images
image_folder = "C:/Users/vk768/OneDrive/Documents/Brain tumor prediction[1]/Brain tumor prediction/pred"
selected_images = os.listdir(image_folder)

# Initialize lists to store true labels and predicted labels
y_true = []
y_pred = []

# Iterate over selected images and generate predictions
for image_name in selected_images:
    # Extract the true label from the image name
    true_label = int(image_name.replace('pred', '').replace('.jpg', ''))
    y_true.append(true_label)  # Append true label to y_true list

    # Paths for images
    original_img_path = os.path.join(pred_folder, image_name)
    processed_img_path = os.path.join(report_folder, "processed_" + image_name)
    
    # Load and preprocess the image
    image = cv2.imread(original_img_path)
    resized_image = cv2.resize(image, (64, 64))
    img = Image.fromarray(resized_image)
    img_array = np.array(img) / 255.0  # Normalize pixel values
    input_img = np.expand_dims(img_array, axis=0)

    # Make prediction
    result = model.predict(input_img)
    prediction = 1 if result[0][0] > 0.5 else 0  # Convert prediction to binary label
    y_pred.append(prediction)  # Append predicted label to y_pred list
    
    # Generate insights
    
    # If tumor is present, process image for visualization
    if prediction == 1:
        # Convert the processed image to grayscale
        gray_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
        # Find contours around the detected area
        contours, _ = cv2.findContours(gray_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Draw contours on the processed image
        cv2.drawContours(resized_image, contours, -1, (0, 255, 0), 2)
        for contour in contours:
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)
            cv2.circle(resized_image, center, radius, (0, 0, 255), 2)
    
    # Save the processed image
    cv2.imwrite(processed_img_path, resized_image)

    # Add original image, processed image, prediction, and insights to the PDF
    add_image_and_visualizations_to_pdf(pdf, original_img_path, processed_img_path, prediction, image_name)

# Sample data (replace with actual data)
y_true = np.array(y_true)
y_pred = np.array(y_pred)

# Calculate accuracy and F1 score
accuracy = accuracy_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred, average='micro')  # Set average parameter to 'micro'

# Add final accuracy to PDF
pdf.cell(200, 10, txt=f"Final Accuracy: {accuracy:.2f}", ln=True)
pdf.cell(200, 10, txt=f"F1 Score: {f1:.2f}", ln=True)



# Save the PDF to a file
pdf.output("Brain_Tumor_Predictions_Report.pdf")

print("The report has been generated and saved as Brain_Tumor_Predictions_Report.pdf")
