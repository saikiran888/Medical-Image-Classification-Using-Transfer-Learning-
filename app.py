import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import torchvision.models as models

# Load the pre-trained DenseNet model
model = models.densenet161(weights=models.DenseNet161_Weights.IMAGENET1K_V1)

# Adjust the classifier to match your trained model
num_classes = 2  # Assuming your model has 2 classes
model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)

# Load your state_dict
state_dict_path = 'my_model_pneumonia.pt'
state_dict = torch.load(state_dict_path, map_location=torch.device('cpu'))
new_state_dict = {k.replace('model.', ''): v for k, v in state_dict.items()}
model.load_state_dict(new_state_dict)
model.eval()

# Define the image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
])

def predict_image(image, model):
    image = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(image)
        _, predicted = torch.max(output, 1)
        prediction_score = torch.softmax(output, dim=1).max().item()
    return predicted.item(), prediction_score

st.title("Pneumonia Detection from Chest X-rays")
st.write("Upload a chest X-ray image to predict if it is infected with pneumonia.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    st.write("")
    st.write("Classifying...")
    prediction, prediction_score = predict_image(image, model)
    
    if prediction == 1 and prediction_score > 0.8:
        st.markdown(unsafe_allow_html=True, body="<span style='color:red; font-size: 50px'><strong><h4>Pneumonia! :slightly_frowning_face:</h4></strong></span>")
    else:
        st.markdown(unsafe_allow_html=True, body="<span style='color:green; font-size: 50px'><strong><h3>Healthy! :smile:</h3></strong></span>")
