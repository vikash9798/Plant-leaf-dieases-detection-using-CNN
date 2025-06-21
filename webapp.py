import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# CSS for  styling
st.markdown("""
<style>
    :root {
        --primary: #2e8b57;
        --secondary: #f8f9fa;
        --accent: #4CAF50;
        --dark: #343a40;
        --light: #ffffff;
    }
    
    .main {
        max-width: 1200px;
        padding: 2rem 4rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    .title {
        color: var(--primary);
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    
    .sidebar .sidebar-content {
        background-color: var(--secondary);
        padding: 1.5rem 1rem;
    }
    
    .stButton>button {
        background-color: var(--accent);
        color: white;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        width: 100%;
        transition: all 0.3s;
        font-weight: 600;
        border: none;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    
    .stButton>button:hover {
        background-color: #3e8e41;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    
    .prediction-card {
        border-radius: 12px;
        padding: 2rem;
        margin: 2rem 0;
        box-shadow: 0 6px 18px rgba(0,0,0,0.08);
        background-color: white;
        border: 1px solid #e9ecef;
    }
    
    .confidence-meter {
        height: 24px;
        background-color: #e9ecef;
        border-radius: 12px;
        margin: 1.5rem 0;
        overflow: hidden;
    }
    
    .confidence-fill {
        height: 100%;
        border-radius: 12px;
        background: linear-gradient(90deg, var(--accent), #81c784);
        text-align: center;
        color: white;
        line-height: 24px;
        font-size: 14px;
        font-weight: 600;
    }
    
    .uploaded-image {
        border-radius: 12px;
        border: 2px solid #e9ecef;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    .feature-card {
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        background-color: white;
        border: 1px solid #e9ecef;
        transition: all 0.3s;
        height: 100%;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    
    .feature-icon {
        font-size: 2.5rem;
        margin-bottom: 1rem;
        color: var(--primary);
    }
    
    .team-card {
        border-radius: 12px;
        padding: 2rem;
        background: linear-gradient(135deg, #f5f7fa 0%, #e4e8eb 100%);
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        text-align: center;
        max-width: 600px;
        margin: 2rem auto;
    }
    
    .team-image {
        width: 150px;
        height: 150px;
        border-radius: 50%;
        object-fit: cover;
        border: 5px solid white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 0 auto 1.5rem;
    }
    
    .social-links {
        display: flex;
        justify-content: center;
        gap: 1rem;
        margin-top: 1.5rem;
    }
    
    .social-icon {
        width: 40px;
        height: 40px;
        border-radius: 50%;
        background-color: white;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        transition: all 0.3s;
    }
    
    .social-icon:hover {
        transform: translateY(-3px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    .tech-stack {
        display: flex;
        flex-wrap: wrap;
        gap: 1rem;
        justify-content: center;
        margin: 2rem 0;
    }
    
    .tech-item {
        background-color: white;
        padding: 0.75rem 1.5rem;
        border-radius: 50px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        font-weight: 600;
        color: var(--dark);
        border: 1px solid #e9ecef;
    }
    
    .section-title {
        color: var(--primary);
        margin: 3rem 0 1.5rem;
        font-weight: 700;
        position: relative;
        padding-bottom: 0.5rem;
    }
    
    .section-title:after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        width: 60px;
        height: 3px;
        background-color: var(--accent);
    }
    
    .tab-content {
        padding: 2rem 0;
    }
    
    .metric-card {
        background-color: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        text-align: center;
        border: 1px solid #e9ecef;
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: var(--primary);
        margin: 0.5rem 0;
    }
    
    .metric-label {
        color: #6c757d;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

#getting  the Model for plant leaf dieases detection 
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("trained_plant_disease_model_grayscale.keras",compile=False)

# Model Prediction Function
def model_prediction(test_image):
    model = load_model()

    # all images are loaded  in grayscale
    image = tf.keras.preprocessing.image.load_img(
        test_image, color_mode="grayscale", target_size=(128, 128)
    )

    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = input_arr / 255.0 
    input_arr = np.expand_dims(input_arr, axis=0) 

    predictions = model.predict(input_arr)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions)

    return predicted_index, confidence

# List of class names of all pant leaf that have been used in model training
class_names = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    # 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
    # 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    # 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
    # 'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
    # 'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    # 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
    # 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
    # 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
    # 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
    # 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    # 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
    # 'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
    # 'Tomato___healthy'
]

# Sidebar
st.sidebar.title("🌱 PlantAI Diagnostics")
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2913/2913103.png", width=80)
app_mode = st.sidebar.radio("", ["Home", "About", "Disease Recognition"])

# Footer in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
<small>© 2025 All rights reserved.Desgine & Developed by Vikash</small>
""", unsafe_allow_html=True)

# Home Page
if app_mode == "Home":
    st.markdown("<h1 class='title'>AI Powered Model to detect diseases of your plant leaf</h1>", unsafe_allow_html=True)
    
    # Hero section
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        ### Revolutionizing Plant Health Monitoring with Deep Learning
        
        Our cutting-edge and ML trained system empowers farmers, gardeners, and agricultural professionals 
        to identify plant diseases with unprecedented accuracy and speed.
        
        **Early detection leads to better crop protection and higher yields!**
        """)
        st.markdown("""
        <a href="#disease-recognition" style="
            background-color: #2e8b57;
            color: white;
            padding: 0.75rem 1.5rem;
            text-align: center;
            text-decoration: none;
            display: inline-block;
            font-size: 1rem;
            font-weight: 600;
            margin: 1rem 0;
            border-radius: 8px;
            cursor: pointer;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
            transition: all 0.4s;
        ">See More →</a>
        """, unsafe_allow_html=True)
    
    with col2:
        st.image("https://images.pexels.com/photos/1314186/pexels-photo-1314186.jpeg?auto=compress&cs=tinysrgb&w=600",width=80,use_container_width=True)
    
    st.markdown("---")
    
    # Features section
    st.markdown("<h2 class='section-title'>Key Features</h2>", unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🔍</div>
            <h4>Advanced Detection</h4>
            <p>State-of-the-art CNN model trained on 87,000 leaf images with very good accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[1]:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">⚡</div>
            <h4>Real-Time Analysis</h4>
            <p>Get instant results in under few seconds without waiting for lab tests</p>
        </div>
        """, unsafe_allow_html=True)
    
    with cols[2]:
        st.markdown("""
        <div class="feature-card">
            <div class="feature-icon">🌐</div>
            <h4>Comprehensive Coverage</h4>
            <p>38+ disease classes across 14 plant species with continuous updates</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Features  wirking model section
    st.markdown("<h2 class='section-title'>How It Works</h2>", unsafe_allow_html=True)
    steps = st.columns(3)
    with steps[0]:
        st.image("https://cdn-icons-png.flaticon.com/512/3342/3342137.png", width=100)
        st.markdown("""
        <h4 style='color: #2e8b57;'>1. Upload Image</h4>
        <p>Capture or upload a clear image of your plant leaf</p>
        """, unsafe_allow_html=True)
    with steps[1]:
        st.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
        st.markdown("""
        <h4 style='color: #2e8b57;'>2.ML Model Processing</h4>
        <p>Our deep learning model analyzes the image features</p>
        """, unsafe_allow_html=True)
    with steps[2]:
        st.image("https://cdn-icons-png.flaticon.com/512/190/190411.png", width=100)
        st.markdown("""
        <h4 style='color: #2e8b57;'>3. Get Diagnosis</h4>
        <p>Receive detailed report with confidence score</p>
        """, unsafe_allow_html=True)

# About Page
elif app_mode == "About":
    st.markdown("<h2 class='title'>PlantAI Data: What Powers Our Predictions</h2>", unsafe_allow_html=True)
    
    # different about sections
    tab1, tab2, tab3 = st.tabs(["📊 Dataset", "⚙️ Technology", "👨‍💻 Developer"])
    
    with tab1:
        st.markdown("<h2 class='section-title'>Dataset Information</h2>", unsafe_allow_html=True)
        st.markdown("""
        This model is built on a high-quality dataset of plant leaf disease images,
        thoroughly curated and augmented using state-of-the-art techniques to boost generalization and model robustness..
        """)
        
        # Dataset stats
        st.markdown("<h3 style='color: #2e8b57;'>Dataset Statistics</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">87K</div>
                <div class="metric-label">Total Images</div>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">38</div>
                <div class="metric-label">Disease Classes</div>
            </div>
            """, unsafe_allow_html=True)
        with col3:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-value">14</div>
                <div class="metric-label">Plant Species</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("""
        #### Dataset Composition:
        - **Training Set:** 17,572 images  
        - **Validation Set:** 17,572 images  
        - **Test Set:** 33 images  
        """)
        
        st.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcTvAL9dK7u02wMaSE0TTE89QuJ3gQ2HnYC5gQ&s",width=100,use_container_width=True)
        
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem;">
            <a href="#disease-recognition" style="
                background-color: #2e8b57;
                color: white;
                padding: 0.75rem 1.5rem;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 1rem;
                font-weight: 600;
                margin: 1rem 0;
                border-radius: 8px;
                cursor: pointer;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                transition: all 0.3s;
            ">Try Disease Detection →</a>
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        st.markdown("<h2 class='section-title'>Technology Stack</h2>", unsafe_allow_html=True)
        st.markdown("""
        At the core of our technology stack is a Convolutional Neural Network (CNN) model,
        designed to analyze plant leaf images with remarkable accuracy, 
        supported by robust data processing and deployment frameworks
        """)
        
        st.markdown("<h3 style='color: #2e8b57;'>Core Technologies</h3>", unsafe_allow_html=True)
        
        st.markdown("""
        <div class="tech-stack">
            <div class="tech-item">TensorFlow</div>
            <div class="tech-item">Keras</div>
            <div class="tech-item">CNN</div>
            <div class="tech-item">python</div>
            <div class="tech-item">Numpy</div>
            <div class="tech-item">pandas</div>
            <div class="tech-item">openCv</div>
            <div class="tech-item">Seaborn</div>
            <div class="tech-item">Stremlit</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        ### Model Architecture
        Our convolutional neural network is specifically designed for plant disease recognition:
        """)
        st.code("""
        Model: "sequential"
        _____________________________________________________________
        Layer (type)                Output Shape              Param #   
        =============================================================
        conv2d (Conv2D)            (None, 126, 126, 32)      320       
        max_pooling2d (MaxPooling2D)(None, 63, 63, 32)        0         
        conv2d_1 (Conv2D)          (None, 61, 61, 64)        18496     
        max_pooling2d_1 (MaxPooling2D)(None, 30, 30, 64)      0         
        flatten (Flatten)          (None, 57600)             0         
        dense (Dense)              (None, 128)               7372928   
        dense_1 (Dense)            (None, 38)                4902      
        =============================================================
        Total params: 7,396,646
        Trainable params: 7,396,646
        Non-trainable params: 0
        """, language='python')
        
        # CSS section for core technology
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem;">
            <a href="#disease-recognition" style="
                background-color: #2e8b57;
                color: white;
                padding: 0.75rem 1.5rem;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 1rem;
                font-weight: 600;
                margin: 1rem 0;
                border-radius: 8px;
                cursor: pointer;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                transition: all 0.3s;
            ">Try Our Technology →</a>
        </div>
        """, unsafe_allow_html=True)
    
    with tab3:
        st.markdown("""
<style>
    .team-card {
        display: flex;
        flex-direction: row;
        align-items: center;
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        gap: 20px;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    .team-image {
        width: 130px;
        height: 130px;
        border-radius: 50%;
        border: 3px solid #2e8b57;
        object-fit: cover;
    }
    .bio-info {
        max-width: 600px;
    }
    .social-links a {
        display: inline-block;
        margin-right: 15px;
        text-decoration: none;
        color: #2e8b57;
        font-weight: bold;
    }
    .social-links a:hover {
        text-decoration: underline;
    }
</style>

<h2>👨‍💻Behind the Model</h2>

<div class="team-card">
    <img src="https://lh3.googleusercontent.com/a/ACg8ocLQtMa-QxUyJfTQ2FR5rtkgd09K4JKupIB1go5DSaQT_rnlQ3fbBw=s432-c-no" class="team-image">
    <div class="bio-info">
        <h3>Vikash Kumar</h3>
        <p style="color: #6c757d; font-style: italic;">Data Scientist</p>
        <p>Aspiring Data Scientist focused on applying AI and Deep Learning to real-world problems to improve efficiency,
         decision-making, and sustainability across industries.</p>
        <div class="social-links">
            🔗 <a href="https://www.linkedin.com/in/vikash-kumar-26b186292/" target="_blank">LinkedIn</a>
            💻 <a href="https://github.com/vikash9798" target="_blank">GitHub</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

        st.markdown("""
        ### 🎓 Career Snapshot
        - **Education:** Master in Computer Application, Dr.Hari Singh Gour University, Sagar  
        - **Experience:** 6 months of industry experience in Data Analysis with over 10 completed projects in Data domains.   
        - **Research Interests** Machine learning applications, data-driven decision making, computer vision, and AI for real-world problem solving across industries.
        ### 💡 Purpose
        *"To bridge the gap between advanced AI technologies and real-world applications by creating intelligent, 
        data-driven solutions that promote efficiency, sustainability, and social impact."*
        """)      
        # Add button to go to Disease Recognition
        st.markdown("""
        <div style="text-align: center; margin-top: 2rem;">
            <a href="#disease-recognition" style="
                background-color: #2e8b57;
                color: white;
                padding: 0.75rem 1.5rem;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 1rem;
                font-weight: 600;
                margin: 1rem 0;
                border-radius: 8px;
                cursor: pointer;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                transition: all 0.3s;
            ">Try the App Now →</a>
        </div>
        """, unsafe_allow_html=True)

# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.markdown("<h1 class='title'>Smart Plant Disease Diagnosis System</h1>", unsafe_allow_html=True)

    # Upload section
    with st.container():
        st.markdown("<h3 class='section-title'>Image Upload & Analysis Guidelines</h3>", unsafe_allow_html=True)
        st.markdown("""
        Upload a clear image of a plant leaf for instant disease detection.  
        For optimal results, ensure:
        - The leaf occupies most of the image  
        - Good lighting conditions  
        - Plain background preferred  
        """)

        test_image = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

        if test_image:
            st.markdown("<h3 style='color: #2e8b57;'>Uploaded Image</h3>", unsafe_allow_html=True)
            st.image(test_image,use_container_width=True, caption="Preview of your uploaded image")

            if st.button("Analyze Image", key="predict_button"):
                with st.spinner("Processing image with AI... Please wait..."):
                    try:
                        st.snow()
                        result_index, confidence =model_prediction(test_image) #correct this section because it failed to load the model
                        predicted_class = class_names[result_index]

                        disease_name = " ".join(predicted_class.split("___")[1].split("_"))
                        plant_name = predicted_class.split("___")[0].replace("_", " ")

                        st.markdown(f"""
                        <div class="prediction-card">
                            <h3>Diagnosis Results</h3>
                            <hr style='border: 1px solid #e9ecef; margin: 1rem 0;'>
                            <div style='margin-bottom: 1rem;'>
                                <p style='font-size: 1.1rem; margin-bottom: 0.5rem;'><strong>Plant:</strong> {plant_name}</p>
                                <p style='font-size: 1.1rem;'><strong>Condition:</strong> 
                                    <span style='color: {"#4CAF50" if "healthy" in predicted_class.lower() else "#f39c12"};'>{disease_name if disease_name.lower() != "healthy" else "✅ Healthy"}</span>
                                </p>
                            </div>
                            <div style='margin: 1.5rem 0;'>
                                <p style='margin-bottom: 0.5rem;'>Confidence Level:</p>
                                <div class="confidence-meter">
                                    <div class="confidence-fill" style="width: {confidence * 100}%">{round(confidence * 100, 2)}%</div>
                                </div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        st.markdown("<h3 style='color: #2e8b57;'>Recommendations</h3>", unsafe_allow_html=True)

                        if "healthy" in predicted_class.lower():
                            st.success("""
                            **🎉 Excellent News!**  
                            Your plant appears to be in good health.  
                            **Maintenance Tips:**  
                            - Continue current care regimen  
                            - Monitor weekly for any changes  
                            - Ensure proper sunlight and watering  
                            - Consider periodic nutrient supplements  
                            """)
                        else:
                            st.warning("""
                            **⚠️ Disease Detected**  
                            Our system identified potential disease symptoms.  
                            **Immediate Actions:**  
                            - Isolate affected plants to prevent spread  
                            - Remove severely infected leaves carefully  
                            - Consult with agricultural expert for confirmation  
                            - Review treatment options below  
                            """)

                            with st.expander("📚 Detailed Disease Information & Treatment Options", expanded=False):
                                st.markdown(f"""
                                **{disease_name} in {plant_name}**  
                                *Recommended treatment protocol*

                                **Symptoms:**  
                                (Detailed description of symptoms would appear here)

                                **Lifecycle:**  
                                (Information about disease development)

                                **Treatment Options:**  
                                1. **Organic:** Neem oil, copper fungicides  
                                2. **Chemical:** (Specific fungicides if applicable)  
                                3. **Cultural:** Crop rotation, proper spacing  

                                **Prevention:**  
                                - Regular monitoring  
                                - Proper irrigation practices  
                                - Resistant varieties  

                                *Consult local agricultural extension for region-specific recommendations.*
                                """)
                    except Exception as e:
                        st.error(f"""
                        **❌ Analysis Error**  
                        We encountered an issue processing your image:  
                        `{str(e)}`  
                        
                        **Troubleshooting Tips:**  
                        - Try a different image file  
                        - Ensure the image is in JPG, JPEG, or PNG format  
                        - Check that the file isn't corrupted  
                        - Contact support if issue persists  
                        """)

        else:
            st.info("""
            **ℹ️ How to Get Started**  
            Upload an image above or try with these example images:
            """)
            example_cols = st.columns(3)
            example_images = [
                "https://www.lovethegarden.com/sites/default/files/styles/scale_xl_2x_col_6/public/content/articles/uk/plant-leaf-problems-aphids.jpg.webp?itok=GE689jwm",
                "https://www.lovethegarden.com/sites/default/files/styles/scale_xl_2x_col_6/public/content/articles/uk/plant-leaf-problems-discoloured-leaves.jpg.webp?itok=iOZk8GQg",
                "https://www.lovethegarden.com/sites/default/files/styles/scale_xl_2x_col_6/public/content/articles/uk/plant-leaf-problems-vine-weevils.jpg.webp?itok=DBBoTRqA"
            ]
            for col, img in zip(example_cols, example_images):
                with col:
                    st.image(img,use_container_width=True)
                    st.caption("Example plant leaf")