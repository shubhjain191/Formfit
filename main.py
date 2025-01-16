import streamlit as st
import cv2
import time
import ExerciseAiTrainer as exercise
from chatbot_ui import chat_ui

def main():
    # Set configuration for the theme before any other Streamlit command
    st.set_page_config(page_title='Formfit', layout='centered')
    
    # Define App Title and Structure
    st.title('Formfit: Fitness AI Coach 💪')

    # 2 Options: Chatbot, Auto Classify
    options = st.sidebar.selectbox(
        'Select Option 🏋️‍♀️', 
        ('Auto Classify', 'Chatbot')
    )

    # Chatbot section
    if options == 'Chatbot':
        st.markdown('---')
        st.markdown("The chatbot can make mistakes. Please verify important info. 🤖")
        chat_ui()

    # Auto Classify section
    elif options == 'Auto Classify':
        st.markdown('---')
        st.write('Click the button below to start automatic exercise classification and repetition counting 🏃‍♀️')
        st.markdown('---')
        st.write("Ensure you are clearly visible and facing the camera for accurate AI tracking. 👁️")
        auto_classify_button = st.button('Start Auto Classification 🚀')

        if auto_classify_button:
            with st.spinner('Classifying... Please wait ⏳'):
                time.sleep(2)
                exer = exercise.Exercise()
                exer.auto_classify_and_count()

    # Footer or additional info
    st.markdown('---')
    st.markdown('Made with ❤️ by Formfit Team')

if __name__ == '__main__':
    def load_css():
        with open("static/styles.css", "r") as f:
            css = f"<style>{f.read()}</style>"
            st.markdown(css, unsafe_allow_html=True)
    main()
