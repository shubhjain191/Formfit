import streamlit as st
import cv2
import time
import ExerciseAiTrainer as exercise
from chatbot_ui import chat_ui

def main():
    # Page configuration
    st.set_page_config(
        page_title='Formfit',
        page_icon='💪',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    # Custom CSS for better styling
    st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 2rem;
        }
        .feature-card {
            padding: 1.5rem;
            border-radius: 10px;
            border: 1px solid #e0e0e0;
            background: white;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 1rem;
        }
        .feature-icon {
            font-size: 2rem;
            margin-bottom: 0.5rem;
        }
        .feature-title {
            font-size: 1.2rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .feature-description {
            color: #666;
            font-size: 0.9rem;
        }
        .cta-button {
            background-color: #4ECDC4;
            color: white;
            padding: 0.8rem 1.5rem;
            border-radius: 5px;
            text-decoration: none;
            font-weight: bold;
            display: inline-block;
            margin-top: 1rem;
        }
        .sidebar .selectbox {
            background-color: #f8f9fa;
            border-radius: 5px;
            padding: 0.5rem;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### 🏋️‍♂️ Formfit AI")
        st.markdown("---")
        options = st.selectbox(
            '🎯 Choose Your Mode',
            ('Auto Classify', 'Chatbot'),
            key='mode_select'
        )
        st.markdown("---")
        st.markdown("### Quick Tips")
        st.info("• Ensure good lighting\n• Face the camera\n• Wear fitted clothing\n• Clear your workout space")

    # Main content
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown('<h1 class="main-header">Formfit AI</h1>', unsafe_allow_html=True)
        st.markdown("#### Your Intelligent Workout Companion")

    # Features section
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <div class="feature-title">AI Form Analysis</div>
                <div class="feature-description">Real-time feedback on your exercise technique</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Smart Counter</div>
                <div class="feature-description">Precise repetition tracking with form validation</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">💬</div>
                <div class="feature-title">AI Coach</div>
                <div class="feature-description">Personalized guidance and form correction</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Auto Detection</div>
                <div class="feature-description">Automatic exercise identification</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Mode-specific content
    if options == 'Chatbot':
        st.markdown("### 💬 AI Form Coach")
        st.info("Ask questions about proper form, exercise techniques, or get workout advice!")
        chat_ui()

    elif options == 'Auto Classify':
        st.markdown("### 🎥 AI Form Analysis")
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
                Get real-time form analysis and counting for:
                - Push-ups
                - Squats
                - Bicep Curls
                - Shoulder Press
            """)
        
        with col2:
            start_button = st.button('Start Analysis 🚀', use_container_width=True)
            
        if start_button:
            with st.spinner('Initializing AI Analysis...'):
                time.sleep(1)
                exer = exercise.Exercise()
                exer.auto_classify_and_count()

    # Footer
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div style='text-align: center; color: #666;'>
            Made with ❤️ by Formfit Team<br>
            <small>Powered by Advanced AI & Computer Vision</small>
            </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()
