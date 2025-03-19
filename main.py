import streamlit as st
import cv2
import time
import ExerciseAiTrainer as exercise
from chatbot_ui import chat_ui

def main():
    # Page configuration - sets up the browser tab appearance
    st.set_page_config(
        page_title='FormFit AI',  # Updated name with proper capitalization
        page_icon='💪',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    # Enhanced CSS for modern, clean UI design with better readability and visual hierarchy
    st.markdown("""
        <style>
        /* Main header styling with enhanced gradient */
        .main-header {
            font-size: 3rem;  /* Increased size for better visibility */
            font-weight: 700;  /* Bolder font for emphasis */
            background: linear-gradient(45deg, #FF6B6B, #4ECDC4, #556DFF);  /* Expanded gradient with more colors */
            background-size: 200% auto;  /* Gradient size for animation */
            animation: gradient-shift 10s ease infinite;  /* Subtle animation for visual interest */
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 1.5rem;
            text-align: center;  /* Center align header */
        }
        
        /* Gradient animation keyframes */
        @keyframes gradient-shift {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }
        
        /* Subtitle styling */
        .subtitle {
            font-size: 1.5rem;
            color: #555;
            text-align: center;
            margin-bottom: 2rem;
            font-weight: 400;  /* Lighter weight for contrast with header */
        }
        
        /* Enhanced feature cards with hover effects */
        .feature-card {
            padding: 2rem;  /* More padding for spacious feel */
            border-radius: 12px;  /* More rounded corners */
            border: none;  /* Remove border for cleaner look */
            background: linear-gradient(145deg, #ffffff, #f0f0f0);  /* Subtle gradient background */
            box-shadow: 0 8px 15px rgba(0, 0, 0, 0.1);  /* Enhanced shadow */
            margin-bottom: 1.5rem;
            transition: all 0.3s ease;  /* Smooth transition for hover effects */
            height: 100%;  /* Consistent height */
        }
        
        /* Card hover effect */
        .feature-card:hover {
            transform: translateY(-5px);  /* Slight lift effect */
            box-shadow: 0 12px 20px rgba(0, 0, 0, 0.15);  /* Enhanced shadow on hover */
        }
        
        /* Feature icon with improved styling */
        .feature-icon {
            font-size: 2.5rem;  /* Larger icons */
            margin-bottom: 1rem;  /* More space below icons */
            color: #4ECDC4;  /* Themed color */
            text-align: center;
        }
        
        /* Feature title styling */
        .feature-title {
            font-size: 1.3rem;
            font-weight: 600;
            margin-bottom: 0.75rem;
            color: #333;  /* Darker for better contrast */
            text-align: center;
        }
        
        /* Feature description styling */
        .feature-description {
            color: #555;  /* Slightly darker for better readability */
            font-size: 1rem;
            text-align: center;
            line-height: 1.5;  /* Improved line height for readability */
        }
        
        /* Call-to-action button with improved styling */
        .cta-button {
            background: linear-gradient(90deg, #4ECDC4, #556DFF);  /* Gradient background */
            color: white;
            padding: 1rem 2rem;  /* Larger padding for better presence */
            border-radius: 50px;  /* Pill shape button */
            text-decoration: none;
            font-weight: 600;
            display: inline-block;
            margin-top: 1.5rem;
            transition: all 0.3s ease;  /* Smooth transition */
            text-align: center;
            border: none;
            cursor: pointer;
            box-shadow: 0 4px 10px rgba(78, 205, 196, 0.3);  /* Button shadow */
        }
        
        /* Button hover effect */
        .cta-button:hover {
            transform: translateY(-2px);  /* Slight lift */
            box-shadow: 0 6px 15px rgba(78, 205, 196, 0.4);  /* Enhanced shadow */
            background-position: right center;  /* Shift gradient on hover */
        }
        
        /* Sidebar improvements */
        .sidebar .selectbox {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 0.75rem;
            border: 1px solid #e0e0e0;
            margin-bottom: 1rem;
        }
        
        /* Section dividers */
        .divider {
            margin: 2rem 0;
            border-top: 1px solid #eaeaea;
        }
        
        /* Tips box styling */
        .tips-box {
            background-color: #f1f8ff;
            border-left: 4px solid #4ECDC4;
            padding: 1rem;
            border-radius: 0 8px 8px 0;
            margin-top: 1rem;
        }
        
        /* Footer styling */
        .footer {
            text-align: center;
            color: #666;
            padding: 1.5rem 0;
            font-size: 0.9rem;
            margin-top: 2rem;
        }
        
        /* Mode header styling */
        .mode-header {
            font-size: 1.8rem;
            font-weight: 600;
            color: #333;
            margin-bottom: 1rem;
            border-bottom: 2px solid #4ECDC4;
            padding-bottom: 0.5rem;
            display: inline-block;
        }
        
        /* Info boxes styling */
        .stAlert {
            border-radius: 10px !important;
            border: none !important;
            box-shadow: 0 4px 10px rgba(0, 0, 0, 0.05) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    # Sidebar for navigation - improved styling and organization
    with st.sidebar:
        # App logo and branding in sidebar
        st.markdown("""
            <div style="text-align: center; margin-bottom: 2rem;">
                <h2 style="color: #4ECDC4; font-weight: 700;">🏋️‍♂️ FormFit AI</h2>
                <p style="font-size: 0.9rem; color: #666;">Smart Workout Analysis</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)  # Visual separator
        
        # Mode selection with improved UI
        st.markdown("#### 🎯 Choose Your Mode")
        options = st.selectbox(
            'Select training mode',
            ('Auto Classify', 'Chatbot'),
            key='mode_select',
            help="Choose how you want to use FormFit AI"  # Help tooltip
        )
        
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)  # Visual separator
        
        # Enhanced tips section with better formatting
        st.markdown("#### 💡 Quick Tips")
        st.markdown("""
            <div class="tips-box">
                <ul style="margin-left: 1rem; padding-left: 0;">
                    <li><strong>Lighting:</strong> Ensure good, even lighting</li>
                    <li><strong>Position:</strong> Face the camera directly</li>
                    <li><strong>Clothing:</strong> Wear fitted, high-contrast clothing</li>
                    <li><strong>Space:</strong> Clear your workout area of obstacles</li>
                    <li><strong>Distance:</strong> Position 6-8 feet from camera</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        # Version information for transparency
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
        st.markdown("""
            <div style="text-align: center; margin-top: 2rem;">
                <p style="font-size: 0.8rem; color: #999;">FormFit AI v2.0</p>
            </div>
        """, unsafe_allow_html=True)

    # Main content area with improved header styling
    col1, col2, col3 = st.columns([1, 2, 1])  # Column layout for centered content
    with col2:
        # Main header with animated gradient effect
        st.markdown('<h1 class="main-header">FormFit AI</h1>', unsafe_allow_html=True)
        # Descriptive subtitle
        st.markdown('<p class="subtitle">Advanced Exercise Analysis & Real-time Coaching</p>', unsafe_allow_html=True)

    # Features section with enhanced visual cards
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    
    # Creating four equal columns for feature cards
    col1, col2, col3, col4 = st.columns(4)

    # Enhanced feature cards with better descriptions and visual styling
    with col1:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🤖</div>
                <div class="feature-title">AI Form Analysis</div>
                <div class="feature-description">Get precise feedback on your exercise technique with computer vision analysis</div>
            </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">📊</div>
                <div class="feature-title">Smart Rep Counter</div>
                <div class="feature-description">Automatic counting with form validation ensures quality over quantity</div>
            </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">💬</div>
                <div class="feature-title">AI Coach</div>
                <div class="feature-description">Receive personalized guidance and form correction in real-time</div>
            </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown("""
            <div class="feature-card">
                <div class="feature-icon">🎯</div>
                <div class="feature-title">Auto Detection</div>
                <div class="feature-description">Smart exercise identification without manual selection needed</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Mode-specific content with improved UI elements
    if options == 'Chatbot':
        # Chatbot interface with better introduction and explanation
        st.markdown('<div class="mode-header">💬 AI Form Coach</div>', unsafe_allow_html=True)
        
        # More informative instructions for the chatbot
        st.info("""
            Your personal AI fitness assistant is ready to help!
            - Ask about proper exercise techniques
            - Get personalized workout recommendations
            - Learn about injury prevention
            - Receive nutrition and recovery advice
        """)
        
        # Launch the chatbot UI
        chat_ui()  # This calls the imported chat interface

    elif options == 'Auto Classify':
        # Auto classification interface with better layout and instructions
        st.markdown('<div class="mode-header">🎥 AI Form Analysis</div>', unsafe_allow_html=True)
        
        # Create two columns for instructions and start button
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # More detailed instructions with visual hierarchy
            st.markdown("""
                ### How It Works
                1. **Position yourself** in view of your camera
                2. **Begin your exercise** when ready
                3. **AI will automatically detect** your movement type
                4. **Receive real-time feedback** on your form
                
                #### Supported Exercises:
                - Push-ups
                - Squats
                - Bicep Curls
                - Shoulder Press
            """)
        
        with col2:
            # More prominent and attractive start button
            st.markdown("""
                <div style="background: #f8f9fa; padding: 1.5rem; border-radius: 12px; text-align: center; box-shadow: 0 4px 10px rgba(0,0,0,0.05);">
                    <p style="margin-bottom: 1rem;">Ready to analyze your form?</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Start button with more prominent styling
            start_button = st.button('Start Analysis 🚀', use_container_width=True)
            
        # Logic for when start button is clicked
        if start_button:
            # Loading state with more informative message
            with st.spinner('Initializing AI Analysis... Please prepare for your workout'):
                time.sleep(1)  # Brief delay to initialize components
                exer = exercise.Exercise()  # Create Exercise class instance
                exer.auto_classify_and_count()  # Start the exercise analysis

    # Footer section with improved styling and content
    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
            <div class="footer">
                <p>Made with ❤️ by Shubh Jain</p>
            </div>
        """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()  # Run the main application function
