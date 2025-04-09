#--- START OF FILE main.py ---

import streamlit as st
import cv2 # Keep cv2 import though not directly used in main UI for consistency if ExerciseAiTrainer needs it loaded
import time
import ExerciseAiTrainer as exercise
from chatbot_ui import chat_ui

def main():
    # Page configuration - sets up the browser tab appearance
    st.set_page_config(
        page_title='FormFit AI - Smart Workout Analysis', # More descriptive title
        page_icon='💪',
        layout='wide',
        initial_sidebar_state='expanded'
    )

    # --- Enhanced CSS for a Premium, Modern UI/UX ---
    st.markdown("""
        <style>
        /* === Variables for Consistent Theming === */
        :root {
            --primary-color: #4ECDC4; /* Teal */
            --secondary-color: #556DFF; /* Blue */
            --accent-color: #FF6B6B; /* Coral */
            --text-color: #333; /* Dark Grey */
            --text-color-light: #555; /* Medium Grey */
            --bg-color-light: #f8f9fa; /* Very Light Grey */
            --bg-color-white: #ffffff;
            --border-color: #eaeaea; /* Light Grey Border */
            --border-radius-md: 12px;
            --border-radius-sm: 8px;
            --shadow-sm: 0 2px 4px rgba(0, 0, 0, 0.05);
            --shadow-md: 0 8px 15px rgba(0, 0, 0, 0.1);
            --shadow-lg: 0 12px 25px rgba(0, 0, 0, 0.15);
            --font-family: 'Inter', sans-serif; /* Clean sans-serif font */
        }

        /* Apply base font globally if possible */
        body {
            font-family: var(--font-family) !important;
            color: var(--text-color);
        }

        /* === Main Header Styling === */
        .main-header {
            font-size: 3.5rem; /* Slightly larger */
            font-weight: 700;
            background: linear-gradient(45deg, var(--accent-color), var(--primary-color), var(--secondary-color));
            background-size: 300% auto; /* Slower, wider gradient */
            animation: gradient-shift 12s ease infinite;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.5rem; /* Reduced margin */
            text-align: center;
            line-height: 1.2; /* Tighter line height for large header */
        }

        /* Gradient Animation */
        @keyframes gradient-shift {
            0% {background-position: 0% 50%;}
            50% {background-position: 100% 50%;}
            100% {background-position: 0% 50%;}
        }

        /* === Subtitle Styling === */
        .subtitle {
            font-size: 1.6rem; /* Slightly larger */
            color: var(--text-color-light);
            text-align: center;
            margin-bottom: 2.5rem; /* Increased spacing after subtitle */
            font-weight: 400;
        }

        /* === Feature Cards - Enhanced === */
        .feature-card {
            padding: 2rem 1.5rem; /* Adjusted padding */
            border-radius: var(--border-radius-md);
            border: 1px solid var(--border-color); /* Subtle border */
            background: var(--bg-color-white); /* Clean white background */
            box-shadow: var(--shadow-sm); /* Softer shadow */
            margin-bottom: 1.5rem;
            transition: transform 0.3s ease, box-shadow 0.3s ease; /* Smooth transitions */
            height: 100%; /* Ensure cards in a row have same height */
            display: flex; /* Use flexbox for alignment */
            flex-direction: column; /* Stack content vertically */
            align-items: center; /* Center content horizontally */
            text-align: center; /* Center text */
        }

        .feature-card:hover {
            transform: translateY(-6px); /* More noticeable lift */
            box-shadow: var(--shadow-md); /* Enhanced shadow on hover */
        }

        /* Feature Icon */
        .feature-icon {
            font-size: 3rem; /* Larger icon */
            margin-bottom: 1rem;
            color: var(--primary-color); /* Use primary theme color */
        }

        /* Feature Title */
        .feature-title {
            font-size: 1.25rem; /* Slightly adjusted size */
            font-weight: 600;
            margin-bottom: 0.75rem;
            color: var(--text-color);
        }

        /* Feature Description */
        .feature-description {
            color: var(--text-color-light);
            font-size: 0.95rem; /* Slightly adjusted size */
            line-height: 1.6; /* Better readability */
            flex-grow: 1; /* Allows description to push footer elements down if needed */
        }

        /* === Buttons Styling (Targeting Streamlit's Button) === */
        /* Style the container around the button for better control */
        div[data-testid="stButton"] > button {
            background: linear-gradient(90deg, var(--primary-color), var(--secondary-color));
            color: white;
            padding: 0.75rem 1.5rem; /* Adjusted padding */
            border-radius: 50px; /* Pill shape */
            border: none;
            font-weight: 600;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            box-shadow: var(--shadow-sm);
            width: 100%; /* Make button fill container if use_container_width=True */
        }

        div[data-testid="stButton"] > button:hover {
            transform: translateY(-2px);
            box-shadow: var(--shadow-md);
            filter: brightness(110%); /* Slight brightness increase */
        }
        
        div[data-testid="stButton"] > button:active {
            transform: translateY(0px);
            box-shadow: var(--shadow-sm);
            filter: brightness(100%);
        }


        /* === Sidebar Enhancements === */
        .st-emotion-cache-1lcbmhc { /* More specific selector for sidebar background */
             background-color: var(--bg-color-white);
        }

        .st-emotion-cache-16txtl3 { /* Selector for sidebar content area */
            padding: 1.5rem;
        }

        /* Sidebar Header */
        .sidebar-header {
            text-align: center;
            margin-bottom: 2rem;
        }
        .sidebar-title {
            color: var(--primary-color);
            font-weight: 700;
            font-size: 1.5rem;
            margin-bottom: 0.25rem;
        }
        .sidebar-subtitle {
            font-size: 0.9rem;
            color: var(--text-color-light);
        }

        /* Sidebar Selectbox */
        div[data-testid="stSelectbox"] > div {
            background-color: var(--bg-color-light);
            border-radius: var(--border-radius-sm);
            border: 1px solid var(--border-color);
            padding: 0.5rem; /* Add internal padding */
            box-shadow: none;
        }
        div[data-testid="stSelectbox"] label {
             font-weight: 600;
             font-size: 1rem;
             margin-bottom: 0.5rem;
        }


        /* === Section Dividers === */
        hr.styled-divider {
            border: none;
            height: 1px;
            background-color: var(--border-color);
            margin: 3rem 0; /* Increased margin for spacing */
        }

        /* === Tips Box Styling === */
        .tips-box {
            background-color: #e6f7ff; /* Lighter blue */
            border-left: 4px solid var(--primary-color);
            padding: 1rem 1.5rem;
            border-radius: 0 var(--border-radius-sm) var(--border-radius-sm) 0;
            margin-top: 1rem;
            box-shadow: var(--shadow-sm);
        }
        .tips-box ul {
            margin-left: 0.5rem; /* Reduced indent */
            padding-left: 1rem;
            list-style: none; /* Remove default bullets */
        }
         .tips-box li {
            margin-bottom: 0.5rem;
            position: relative;
            padding-left: 1.2rem; /* Space for custom bullet */
        }
        .tips-box li::before {
            content: '💡'; /* Use icon as bullet */
            position: absolute;
            left: 0;
            top: 0;
            font-size: 0.9rem;
        }
         .tips-box strong {
             color: var(--primary-color);
         }

        /* === Footer Styling === */
        .footer {
            text-align: center;
            color: #aaa; /* Lighter grey for less emphasis */
            padding: 2rem 0 1rem 0;
            font-size: 0.85rem;
            border-top: 1px solid var(--border-color); /* Add subtle top border */
            margin-top: 3rem;
        }

        /* === Mode Header Styling === */
        .mode-header {
            font-size: 2rem; /* Larger section header */
            font-weight: 600;
            color: var(--text-color);
            margin-bottom: 1.5rem; /* More space below header */
            border-bottom: 3px solid var(--primary-color); /* Thicker, colored border */
            padding-bottom: 0.5rem;
            display: inline-block; /* Ensure border only spans text width */
        }

        /* === Info Boxes (st.info) Styling === */
        div[data-testid="stAlert"] {
            border-radius: var(--border-radius-sm) !important;
            border: 1px solid var(--primary-color) !important; /* Use primary color for border */
            background-color: #e6f7ff !important; /* Lighter blue background */
            box-shadow: var(--shadow-sm) !important;
            padding: 1rem 1.5rem !important; /* Adjust padding */
            color: var(--text-color) !important; /* Ensure good text contrast */
        }
        div[data-testid="stAlert"] ul {
             margin-left: 0.5rem;
             padding-left: 1rem;
        }
         div[data-testid="stAlert"] li {
             margin-bottom: 0.3rem;
         }

        /* === Instruction Box Styling (Auto Classify) === */
        .instruction-box {
            background: var(--bg-color-light);
            padding: 1.5rem 2rem;
            border-radius: var(--border-radius-md);
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
            height: 100%; /* Match height with sibling column */
        }
        .instruction-box h3 {
            margin-top: 0; /* Remove default top margin */
            margin-bottom: 1rem;
            color: var(--secondary-color); /* Use secondary color for emphasis */
            font-weight: 600;
        }
         .instruction-box h4 {
            margin-top: 1.5rem;
            margin-bottom: 0.5rem;
            color: var(--text-color);
            font-weight: 600;
        }
        .instruction-box li {
            margin-bottom: 0.5rem;
            line-height: 1.5;
        }
        .instruction-box strong {
            color: var(--primary-color);
        }

        /* === Call to Action Box Styling (Auto Classify) === */
         .cta-box {
            background: var(--bg-color-white);
            padding: 2rem;
            border-radius: var(--border-radius-md);
            text-align: center;
            border: 1px solid var(--border-color);
            box-shadow: var(--shadow-sm);
            height: 100%; /* Match height with sibling column */
            display: flex;
            flex-direction: column;
            justify-content: center; /* Center content vertically */
            align-items: center;
        }
        .cta-box p {
            font-size: 1.1rem;
            color: var(--text-color-light);
            margin-bottom: 1.5rem; /* Space before button */
        }

        /* General spacing helper */
        .section-spacer {
            margin: 3rem 0;
        }

        </style>
    """, unsafe_allow_html=True)

    # --- Sidebar ---
    with st.sidebar:
        # App logo and branding
        st.markdown("""
            <div class="sidebar-header">
                <div class="sidebar-title">🏋️‍♂️ FormFit AI</div>
                <div class="sidebar-subtitle">Smart Workout Analysis</div>
            </div>
        """, unsafe_allow_html=True)

        # Mode selection
        st.markdown("##### **🎯 Choose Your Mode**") # Bolder label
        options = st.selectbox(
            'Select training mode:', # Clearer label
            ('Auto Classify', 'Chatbot'),
            key='mode_select',
            label_visibility="collapsed", # Hide default label, use markdown above
            help="Select 'Auto Classify' for real-time form analysis or 'Chatbot' for fitness advice."
        )

        st.markdown('<hr class="styled-divider" style="margin: 1.5rem 0;">', unsafe_allow_html=True) # Divider

        # Quick Tips section
        st.markdown("##### **💡 Quick Setup Tips**") # Bolder label
        st.markdown("""
            <div class="tips-box">
                <ul>
                    <li><strong>Lighting:</strong> Ensure bright, even illumination. Avoid backlighting.</li>
                    <li><strong>Visibility:</strong> Keep your full body in frame throughout the exercise.</li>
                    <li><strong>Clothing:</strong> Wear form-fitting clothes that contrast with your background.</li>
                    <li><strong>Space:</strong> Clear the area around you for safe movement.</li>
                    <li><strong>Distance:</strong> Stand about 6-8 feet (2-2.5m) from the camera.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

        # Version info
        st.markdown('<hr class="styled-divider" style="margin: 1.5rem 0;">', unsafe_allow_html=True) # Divider
        st.markdown("""
            <div style="text-align: center; margin-top: 2rem;">
                <p style="font-size: 0.8rem; color: #aaa;">FormFit AI v2.1</p>
            </div>
        """, unsafe_allow_html=True)

    # --- Main Content Area ---
    st.markdown('<h1 class="main-header">FormFit AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="subtitle">Your Personal AI Fitness Trainer</p>', unsafe_allow_html=True)

    # --- Features Section ---
    st.markdown("### Key Features") # Simple section header
    cols = st.columns(4, gap="medium") # Add medium gap between columns

    feature_data = [
        {"icon": "🤖", "title": "AI Form Analysis", "desc": "Get precise, real-time feedback on exercise technique using advanced computer vision."},
        {"icon": "📊", "title": "Smart Rep Counter", "desc": "Automatically count valid reps, focusing on quality movements, not just quantity."},
        {"icon": "💬", "title": "AI Coaching", "desc": "Receive instant, personalized guidance and form correction cues during your workout."},
        {"icon": "🎯", "title": "Auto Exercise ID", "desc": "The AI intelligently identifies the exercise you're performing - no manual selection needed."}
    ]

    for i, feature in enumerate(feature_data):
        with cols[i]:
            st.markdown(f"""
                <div class="feature-card">
                    <div class="feature-icon">{feature['icon']}</div>
                    <div class="feature-title">{feature['title']}</div>
                    <div class="feature-description">{feature['desc']}</div>
                </div>
            """, unsafe_allow_html=True)

    st.markdown('<hr class="styled-divider">', unsafe_allow_html=True)

    # --- Mode-Specific Content ---
    if options == 'Chatbot':
        st.markdown('<div class="mode-header">💬 AI Form Coach & Fitness Advisor</div>', unsafe_allow_html=True)
        st.info("""
            **Chat with your AI Fitness Assistant!** Ask about:
            *   Proper exercise form and technique variations.
            *   Personalized workout routines and progression.
            *   Injury prevention strategies and recovery tips.
            *   General fitness and nutrition advice.
        """)
        chat_ui() # Launch the chatbot UI

    elif options == 'Auto Classify':
        st.markdown('<div class="mode-header">🎥 Real-Time Exercise Analysis</div>', unsafe_allow_html=True)

        # Layout for instructions and start button
        col1, col2 = st.columns([3, 2], gap="large") # Adjust column ratio and gap

        with col1:
            st.markdown("""
                <div class="instruction-box">
                    <h3>How It Works:</h3>
                    <ol>
                        <li><strong>Prepare:</strong> Position yourself clearly in your camera's view (check tips in sidebar!).</li>
                        <li><strong>Start Exercise:</strong> Begin performing one of the supported exercises.</li>
                        <li><strong>AI Detection:</strong> The system will automatically identify your exercise.</li>
                        <li><strong>Get Feedback:</strong> Receive real-time form analysis, rep counts, and coaching cues directly on screen.</li>
                    </ol>
                    <h4>Supported Exercises:</h4>
                     <ul>
                        <li>Push-ups</li>
                        <li>Squats</li>
                        <li>Bicep Curls</li>
                        <li>Shoulder Press</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)

        with col2:
             st.markdown("""
                <div class="cta-box">
                    <p>Ready to perfect your form?</p>
                </div>
            """, unsafe_allow_html=True)
             # Place button inside the styled box using columns structure
             start_button = st.button('Start AI Analysis Session 🚀', use_container_width=True, key="start_analysis")


        # Logic for starting analysis
        if start_button:
            st.markdown('<hr class="styled-divider" style="margin: 1.5rem 0;">', unsafe_allow_html=True)
            with st.spinner('Initializing AI Analysis... Get ready to move!'):
                time.sleep(1.5) # Slightly longer perceived loading time
                try:
                    exer = exercise.Exercise() # Create Exercise class instance
                    # This function likely contains the video loop and Streamlit element updates
                    exer.auto_classify_and_count()
                except Exception as e:
                    st.error(f"An error occurred during analysis setup: {e}")
                    st.warning("Please ensure your camera is connected and permissions are granted.")

    # --- Footer ---
    st.markdown("""
        <div class="footer">
             Made with <span style="color: #FF6B6B;">♥</span> by Shubh Jain
        </div>
        """, unsafe_allow_html=True)


if __name__ == '__main__':
    main()

#--- END OF FILE main.py ---
