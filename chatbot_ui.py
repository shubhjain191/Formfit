import pandas as pd
import cohere
import os
import time
import logging
from dotenv import load_dotenv
import streamlit as st

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load environment variables
load_dotenv()
cohere_api_key = os.environ.get("COHERE_API_KEY")

# Add debug logging for API key
if not cohere_api_key:
    st.error("❌ Cohere API key not found! Please check your .env file.")
    
try:
    co = cohere.Client(cohere_api_key)
except Exception as e:
    st.error(f"❌ Error initializing Cohere client: {str(e)}")

# --- Dataset Loading ---
def load_exercise_data(csv_file):
    try:
        df = pd.read_csv(csv_file)
        return df
    except FileNotFoundError:
        st.error("📁 Dataset file not found. Please check the file path.")
        return pd.DataFrame()
    except Exception as e:
        st.error(f"📁 Error loading dataset: {str(e)}")
        return pd.DataFrame()

# Load the dataset
exercise_data = load_exercise_data('megaGymDataset.csv')

# --- Need Follow-Up Logic ---
def need_follow_up(query):
    # Simple follow-up detection based on keywords
    if "more" in query.lower() or "details" in query.lower():
        return "It seems like you want more details. Could you specify what exactly you'd like to know?"
    return None

# --- Process User Query ---
def process_query(query, exercise_data, user_preferences):
    try:
        # Debug logging using logging module
        logging.debug(f"Processing query: {query}")
        logging.debug(f"User preferences: {user_preferences}")
        
        # First, check if it's a follow-up question needed
        follow_up = need_follow_up(query)
        if follow_up:
            return {"type": "follow_up", "message": follow_up}
        
        if "describe" in query.lower():
            exercise_name = extract_exercise_name(query)
            description = describe_exercise(exercise_name, exercise_data)
            return {"type": "description", "message": description if description else f"Sorry, I couldn't find details about '{exercise_name}'."}
        else:
            response = generate_response(query, user_preferences)
            return {"type": "general", "message": response}
    except Exception as e:
        error_msg = f"❌ Error processing query: {str(e)}"
        st.error(error_msg)
        return {"type": "error", "message": error_msg}

def generate_response(query, user_preferences):
    try:
        prompt = craft_fitness_prompt(query, user_preferences)
        
        # Debug logging
        logging.debug(f"Generated prompt: {prompt}")
        
        response = co.generate(
            model='command-nightly',
            prompt=prompt,
            max_tokens=500,
            temperature=0.7,
            stop_sequences=["--"]
        )
        
        # Debug logging
        logging.debug(f"Raw Cohere response: {response}")
        
        if response and hasattr(response, 'generations') and response.generations:
            return response.generations[0].text.strip()
        else:
            return "I apologize, but I couldn't generate a response. Please try rephrasing your question."
            
    except Exception as e:
        error_msg = f"⚠️ Error generating response: {str(e)}"
        st.error(error_msg)
        return error_msg

def craft_fitness_prompt(query, user_preferences):
    prompt = (
        f"You are a knowledgeable and friendly fitness expert chatbot. "
        f"Provide a detailed and helpful response to the following question, "
        f"taking into account the user's preferences:\n\n"
        f"User Profile:\n"
        f"- Goal: {user_preferences['goal']}\n"
        f"- Experience Level: {user_preferences['experience']}\n"
        f"- Available Time: {user_preferences['available_time']}\n"
        f"- Workout Frequency: {user_preferences['workout_frequency']}\n"
        f"- Equipment Access: {', '.join(user_preferences['equipment_access'])}\n"
        f"- Restrictions: {user_preferences['restrictions']}\n\n"
        f"User Question: {query}\n\n"
        f"Please provide a detailed, helpful, and encouraging response with relevant fitness advice. "
        f"Include specific recommendations when appropriate."
    )
    return prompt

# --- Streamlit UI ---
def chat_ui():
    st.title("🏋️‍♂️ Fitness Knowledge Bot")
    
    # Add an introduction
    st.markdown("""
    ### Welcome to your personal fitness assistant! 🌟
    I'm here to help you achieve your fitness goals with personalized advice and workout recommendations.
    Start by setting your preferences in the sidebar! 👈
    """)
    
    # Gather user preferences
    user_preferences = gather_user_preferences()
    
    # Show current profile summary
    with st.expander("👤 Your Current Profile"):
        st.write(f"**Goal**: {user_preferences['goal']} 🎯")
        st.write(f"**Experience**: {user_preferences['experience']} 💪")
        st.write(f"**Workout Time**: {user_preferences['available_time']} ⏰")
        st.write(f"**Frequency**: {user_preferences['workout_frequency']} days/week 📅")
        st.write(f"**Equipment**: {', '.join(user_preferences['equipment_access'])} 🏋️‍♂️")
    
    # User query input with placeholder
    user_input = st.text_input(
        "Ask me about workouts, exercises, or fitness tips! 💭",
        placeholder="E.g., 'Create a workout plan' or 'Describe squats'"
    )
    
    # Initialize chat history
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []

    # Create tabs for current conversation and history
    tab_conversation, tab_history = st.tabs(["💬 Current Conversation", "📜 Chat History"])

    with tab_conversation:
        if st.button("Submit 🚀"):
            if not user_input:
                st.warning("⚠️ Please enter a question or query.")
            else:
                with st.spinner("Thinking... 🤔"):
                    # Process the query and get response
                    response = process_query(user_input, exercise_data, user_preferences)
                    
                    # Add to chat history with timestamp
                    timestamp = int(time.time() * 1000)
                    st.session_state.chat_history.append({
                        "user": user_input,
                        "bot": response,
                        "timestamp": timestamp
                    })
                
                # Display the bot's response
                if response["type"] == "error":
                    st.error(response["message"])
                elif response["type"] == "follow_up":
                    st.write("**Bot**: " + response["message"])
                else:
                    st.write("**Bot**: " + response["message"])
                st.markdown("---")

        # Add helpful tips
        with st.expander("💡 Tips for better responses"):
            st.markdown("""
            - Be specific about exercises you want to learn
            - Mention any equipment you plan to use
            - Include your fitness level when asking for routines
            - Ask for alternatives if an exercise doesn't suit you
            """)

    # In the Chat History tab
    with tab_history:
        st.write("### Chat History 📜")
        if st.session_state.chat_history:
            for chat in st.session_state.chat_history:
                st.write("**You**: " + chat["user"])
                if chat["bot"]["type"] == "error":
                    st.error(chat["bot"]["message"])
                elif chat["bot"]["type"] == "follow_up":
                    st.write("**Bot**: " + chat["bot"]["message"])
                else:
                    st.write("**Bot**: " + chat["bot"]["message"])
                st.markdown("---")
        else:
            st.write("No conversation history yet. Start chatting!")

# --- User Preferences ---
def gather_user_preferences():
    goal = st.selectbox("What's your main fitness goal?", 
                        ["Weight Loss", "Build Muscle", "Endurance", "General Fitness"])
    experience = st.radio("What's your experience level?",
                          ["Beginner", "Intermediate", "Advanced"])
    available_time = st.selectbox("How much time can you dedicate to each workout?", 
                                  ["< 30 minutes", "30-45 minutes", "45-60 minutes", "60+ minutes"])
    workout_frequency = st.slider("How many days a week do you plan to workout?", 
                                  min_value=1, max_value=7, value=3)
    equipment_access = st.multiselect("What equipment do you have access to?", 
                                      ["Dumbbells", "Barbell", "Kettlebells", "Resistance Bands", "Bodyweight"])
    restrictions = st.text_input("Do you have any injury or limitation? (Optional)")

    return {
        "goal": goal,
        "experience": experience,
        "available_time": available_time,
        "workout_frequency": workout_frequency,
        "equipment_access": equipment_access,
        "restrictions": restrictions
    }

# Entry point for Streamlit
if __name__ == "__main__":
    st.set_page_config(
        page_title="Fitness Knowledge Bot",
        page_icon="🏋️‍♂️",
        layout="wide"
    )
    chat_ui()
