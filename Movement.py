import asyncio
import websockets
import json
import random
import math
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import torch.nn as nn
import sys
import logging
from collections import deque
import tkinter as tk
from threading import Thread

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
API_URL = "ws://127.0.0.1:8001"
TOKEN_FILE = "auth_token.txt"
INITIAL_RECONNECT_DELAY = 1.0
MAX_RECONNECT_DELAY = 30.0
UPDATE_INTERVAL = 0.02  # ~50 FPS
PING_INTERVAL = 10
PING_TIMEOUT = 25
TARGET_UPDATE_INTERVAL = 0.20
KEEP_ALIVE_INTERVAL = 5.0
DEBUG_MODE = True

PARAM_RANGES = {
    "HeadX": (-60, 60), "HeadY": (-70, 70), "BodyTilt": (-50, 50),
    "BodySwing": (-55, 55), "Step": (-70, 70), "FaceLean": (-40, 40),
    "BodyLean": (-50, 50), "EyesX": (-3.5, 3.5), "EyesY": (-3.5, 3.5),
    "ArmLeftX": (-70, 70), "ArmRightX": (-70, 70),
    "EyebrowLeft": (-35, 35), "EyebrowRight": (-35, 35),
    "BrowAngleLeft": (-70, 70), "BrowAngleRight": (-70, 70),
    "HandLeftX": (-70, 70), "HandRightX": (-70, 70),
    "ShouldersY": (-60, 60), "PositionX": (-70, 70), "PositionY": (-80, 80),
    "MoveSpeed": (0.08, 0.25),
    "Smile": (0, 1),
    "Eyelids": (0, 1)
}

MODEL_PATH = "./neuro_sama_model"
MODEL_FILE = os.path.join(MODEL_PATH, "model.pt")

# Emotion modifiers with slower speeds
emotion_modifiers = {
    "playful": {"amplitude": 6.5, "speed": 1.8, "body_boost": 4.5, "overshoot": 0.6, "osc_multiplier": 4.5},
    "happy": {"amplitude": 4.5, "speed": 1.3, "body_boost": 3.0, "overshoot": 0.5, "osc_multiplier": 2.8},
    "sad": {"amplitude": 2.0, "speed": 0.15, "body_boost": 1.0, "overshoot": 0.1, "osc_multiplier": 0.1},
    "eh": {"amplitude": 1.0, "speed": 0.4, "body_boost": 0.8, "overshoot": 0.1, "osc_multiplier": 0.5},
    "surprised": {"amplitude": 3.5, "speed": 1.5, "body_boost": 2.5, "overshoot": 0.4, "osc_multiplier": 1.5}
}

emotion_keywords = {
    "playful": ["dance", "dancing", "boogie", "groove", "vibe", "playful"],
    "surprised": ["surprised", "shocked", "wow", "oh"],
    "happy": ["happy", "joy", "excited", "gleeful"],
    "sad": ["sad", "down", "blue", "depressed"],
    "eh": ["eh", "meh", "whatever", "indifferent"]
}

# Global state for emotion selection
current_emotion = "happy"

def clamp(value, min_val, max_val):
    return max(min_val, min(value, max_val))

def log(message, silent=False):
    if not silent:
        logger.info(message)

def quintic_ease(t):
    t = clamp(t, 0, 1)
    return t * t * t * (t * (t * 6 - 15) + 10)

def interpolate(current, target, velocity, param, speed_multiplier=1.0, state=None):
    try:
        min_val, max_val = PARAM_RANGES[param]
        emotion = state.get("current_emotion", "happy") if state else "happy"
        overshoot = emotion_modifiers[emotion]["overshoot"]
        osc_multiplier = emotion_modifiers[emotion]["osc_multiplier"]
        if param in ["HeadX", "HeadY"]:
            k = 0.20 if emotion == "playful" else 0.15
            d = max(0.75 - overshoot, 0.5) if emotion == "playful" else 0.90
        elif param in ["PositionX", "PositionY"]:
            k = 0.10
            d = 0.95
        elif param in ["BodyTilt", "BodySwing", "BodyLean", "ShouldersY"]:
            k = 0.18 if emotion == "playful" else 0.15
            d = max(0.85 - overshoot, 0.6) if emotion == "playful" else max(0.80 - overshoot, 0.6)
        elif param in ["ArmLeftX", "ArmRightX", "HandLeftX", "HandRightX"]:
            k = 0.18
            d = 0.90
        elif param in ["EyesX", "EyesY", "EyebrowLeft", "EyebrowRight", "Smile", "Eyelids"]:
            k = 0.35 if emotion == "playful" else 0.30
            d = 0.88
        elif param in ["BrowAngleLeft", "BrowAngleRight"]:
            k = 0.35 if emotion == "playful" else 0.30
            d = 0.88
        else:
            k = 0.20
            d = 0.90
        k *= speed_multiplier
        velocity_magnitude = abs(velocity) / (0.15 * (max_val - min_val))
        overshoot_factor = 1.0 + overshoot * velocity_magnitude
        accel = k * (target - current) * overshoot_factor - d * velocity
        velocity = clamp(velocity + accel * UPDATE_INTERVAL, -0.15 * (max_val - min_val), 0.15 * (max_val - min_val))
        new_current = current + velocity * UPDATE_INTERVAL
        if random.random() < 0.10 and emotion == "playful" and param in ["HeadX", "HeadY", "BodyTilt", "BodySwing", "BodyLean", "ShouldersY"]:
            jitter = random.uniform(-0.15 * (max_val - min_val), 0.15 * (max_val - min_val))
            new_current += jitter
            new_current = clamp(new_current, min_val, max_val)
        velocity *= 0.96
        return clamp(new_current, min_val, max_val), velocity
    except Exception as e:
        log(f"Error in interpolate for {param}: {e}")
        return current, velocity

def load_token():
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE, "r") as f:
            token = f.read().strip()
            log(f"Loaded token: {token[:10]}...")
            return token
    return None

def save_token(token):
    with open(TOKEN_FILE, "w") as f:
        f.write(token)
    log("Saved new token.")

MOCK_CONVERSATION = [
    "Assistant: I'm so happy, let's bounce around! 😄 (happy)",
    "Assistant: Feeling a bit down today... 😔 (sad)",
    "Assistant: I want to dance right now! 🕺 (playful)",
    "Assistant: Oh my gosh, I’m so surprised! 😲 (surprised)",
    "Assistant: Eh, whatever... 😐 (eh)"
]

def load_conversation_history_fast():
    if not os.path.exists("conversation_history.json"):
        log("No conversation history found, using mock data.")
        return MOCK_CONVERSATION
    try:
        with open("conversation_history.json", "r") as f:
            lines = f.read().splitlines()[-3:]
            return lines if lines else MOCK_CONVERSATION
    except Exception as e:
        log(f"Error reading conversation history: {e}")
        return MOCK_CONVERSATION

def get_latest_ai_message_fast(history, debug_input=None):
    if DEBUG_MODE and debug_input:
        return debug_input
    for line in reversed(history):
        if "Assistant:" in line or "[Chrissy]" in line:
            return line.split(":", 1)[1].strip() if ":" in line else line
    return random.choice(MOCK_CONVERSATION).split(":", 1)[1].strip()

def get_current_movement():
    try:
        with open("body_movement.json", "r") as f:
            data = json.load(f)
            return data.get("movement", "")
    except Exception as e:
        logger.error(f"Error reading body_movement.json: {e}")
        return ""

class CustomDistilBert(nn.Module):
    def __init__(self, num_params=len(PARAM_RANGES)):
        super().__init__()
        self.distilbert = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
        self.regressor = nn.Sequential(
            nn.Linear(self.distilbert.config.hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_params)
        )

    def forward(self, input_ids, attention_mask=None):
        outputs = self.distilbert.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0, :]
        logits = self.regressor(hidden_state)
        return logits

def load_model_and_tokenizer():
    if not os.path.exists(MODEL_FILE):
        log(f"Model file {MODEL_FILE} not found. Please run train.py first.")
        sys.exit(1)
    tokenizer = DistilBertTokenizer.from_pretrained(MODEL_PATH)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomDistilBert().to(device)
    try:
        state_dict = torch.load(MODEL_FILE, map_location=device, weights_only=True)
        model.load_state_dict(state_dict)
        log("Loaded trained model.")
    except Exception as e:
        log(f"Model loading failed: {e}. Using untrained model with fallback logic.")
    model.eval()
    return tokenizer, model, device

def detect_emotion(prompt):
    global current_emotion
    try:
        if current_emotion != "default":
            return current_emotion
        prompt_lower = prompt.lower()
        for emotion, keywords in emotion_keywords.items():
            if any(keyword in prompt_lower for keyword in keywords):
                return emotion
        if "(" in prompt and ")" in prompt:
            start = prompt.rfind("(")
            end = prompt.rfind(")")
            if start < end:
                mood = prompt[start+1:end].strip().lower()
                if mood in emotion_modifiers:
                    return mood
        return "happy"
    except Exception as e:
        log(f"Error in detect_emotion: {e}")
        return "happy"

def generate_targets(tokenizer, model, device, prompt, state, prev_values=None, target_buffer=None):
    try:
        encoding = tokenizer(
            prompt,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].to(device)
        attention_mask = encoding["attention_mask"].to(device)
        
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            values = logits.squeeze().cpu().numpy()
        
        param_list = list(PARAM_RANGES.keys())
        if len(values) != len(param_list):
            log(f"Model output length mismatch: {len(values)}, expected {len(param_list)}. Using fallback.")
            values = [random.uniform(*PARAM_RANGES[param]) for param in param_list]
        
        for i, param in enumerate(param_list):
            min_val, max_val = PARAM_RANGES[param]
            range_scale = 0.90
            scaled_min = min_val + (max_val - min_val) * (1 - range_scale) / 2
            scaled_max = max_val - (max_val - min_val) * (1 - range_scale) / 2
            values[i] = clamp(values[i], scaled_min, scaled_max)
        
        # Read and apply movement from JSON
        movement_desc = get_current_movement().lower()
        if "waving" in movement_desc:
            for param in ["ArmLeftX", "ArmRightX", "HandLeftX", "HandRightX"]:
                i = param_list.index(param)
                values[i] = random.uniform(-50, 50)  # Wave arms
        elif "nodding" in movement_desc:
            i = param_list.index("HeadY")
            values[i] = random.uniform(-30, 30)  # Nod head
        elif "dancing" in movement_desc:
            for param in ["BodySwing", "Step", "ArmLeftX", "ArmRightX"]:
                i = param_list.index(param)
                values[i] = random.uniform(-40, 40)  # Dance movements
        elif "shrugging" in movement_desc:
            i = param_list.index("ShouldersY")
            values[i] = random.uniform(20, 40)  # Shrug shoulders
        
        if target_buffer is not None and prev_values is not None:
            for i, param in enumerate(param_list):
                target_buffer[param].append(values[i])
                avg_value = sum(target_buffer[param]) / len(target_buffer[param])
                values[i] = 0.8 * avg_value + 0.2 * prev_values[i]
        
        emotion = detect_emotion(prompt)
        if state.get("current_emotion") != emotion or state.get("force_target_update", False):
            state["prev_emotion"] = state.get("current_emotion", "happy")
            state["prev_speed"] = state.get("speed_multiplier", 1.2)
            state["emotion_transition_time"] = 0.0
            state["current_emotion"] = emotion
            state["target_speed"] = emotion_modifiers[emotion]["speed"]
            state["reset_velocities"] = False
            state["transition_buffer"] = prev_values.copy() if prev_values is not None else [0.0] * len(param_list)
            state["force_target_update"] = False
        
        transition_duration = 1.2  # Slower transitions
        if state.get("emotion_transition_time", 0.0) < transition_duration:
            t = state["emotion_transition_time"] / transition_duration
            t = quintic_ease(t)
            prev_amp = emotion_modifiers[state.get("prev_emotion", "happy")]["amplitude"]
            curr_amp = emotion_modifiers[emotion]["amplitude"]
            prev_speed = state.get("prev_speed", 1.2)
            curr_speed = state.get("target_speed", emotion_modifiers[emotion]["speed"])
            prev_body_boost = emotion_modifiers[state.get("prev_emotion", "happy")]["body_boost"]
            curr_body_boost = emotion_modifiers[emotion]["body_boost"]
            amplitude_multiplier = (1 - t) * prev_amp + t * curr_amp
            state["speed_multiplier"] = (1 - t) * prev_speed + t * curr_speed
            state["body_boost"] = (1 - t) * prev_body_boost + t * curr_body_boost
            state["emotion_transition_time"] += UPDATE_INTERVAL
            if state.get("transition_buffer"):
                for i in range(len(values)):
                    values[i] = (1 - t**2) * state["transition_buffer"][i] + (t**2) * values[i]
        else:
            amplitude_multiplier = emotion_modifiers[emotion]["amplitude"]
            state["speed_multiplier"] = emotion_modifiers[emotion]["speed"]
            state["body_boost"] = emotion_modifiers[emotion]["body_boost"]
            state["transition_buffer"] = None
        
        head_x = next((v for i, v in enumerate(values) if param_list[i] == "HeadX"), 0.0)
        head_y = next((v for i, v in enumerate(values) if param_list[i] == "HeadY"), 0.0)
        position_x = next((v for i, v in enumerate(values) if param_list[i] == "PositionX"), 0.0)
        position_y = next((v for i, v in enumerate(values) if param_list[i] == "PositionY"), 0.0)
        for i, param in enumerate(param_list):
            min_val, max_val = PARAM_RANGES[param]
            body_boost = state.get("body_boost", 1.0)
            if param == "BodyTilt":
                values[i] = clamp((head_x * 0.8 + position_x * 0.15) * body_boost, min_val, max_val)
                if emotion == "sad":
                    values[i] = clamp(values[i] - 20, min_val, max_val)
                elif emotion == "surprised":
                    values[i] = clamp(values[i] + 15, min_val, max_val)
            elif param == "BodySwing":
                values[i] = clamp((head_x * 0.85 + position_x * 0.20) * body_boost, min_val, max_val)
                if emotion == "playful":
                    values[i] = clamp(values[i] + 10, min_val, max_val)
            elif param == "BodyLean":
                values[i] = clamp((head_y * 0.8 + position_y * 0.15) * body_boost, min_val, max_val)
                if emotion == "sad":
                    values[i] = clamp(values[i] - 30, min_val, max_val)
                elif emotion == "happy":
                    values[i] = clamp(values[i] + 10, min_val, max_val)
            elif param == "ShouldersY":
                values[i] = clamp((head_y * 0.85 + position_y * 0.20) * body_boost, min_val, max_val)
                if emotion == "sad":
                    values[i] = clamp(values[i] - 25, min_val, max_val)
            elif emotion in ["playful", "happy"] and param in ["BodyTilt", "BodySwing", "BodyLean", "ShouldersY"]:
                values[i] = clamp(values[i] * 1.3 * body_boost, min_val, max_val)
            elif emotion == "sad" and param in ["BodyLean", "ShouldersY"]:
                values[i] = min_val * 0.8
            elif emotion == "eh" and param in ["BodyLean", "ShouldersY"]:
                values[i] = clamp(values[i] * 0.9, min_val * 0.5, max_val * 0.5)
            elif emotion == "surprised" and param in ["BodyTilt", "BodySwing"]:
                values[i] = clamp(values[i] * 0.95, min_val, max_val)
            elif param == "Smile":
                if emotion == "happy":
                    values[i] = clamp(0.95 + random.uniform(-0.01, 0.01), 0.94, 0.96)
                elif emotion == "sad":
                    values[i] = clamp(0.02 + random.uniform(-0.01, 0.01), 0.01, 0.03)
                elif emotion == "surprised":
                    values[i] = clamp(0.55 + random.uniform(-0.01, 0.01), 0.54, 0.56)
                elif emotion == "playful":
                    values[i] = clamp(0.92 + random.uniform(-0.01, 0.01), 0.91, 0.93)
                elif emotion == "eh":
                    values[i] = clamp(0.25 + random.uniform(-0.01, 0.01), 0.24, 0.26)
            elif param == "Eyelids":
                if emotion == "surprised":
                    values[i] = clamp(0.99, 0.97, 1.0)
                elif emotion == "sad":
                    values[i] = clamp(0.30, 0.28, 0.32)
                elif emotion == "eh":
                    values[i] = clamp(0.45, 0.43, 0.47)
                elif emotion in ["happy", "playful"]:
                    values[i] = clamp(0.85, 0.83, 0.87)
            elif param == "EyebrowLeft":
                if emotion == "happy":
                    values[i] = clamp(30 + random.uniform(-3, 3), 29, 31)
                elif emotion == "sad":
                    values[i] = clamp(-30 + random.uniform(-3, 3), -31, -29)
                elif emotion == "playful":
                    values[i] = clamp(25 + random.uniform(-3, 3), 24, 26)
                elif emotion == "eh":
                    values[i] = clamp(0 + random.uniform(-3, 3), -1, 1)
                elif emotion == "surprised":
                    values[i] = clamp(35 + random.uniform(-3, 3), 34, 36)
            elif param == "EyebrowRight":
                if emotion == "happy":
                    values[i] = clamp(30 + random.uniform(-3, 3), 29, 31)
                elif emotion == "sad":
                    values[i] = clamp(-30 + random.uniform(-3, 3), -31, -29)
                elif emotion == "playful":
                    values[i] = clamp(20 + random.uniform(-3, 3), 19, 21)
                elif emotion == "eh":
                    values[i] = clamp(0 + random.uniform(-3, 3), -1, 1)
                elif emotion == "surprised":
                    values[i] = clamp(35 + random.uniform(-3, 3), 34, 36)
            elif param == "BrowAngleLeft":
                if emotion == "happy":
                    values[i] = clamp(25 + random.uniform(-3, 3), 24, 26)
                elif emotion == "sad":
                    values[i] = clamp(35 + random.uniform(-3, 3), 34, 36)
                elif emotion == "playful":
                    values[i] = clamp(30 + random.uniform(-3, 3), 29, 31)
                elif emotion == "eh":
                    values[i] = clamp(5 + random.uniform(-3, 3), 4, 6)
                elif emotion == "surprised":
                    values[i] = clamp(30 + random.uniform(-3, 3), 29, 31)
            elif param == "BrowAngleRight":
                if emotion == "happy":
                    values[i] = clamp(25 + random.uniform(-3, 3), 24, 26)
                elif emotion == "sad":
                    values[i] = clamp(-35 + random.uniform(-3, 3), -36, -34)
                elif emotion == "playful":
                    values[i] = clamp(-20 + random.uniform(-3, 3), -21, -19)
                elif emotion == "eh":
                    values[i] = clamp(5 + random.uniform(-3, 3), 4, 6)
                elif emotion == "surprised":
                    values[i] = clamp(30 + random.uniform(-3, 3), 29, 31)
            if emotion == "playful" and param in ["Smile", "EyebrowLeft", "EyebrowRight", "BrowAngleLeft", "BrowAngleRight"] and random.random() < 0.05:
                values[i] += random.uniform(-0.05 * (max_val - min_val), 0.05 * (max_val - min_val))
            values[i] = clamp(values[i], min_val, max_val)
            if param in ["HeadX", "HeadY", "BodyTilt", "BodySwing", "BodyLean", "ShouldersY", "Smile", "EyebrowLeft", "EyebrowRight", "BrowAngleLeft", "BrowAngleRight"]:
                log(f"Target {param}: {values[i]:.2f}", silent=False)
        
        return values
    except Exception as e:
        log(f"Error in generate_targets: {e}")
        return [random.uniform(*PARAM_RANGES[param]) for param in PARAM_RANGES] if prev_values is None else prev_values

sway_freq = 0.07  # Slower sway
bob_freq = 0.10  # Slower bob
sway_phase = 0.0
bob_phase = math.pi / 2

sway_amplitudes = {
    "PositionX": 5.0, "HeadX": 60.0, "BodySwing": 80.0,
    "ArmLeftX": -75.0, "ArmRightX": 75.0, "HandLeftX": -75.0,
    "HandRightX": 75.0, "EyesX": 3.0
}

bob_amplitudes = {
    "PositionY": 7.0, "HeadY": 55.0, "ShouldersY": 80.0,
    "BodyTilt": 80.0,
}

amplitudes = {
    "HeadX": 60.0, "HeadY": 60.0, "BodyTilt": 80.0,
    "BodySwing": 80.0, "Step": 85.0, "FaceLean": 40.0, "BodyLean": 80.0,
    "EyesX": 3.5, "EyesY": 3.5, "ArmLeftX": 80.0, "ArmRightX": 80.0,
    "EyebrowLeft": 5.0, "EyebrowRight": 5.0, "BrowAngleLeft": 5.0,
    "BrowAngleRight": 5.0, "HandLeftX": 80.0, "HandRightX": 80.0,
    "ShouldersY": 80.0, "PositionX": 5.0, "PositionY": 7.0,
    "MoveSpeed": 0.0, "Smile": 0.05, "Eyelids": 0.12
}

frequencies = {
    "HeadX": [0.07, 0.035, 0.007], "HeadY": [0.07, 0.035, 0.007],
    "BodyTilt": [0.07, 0.035, 0.007], "BodySwing": [0.07, 0.035, 0.007],
    "Step": [0.06, 0.03, 0.006], "FaceLean": [0.04, 0.02, 0.005],
    "BodyLean": [0.07, 0.035, 0.007],
    "EyesX": [0.08, 0.04, 0.008], "EyesY": [0.08, 0.04, 0.008],
    "ArmLeftX": [0.04, 0.02, 0.005], "ArmRightX": [0.04, 0.02, 0.005],
    "EyebrowLeft": [0.03, 0.015, 0.004], "EyebrowRight": [0.03, 0.015, 0.004],
    "BrowAngleLeft": [0.03, 0.015, 0.004], "BrowAngleRight": [0.03, 0.015, 0.004],
    "HandLeftX": [0.04, 0.02, 0.005], "HandRightX": [0.04, 0.02, 0.005],
    "ShouldersY": [0.07, 0.035, 0.007], "PositionX": [0.035, 0.018, 0.004],
    "PositionY": [0.035, 0.018, 0.004], "MoveSpeed": [0.0, 0.0, 0.0],
    "Smile": [0.03, 0.015, 0.004], "Eyelids": [0.04, 0.02, 0.005]
}

# Tkinter GUI
def create_emotion_window(target_base_values, state):
    global current_emotion
    window = tk.Tk()
    window.title("Emotion Control")
    window.geometry("300x200")
    
    def set_emotion(emotion):
        global current_emotion
        current_emotion = emotion
        state["emotion_transition_time"] = 0.0
        state["transition_buffer"] = None
        state["force_target_update"] = True
        log(f"Emotion set to: {emotion}")
    
    tk.Button(window, text="Sad", command=lambda: set_emotion("sad"), width=10).pack(pady=10)
    tk.Button(window, text="Happy", command=lambda: set_emotion("happy"), width=10).pack(pady=10)
    tk.Button(window, text="Playful", command=lambda: set_emotion("playful"), width=10).pack(pady=10)
    tk.Button(window, text="Eh", command=lambda: set_emotion("eh"), width=10).pack(pady=10)
    tk.Button(window, text="Surprised", command=lambda: set_emotion("surprised"), width=10).pack(pady=10)
    
    window.mainloop()

def run_gui(target_base_values, state):
    create_emotion_window(target_base_values, state)

async def debug_input():
    loop = asyncio.get_event_loop()
    while True:
        try:
            print("\nEnter AI message (e.g., 'I want to dance! (playful)') or 'quit' to exit:")
            user_input = await loop.run_in_executor(None, sys.stdin.readline)
            user_input = user_input.strip()
            if user_input.lower() == 'quit':
                return None
            if user_input:
                return user_input
            print("Empty input, try again.")
        except Exception as e:
            log(f"Error in debug_input: {e}")
            return None
        await asyncio.sleep(0.1)

async def target_generator(target_base_values, current_values, tokenizer, model, device, state):
    param_list = list(PARAM_RANGES.keys())
    prev_values = None
    target_buffer = {param: deque(maxlen=15) for param in param_list}
    target_smoothing_factor = 0.5
    while True:
        try:
            if DEBUG_MODE:
                debug_message = await debug_input()
                if debug_message is None:
                    log("Exiting debug mode.")
                    break
                latest = debug_message
            else:
                history = load_conversation_history_fast()
                latest = get_latest_ai_message_fast(history)
            
            log(f"Processing message: {latest}", silent=DEBUG_MODE)
            values = generate_targets(tokenizer, model, device, latest, state, prev_values, target_buffer)
            if prev_values is not None:
                for i, param in enumerate(param_list):
                    values[i] = target_smoothing_factor * prev_values[i] + (1 - target_smoothing_factor) * values[i]
            prev_values = values.copy()
            for i, param in enumerate(param_list):
                min_val, max_val = PARAM_RANGES[param]
                delta = (max_val - min_val) * 0.05
                target = clamp(values[i], current_values[param] - delta, current_values[param] + delta)
                target_base_values[param] = clamp(target, min_val, max_val)
        except Exception as e:
            log(f"Error in target_generator: {e}")
            for param in param_list:
                min_val, max_val = PARAM_RANGES[param]
                delta = (max_val - min_val) * 0.05
                target_base_values[param] = clamp(
                    current_values[param] + random.uniform(-delta, delta), min_val, max_val)
        await asyncio.sleep(TARGET_UPDATE_INTERVAL)

def parameter_generator(base_values, target_base_values, state):
    time = 0.0
    phases = {param: [random.uniform(0, 2 * math.pi) for _ in range(3)] for param in PARAM_RANGES}
    base_frequencies = frequencies.copy()
    body_delay = math.pi / 2
    arm_delays = {param: 0.10 for param in ["ArmLeftX", "ArmRightX", "HandLeftX", "HandRightX"]}
    for param in ["ArmLeftX", "ArmRightX", "HandLeftX", "HandRightX", "BodySwing"]:
        for i in range(3):
            phases[param][i] += body_delay
    velocities = {param: 0.0 for param in PARAM_RANGES}
    ema_values = {param: base_values[param] for param in PARAM_RANGES}
    prev_values = {param: base_values[param] for param in PARAM_RANGES}
    blink_timer = 0.0
    blink_interval = random.uniform(0.4, 1.0)
    body_delay_timers = {param: 0.0 for param in ["BodyTilt", "BodySwing", "BodyLean", "ShouldersY"]}
    while True:
        try:
            sway = math.sin(2 * math.pi * sway_freq * time + sway_phase)
            bob = math.sin(2 * math.pi * bob_freq * time + bob_phase)
            sway_amp_mod = 1 + 0.20 * math.sin(2 * math.pi * 0.006 * time)
            bob_amp_mod = 1 + 0.20 * math.sin(2 * math.pi * 0.008 * time)
            speed_variation = 1 + 0.06 * math.sin(2 * math.pi * 0.004 * time)
            values = {}
            emotion = state.get("current_emotion", "happy")
            speed_multiplier = state.get("speed_multiplier", 1.2) * speed_variation
            amplitude_multiplier = emotion_modifiers[emotion]["amplitude"]
            body_boost = state.get("body_boost", 1.0)
            osc_multiplier = emotion_modifiers[emotion]["osc_multiplier"]
            ema_weight = 0.98 if param in ["Smile", "EyebrowLeft", "EyebrowRight", "BrowAngleLeft", "BrowAngleRight"] else 0.95 if emotion == "playful" else 0.94
            
            blink_timer += UPDATE_INTERVAL
            if blink_timer >= blink_interval:
                if emotion not in ["sad", "eh"]:
                    values["Eyelids"] = 0.04
                blink_timer = 0.0
                blink_interval = random.uniform(0.4, 1.0)
            else:
                values["Eyelids"] = base_values.get("Eyelids", 0.82)
            
            for param in phases:
                if random.random() < 0.10:
                    for i in range(3):
                        phases[param][i] += random.uniform(-0.10, 0.10)
                    base_freq = base_frequencies[param]
                    frequencies[param] = [f * random.uniform(0.96, 1.04) for f in base_freq]
            
            head_x_target = base_values.get("HeadX", 0.0)
            head_y_target = base_values.get("HeadY", 0.0)
            position_x_target = base_values.get("PositionX", 0.0)
            position_y_target = base_values.get("PositionY", 0.0)
            arm_left_target = -(head_x_target + position_x_target * 0.8) * 0.95
            arm_right_target = (head_x_target + position_x_target * 0.8) * 0.95
            hand_left_target = -(head_x_target + position_x_target * 0.8) * 0.90
            hand_right_target = (head_x_target + position_x_target * 0.8) * 0.90
            body_swing_target = (head_x_target * 0.85 + position_x_target * 0.20) * 0.95 * body_boost
            body_tilt_target = (head_x_target * 0.8 + position_x_target * 0.20) * 0.90 * body_boost
            body_lean_target = (head_y_target * 0.8 + position_y_target * 0.20) * 0.90 * body_boost
            shoulders_y_target = (head_y_target * 0.85 + position_y_target * 0.20) * 0.95 * body_boost
            
            for param in body_delay_timers:
                body_delay_timers[param] = max(0, body_delay_timers[param] - UPDATE_INTERVAL)
                if body_delay_timers[param] == 0 and random.random() < 0.10:
                    body_delay_timers[param] = random.uniform(0.15, 0.30) if emotion == "playful" else random.uniform(0.08, 0.20)
            
            for param in PARAM_RANGES:
                min_val, max_val = PARAM_RANGES[param]
                if param != "MoveSpeed" and param != "Eyelids":
                    arm_delay = arm_delays.get(param, 0.0)
                    body_delay = body_delay_timers.get(param, 0.0)
                    if time < arm_delay and param in arm_delays:
                        target = base_values[param]
                    elif param == "ArmLeftX":
                        target = clamp(0.30 * target_base_values[param] + 0.70 * arm_left_target, min_val, max_val)
                    elif param == "ArmRightX":
                        target = clamp(0.30 * target_base_values[param] + 0.70 * arm_right_target, min_val, max_val)
                    elif param == "HandLeftX":
                        target = clamp(0.30 * target_base_values[param] + 0.70 * hand_left_target, min_val, max_val)
                    elif param == "HandRightX":
                        target = clamp(0.30 * target_base_values[param] + 0.70 * hand_right_target, min_val, max_val)
                    elif param == "BodySwing" and body_delay == 0:
                        target = clamp(0.30 * target_base_values[param] + 0.70 * body_swing_target, min_val, max_val)
                    elif param == "BodyTilt" and body_delay == 0:
                        target = clamp(0.30 * target_base_values[param] + 0.70 * body_tilt_target, min_val, max_val)
                    elif param == "BodyLean" and body_delay == 0:
                        target = clamp(0.30 * target_base_values[param] + 0.70 * body_lean_target, min_val, max_val)
                    elif param == "ShouldersY" and body_delay == 0:
                        target = clamp(0.30 * target_base_values[param] + 0.70 * shoulders_y_target, min_val, max_val)
                    else:
                        target = base_values[param] if body_delay > 0 else target_base_values[param]
                    base_values[param], velocities[param] = interpolate(
                        base_values[param], target, velocities[param], param, speed_multiplier, state
                    )
                    amp = amplitudes.get(param, 0.0) * amplitude_multiplier * 0.65
                    freq1, freq2, freq3 = frequencies.get(param, [0.07, 0.035, 0.007])
                    osc = amp * (
                        0.5 * math.sin(2 * math.pi * freq1 * time + phases[param][0]) +
                        0.3 * math.sin(2 * math.pi * freq2 * time + phases[param][1]) +
                        0.2 * math.sin(2 * math.pi * freq3 * time + phases[param][2])
                    ) * osc_multiplier
                    sway_offset = sway * sway_amplitudes.get(param, 0.0) * sway_amp_mod
                    bob_offset = bob * bob_amplitudes.get(param, 0.0) * bob_amp_mod
                    eyebrow_twitch = math.sin(2 * math.pi * 0.3 * time) * 3.0 if param in ["EyebrowLeft", "EyebrowRight"] and emotion == "playful" else 0.0
                    value = base_values[param] + osc + sway_offset + bob_offset + eyebrow_twitch
                    ema_values[param] = ema_weight * ema_values[param] + (1 - ema_weight) * value
                    values[param] = clamp(ema_values[param], min_val, max_val)
                    if param in ["BodyTilt", "BodySwing", "BodyLean", "ShouldersY"]:
                        max_delta = (max_val - min_val) * 0.10
                        values[param] = clamp(values[param], prev_values[param] - max_delta, prev_values[param] + max_delta)
                    if param in ["Smile", "EyebrowLeft", "EyebrowRight", "BrowAngleLeft", "BrowAngleRight"]:
                        max_delta = (max_val - min_val) * 0.05
                        values[param] = clamp(values[param], prev_values[param] - max_delta, prev_values[param] + max_delta)
                    prev_values[param] = values[param]
                    if param in ["HeadX", "HeadY", "BodyTilt", "BodySwing", "BodyLean", "ShouldersY", "Smile", "EyebrowLeft", "EyebrowRight", "BrowAngleLeft", "BrowAngleRight"]:
                        log(f"Generated {param}: {values[param]:.2f}", silent=False)
                elif param == "MoveSpeed":
                    base_values[param], velocities[param] = interpolate(
                        base_values[param], target_base_values[param], velocities[param], param, speed_multiplier, state
                    )
                    values[param] = base_values[param]
                    prev_values[param] = values[param]
            yield values
            time += UPDATE_INTERVAL
        except Exception as e:
            log(f"Error in parameter_generator: {e}")
            continue

async def keep_alive(websocket):
    while True:
        try:
            await websocket.ping()
            await asyncio.sleep(KEEP_ALIVE_INTERVAL)
        except websockets.ConnectionClosed:
            break

async def main():
    base_values = {param: 0.0 for param in PARAM_RANGES}
    target_base_values = {param: 0.0 for param in PARAM_RANGES}
    state = {
        "current_emotion": "happy",
        "speed_multiplier": 1.2,
        "body_boost": 3.0,
        "prev_emotion": "happy",
        "prev_speed": 1.2,
        "target_speed": 1.2,
        "emotion_transition_time": 0.0,
        "reset_velocities": False,
        "transition_buffer": None,
        "force_target_update": False
    }
    
    gui_thread = Thread(target=run_gui, args=(target_base_values, state), daemon=True)
    gui_thread.start()
    
    tokenizer, model, device = load_model_and_tokenizer()
    
    param_gen = parameter_generator(base_values, target_base_values, state)
    token = load_token()
    reconnect_delay = INITIAL_RECONNECT_DELAY
    connection_attempts = 0
    max_attempts = 5
    websocket = None
    frame_buffer = deque(maxlen=4)
    prev_params = None

    while True:
        try:
            async with websockets.connect(
                API_URL,
                ping_interval=PING_INTERVAL,
                ping_timeout=PING_TIMEOUT,
                max_size=10**6
            ) as websocket:
                log("Connected to WebSocket")
                reconnect_delay = INITIAL_RECONNECT_DELAY
                connection_attempts = 0

                if not token:
                    auth_token_request = {
                        "apiName": "VTubeStudioPublicAPI",
                        "apiVersion": "1.0",
                        "requestID": "auth_token",
                        "messageType": "AuthenticationTokenRequest",
                        "data": {"pluginName": "fast_vts", "pluginDeveloper": "Genteki"}
                    }
                    log("Requesting new token...")
                    await websocket.send(json.dumps(auth_token_request))
                    response = json.loads(await websocket.recv())
                    if "data" in response and "authenticationToken" in response["data"]:
                        token = response["data"]["authenticationToken"]
                        save_token(token)
                    else:
                        log(f"Token request failed: {response.get('message', 'Unknown error')}")
                        raise Exception("Token acquisition failed")

                auth_request = {
                    "apiName": "VTubeStudioPublicAPI",
                    "apiVersion": "1.0",
                    "requestID": "auth",
                    "messageType": "AuthenticationRequest",
                    "data": {"pluginName": "fast_vts", "pluginDeveloper": "Genteki", "authenticationToken": token}
                }
                log("Attempting authentication...")
                await websocket.send(json.dumps(auth_request))
                auth_response = json.loads(await websocket.recv())
                if not auth_response.get("data", {}).get("authenticated", False):
                    reason = auth_response.get("data", {}).get("reason", "Unknown reason")
                    log(f"Authentication failed: {reason}")
                    if "token" in reason.lower():
                        token = None
                        if os.path.exists(TOKEN_FILE):
                            os.remove(TOKEN_FILE)
                    raise Exception("Authentication failed")

                log("Authenticated successfully!")

                for param, (min_val, max_val) in PARAM_RANGES.items():
                    if param != "MoveSpeed":
                        create_request = {
                            "apiName": "VTubeStudioPublicAPI",
                            "apiVersion": "1.0",
                            "requestID": f"create_{param}",
                            "messageType": "ParameterCreationRequest",
                            "data": {"parameterName": param, "min": min_val, "max": max_val, "defaultValue": 0}
                        }
                        await websocket.send(json.dumps(create_request))
                        try:
                            response = await asyncio.wait_for(websocket.recv(), timeout=0.002)
                            log(f"Created parameter {param}: {response}", silent=False)
                        except asyncio.TimeoutError:
                            log(f"Timeout receiving response for {param}, continuing...")

                asyncio.create_task(keep_alive(websocket))
                asyncio.create_task(target_generator(target_base_values, base_values, tokenizer, model, device, state))
                log("Started target generator")

                while True:
                    try:
                        params = next(param_gen)
                        if prev_params is not None:
                            for k in params:
                                if k != "MoveSpeed":
                                    params[k] = 0.90 * params[k] + 0.10 * prev_params.get(k, params[k])
                        inject_request = {
                            "apiName": "VTubeStudioPublicAPI",
                            "apiVersion": "1.0",
                            "requestID": "set_params",
                            "messageType": "InjectParameterDataRequest",
                            "data": {"parameterValues": [
                                {"id": k, "value": v} for k, v in params.items() if k != "MoveSpeed"
                            ]}
                        }
                        frame_buffer.append(inject_request)
                        for req in frame_buffer:
                            await websocket.send(json.dumps(req))
                            try:
                                response = await asyncio.wait_for(websocket.recv(), timeout=0.003)
                                log(f"Parameter update response: {response}", silent=True)
                            except asyncio.TimeoutError:
                                log("Timeout receiving response, continuing...")
                        frame_buffer.clear()
                        prev_params = params.copy()
                        await asyncio.sleep(UPDATE_INTERVAL)
                    except StopIteration:
                        log("Parameter generator stopped unexpectedly. Restarting...")
                        param_gen = parameter_generator(base_values, target_base_values, state)
                        prev_params = None
                        await asyncio.sleep(UPDATE_INTERVAL)
                    except websockets.ConnectionClosed as e:
                        log(f"Connection closed during operation: code={e.code}, reason={e.reason}")
                        raise
                    except Exception as e:
                        log(f"Error during parameter update: {e}")
                        await asyncio.sleep(UPDATE_INTERVAL)

        except websockets.ConnectionClosed as e:
            log(f"Connection closed: code={e.code}, reason={e.reason}")
            connection_attempts += 1
            if connection_attempts >= max_attempts:
                log("Max connection attempts reached. Please check VTube Studio and network settings.")
                break
            log(f"Reconnecting in {reconnect_delay} seconds... (Attempt {connection_attempts}/{max_attempts})")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 1.5, MAX_RECONNECT_DELAY)
        except Exception as e:
            log(f"Unexpected error: {e}")
            connection_attempts += 1
            if connection_attempts >= max_attempts:
                log("Max connection attempts reached. Please check VTube Studio and network settings.")
                break
            log(f"Reconnecting in {reconnect_delay} seconds... (Attempt {connection_attempts}/{max_attempts})")
            await asyncio.sleep(reconnect_delay)
            reconnect_delay = min(reconnect_delay * 1.5, MAX_RECONNECT_DELAY)
        except KeyboardInterrupt:
            log("Received KeyboardInterrupt, shutting down gracefully...")
            if websocket:
                await websocket.close()
            log("Shutdown complete.")
            break

if __name__ == "__main__":
    asyncio.run(main())