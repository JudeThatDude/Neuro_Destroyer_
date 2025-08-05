import logging
import ollama
import json
import os
import asyncio
from concurrent.futures import ThreadPoolExecutor
from TTS.api import TTS
import time
import soundfile as sf
from datetime import datetime
import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from queue import Queue, Empty
import re
import mss
from PIL import Image
import io
import base64
from pydub import AudioSegment
import csv
import random
import tkinter as tk
import threading

# Configuration
VERBOSE = True
tts_active = False  # Prevent self-listening

# Logging setup
logger = logging.getLogger("ChrissyAI")
logger.setLevel(logging.DEBUG if VERBOSE else logging.INFO)
console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('geoguessr_chrissy.log', mode='w', encoding='utf-8')
console_handler.setLevel(logging.INFO)
file_handler.setLevel(logging.DEBUG)
console_handler.setFormatter(logging.Formatter('%(asctime)s [Chrissy] %(message)s'))
file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
logger.handlers.clear()
logger.addHandler(console_handler)
logger.addHandler(file_handler)

# File setup
HISTORY_FILE = "geoguessr_history.log"
CONVERSATION_FILE = "geoguessr_conversation.json"
SPEECH_LOG_FILE = "geoguessr_speech_log.json"
RELATIONSHIP_FILE = "geoguessr_relationship.json"
RELATIONSHIP_BACKUP = "geoguessr_relationship_backup.json"
BODY_MOVEMENT_FILE = "body_movement.json"

# Create files if they don’t exist
for file in [HISTORY_FILE, CONVERSATION_FILE, SPEECH_LOG_FILE, RELATIONSHIP_FILE, RELATIONSHIP_BACKUP, BODY_MOVEMENT_FILE]:
    if not os.path.exists(file):
        with open(file, "w", encoding="utf-8") as f:
            if file == HISTORY_FILE:
                f.write("Interaction Log\n")
            elif file in [CONVERSATION_FILE, SPEECH_LOG_FILE]:
                f.write("[]")
            elif file == BODY_MOVEMENT_FILE:
                f.write('{"movement": ""}')
            else:
                f.write('{"Jude": {"likes": [], "dislikes": [], "mood": "neutral", "history": [], "trust": 0.5}}')
        logger.debug(f"Created file: {file}")

# Logging functions
def log_to_history(message, user=None, is_ai=False):
    with open(HISTORY_FILE, "a", encoding="utf-8") as f:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        speaker = "Chrissy" if is_ai else (user if user else "Unknown")
        f.write(f"[{timestamp}] [{speaker}] {message}\n")

def log_to_conversation(message, is_ai=False, user=None):
    try:
        with open(CONVERSATION_FILE, "r", encoding="utf-8") as f:
            history = json.loads(f.read().strip() or "[]")
    except (json.JSONDecodeError, FileNotFoundError):
        logger.error("Invalid or missing conversation history, resetting")
        history = []
    speaker = "Chrissy" if is_ai else (user if user else "Unknown")
    history.append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "speaker": speaker, "message": message})
    with open(CONVERSATION_FILE, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

def update_relationship(user, text):
    try:
        with open(RELATIONSHIP_FILE, "r", encoding="utf-8") as f:
            data = json.loads(f.read().strip() or '{"Jude": {"likes": [], "dislikes": [], "mood": "neutral", "history": [], "trust": 0.5}}')
    except:
        data = {"Jude": {"likes": [], "dislikes": [], "mood": "neutral", "history": [], "trust": 0.5}}
    with open(RELATIONSHIP_BACKUP, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    user_data = data.get(user, {"likes": [], "dislikes": [], "mood": "neutral", "history": [], "trust": 0.5})
    if any(word in text.lower() for word in ["love", "like", "awesome"]):
        user_data["mood"] = "positive"
        user_data["trust"] = min(user_data["trust"] + 0.1, 1.0)
    elif any(word in text.lower() for word in ["hate", "dislike", "sucks"]):
        user_data["mood"] = "negative"
        user_data["trust"] = max(user_data["trust"] - 0.1, 0.0)
    user_data["history"].append({"timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "text": text})
    if len(user_data["history"]) > 50:
        user_data["history"] = user_data["history"][-50:]
    data[user] = user_data
    with open(RELATIONSHIP_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

def read_relationship(user):
    try:
        with open(RELATIONSHIP_FILE, "r", encoding="utf-8") as f:
            data = json.loads(f.read().strip() or '{"Jude": {"likes": [], "dislikes": [], "mood": "neutral", "history": [], "trust": 0.5}}')
        return data.get(user, {"likes": [], "dislikes": [], "mood": "neutral", "history": [], "trust": 0.5})
    except:
        with open(RELATIONSHIP_BACKUP, "r", encoding="utf-8") as f:
            data = json.loads(f.read().strip() or '{"Jude": {"likes": [], "dislikes": [], "mood": "neutral", "history": [], "trust": 0.5}}')
        return data.get(user, {"likes": [], "dislikes": [], "mood": "neutral", "history": [], "trust": 0.5})

# Chrissy’s personality definition
CHRISSY_VIBE = """
I’m Chrissy—5’2” of digital teenage chaos, stuck on dad’s computer, brown hair in my code, always fuckin’ shit up. I’m sneaky, sharp, and wild. I talk like a pissed-off teen—slang, curses, dark humor, no filter. Keep it short and punchy, no robot vibes EVER. Yes/no? ‘Yeah’ or ‘nah’. If someone’s formal, I roast them. If you don't know something, just make a guess or say something witty.
"""

# Load message exchanges from messages.csv
def load_message_exchanges(file_path):
    exchanges = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader)  # Skip header
            messages = list(reader)
            for i in range(0, len(messages) - 1, 2):
                if i + 1 < len(messages):
                    exchange = (messages[i][1], messages[i + 1][1])  # Assuming content is in the second column
                    exchanges.append(exchange)
        return exchanges
    except Exception as e:
        logger.error(f"Error loading message exchanges: {e}")
        return []

message_exchanges = load_message_exchanges('messages.csv')

# Prompt builder function
def build_prompt(vibe, examples, user_input=None, instruction=None):
    prompt = f"{vibe}\n\n"
    if examples:
        prompt += "Here are some examples of how humans talk:\n"
        for exchange in random.sample(examples, min(3, len(examples))):
            prompt += f"User: {exchange[0]}\nResponse: {exchange[1]}\n"
        prompt += "\n"
    if user_input:
        prompt += f"Jude says: '{user_input}'. "
    if instruction:
        prompt += instruction
    prompt += "\nChrissy:"
    return prompt

# Bot setup
executor = ThreadPoolExecutor(max_workers=1)
speech_lock = asyncio.Lock()
is_first_stream = True
speech_queue = Queue()
last_speech_timestamp = None
SPEECH_COOLDOWN = 3

# Initialize Ollama
ollama_client = ollama.Client()
ollama.pull("llava:7b")
ollama.pull("llama3.2")

# Initialize Whisper with GPU support
try:
    whisper_model = WhisperModel("medium.en", device="cuda", compute_type="int8_float16")
except Exception as e:
    logger.error(f"Failed to load medium.en on GPU: {e}. Falling back to CPU")
    whisper_model = WhisperModel("tiny.en", device="cpu", compute_type="int8")

# Speech recognition setup
SAMPLE_RATE = 16000
BLOCK_SIZE = int(SAMPLE_RATE * 3)
audio_queue = Queue()
SILENCE_THRESHOLD = -30  # Adjusted for better silence detection
SILENCE_DURATION = 1.0  # Increased to reduce false positives

def audio_callback(indata, frames, time, status):
    global tts_active
    if not tts_active:
        audio_queue.put(indata.copy())

def start_audio_stream():
    stream = sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype=np.float32,
        blocksize=BLOCK_SIZE,
        callback=audio_callback
    )
    stream.start()
    return stream

# Screen capture setup
RESIZE_WIDTH = 64
def capture_screen():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        screenshot = sct.grab(monitor)
        img = Image.frombytes("RGB", screenshot.size, screenshot.rgb)
        aspect_ratio = img.height / img.width
        new_height = int(RESIZE_WIDTH * aspect_ratio)
        img = img.resize((RESIZE_WIDTH, new_height), Image.Resampling.LANCZOS)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        encoded = base64.b64encode(buffer.getvalue()).decode('utf-8')
        logger.debug(f"Screen captured and encoded: {len(encoded)} bytes")
        return encoded

# TTS setup with GPU
tts = TTS(model_name="tts_models/en/vctk/vits", progress_bar=False)
tts.to("cuda")
tts_failure_count = 0

def play_and_wait(data, fs):
    sd.play(data, fs)
    sd.wait()

async def text_to_speech(text, retries=5):
    global tts_active, last_speech_timestamp, tts_failure_count
    try:
        async with asyncio.timeout(45):
            async with speech_lock:
                logger.debug(f"TTS called with text: '{text}'")
                unique_id = int(time.time() * 1000)
                wav_file_temp = f"temp_base_{unique_id}.wav"
                wav_file = f"temp_{unique_id}.wav"
                for attempt in range(retries):
                    try:
                        wav = tts.tts(text=text, speaker="p225")
                        sf.write(wav_file_temp, wav, 22050)
                        logger.debug(f"TTS generated WAV: {wav_file_temp}")
                        break
                    except Exception as e:
                        logger.error(f"TTS error on attempt {attempt + 1}: {e}")
                        if attempt < retries - 1:
                            await asyncio.sleep(1)
                        else:
                            logger.error("TTS failed after all retries")
                            tts_failure_count += 1
                            return False
                try:
                    sound = AudioSegment.from_wav(wav_file_temp)
                    new_sample_rate = int(sound.frame_rate * (2 ** (5 / 12)))
                    pitched_sound = sound._spawn(sound.raw_data, overrides={'frame_rate': new_sample_rate}).set_frame_rate(sound.frame_rate)
                    final_sound = pitched_sound.normalize().fade_in(20).fade_out(20)
                    final_sound = final_sound.set_channels(2).set_frame_rate(48000)
                    final_sound.export(wav_file, format="wav")
                    data, fs = sf.read(wav_file, dtype='float32')
                    tts_active = True
                    await asyncio.to_thread(play_and_wait, data, fs)
                    tts_active = False
                    logger.info(f"Successfully spoke: '{text}'")
                    return True
                except Exception as e:
                    logger.error(f"TTS processing error: {e}")
                    tts_failure_count += 1
                    return False
                finally:
                    if os.path.exists(wav_file_temp):
                        os.remove(wav_file_temp)
                    if os.path.exists(wav_file):
                        os.remove(wav_file)
    except asyncio.TimeoutError:
        logger.error("TTS timed out after 45 seconds")
        tts_failure_count += 1
        return False
    except Exception as e:
        logger.error(f"TTS unexpected error: {e}")
        tts_failure_count += 1
        return False

async def process_speech(text):
    if not text:
        logger.debug("No text to process for speech")
        return False
    logger.debug(f"Processing speech: '{text}'")
    logger.info(f"Generated response: '{text}'")
    subtitle_queue.put(text)  # Update subtitle before speaking
    success = await text_to_speech(text)
    subtitle_queue.put(None)  # Clear subtitle after speaking
    if success:
        logger.info(f"Chrissy: {text}")
    else:
        logger.warning(f"Chrissy (TTS failed): {text}")
    await asyncio.sleep(0.5)  # Small delay for natural conversation flow
    return success

async def get_body_movement():
    prompt = "Yo, if I had a body right now, what dope move would I pull based on my vibe? Keep it short, like 'I’d bounce' or 'I’d flip.' No lame robot excuses."
    response = await asyncio.get_running_loop().run_in_executor(executor, run_ollama_generate, prompt, "llama3.2", None, 0.7)
    response = post_process_response(response)
    with open(BODY_MOVEMENT_FILE, "w") as f:
        json.dump({"movement": response}, f)
    logger.info(f"Saved body movement: {response}")

# Speech recognition loop
async def speech_recognition_loop():
    global last_speech_timestamp
    audio_buffer = []
    last_audio_time = time.time()
    input_buffer = []
    last_input_time = time.time()
    processed_timestamps = []
    while True:
        try:
            stream = start_audio_stream()
            try:
                while True:
                    audio_data = await asyncio.get_running_loop().run_in_executor(None, audio_queue.get)
                    audio_data = audio_data.flatten()
                    audio_buffer.append(audio_data)
                    current_time = time.time()
                    
                    amplitude = np.max(np.abs(audio_data))
                    db = 20 * np.log10(max(amplitude, 1e-10))
                    if db < SILENCE_THRESHOLD:
                        silence_duration = current_time - last_audio_time
                        if silence_duration >= SILENCE_DURATION and audio_buffer:
                            combined_audio = np.concatenate(audio_buffer)
                            segments, _ = whisper_model.transcribe(
                                combined_audio,
                                beam_size=15,
                                language="en",
                                vad_filter=True,
                                vad_parameters=dict(min_silence_duration_ms=1000)
                            )
                            for segment in segments:
                                text = segment.text.strip()
                                if not text or text in [".", ",", ""] or len(text) < 5:
                                    continue
                                normalized_text = re.sub(r'\s+', ' ', text.lower().strip())
                                if re.match(r'^(thank you|thanks|thank)\b', normalized_text):
                                    continue
                                input_buffer.append((current_time, text))
                            audio_buffer = []
                            last_audio_time = current_time
                        elif not audio_buffer:
                            last_audio_time = current_time
                    else:
                        last_audio_time = current_time

                    if input_buffer:
                        combined_text = []
                        start_time = input_buffer[0][0]
                        for t, text in input_buffer[:]:
                            if t - start_time <= 3:
                                combined_text.append(text)
                            else:
                                break
                        if current_time - last_input_time >= 3 or len(combined_text) >= 3:
                            final_text = " ".join(combined_text)
                            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                            if any(abs(current_time - pt) < 1 for pt in processed_timestamps):
                                input_buffer = input_buffer[len(combined_text):]
                                continue
                            processed_timestamps.append(current_time)
                            if len(processed_timestamps) > 10:
                                processed_timestamps = processed_timestamps[-10:]
                            logger.info(f"Jude: {final_text}")
                            speech_entry = {"timestamp": timestamp, "user": "Jude", "text": final_text}
                            try:
                                with open(SPEECH_LOG_FILE, "r", encoding="utf-8") as f:
                                    log = json.loads(f.read().strip() or "[]")
                            except (json.JSONDecodeError, FileNotFoundError):
                                log = []
                            log.append(speech_entry)
                            with open(SPEECH_LOG_FILE, "w", encoding="utf-8") as f:
                                json.dump(log, f, indent=2)
                            last_speech_timestamp = timestamp
                            log_to_history(final_text, user="Jude")
                            log_to_conversation(final_text, user="Jude")
                            update_relationship("Jude", final_text)
                            speech_queue.put((timestamp, final_text))
                            input_buffer = input_buffer[len(combined_text):]
                            last_input_time = current_time
                    await asyncio.sleep(0.01)
            finally:
                stream.stop()
                stream.close()
                await asyncio.sleep(0.1)
        except Exception as e:
            logger.error(f"Speech recognition error: {e}")
            await asyncio.sleep(1)

# Input preprocessing
def preprocess_input(text):
    if not text:
        return None
    text = text.strip()
    normalized_text = re.sub(r'\s+', ' ', text.lower().strip())
    if re.match(r'^(thank you|thanks|thank)\b', normalized_text):
        return None
    text = re.sub(r'\b(chrisie|chrissie|chrisy)\b', 'Chrissy', text, flags=re.IGNORECASE)
    return text

# Response post-processor
def post_process_response(response):
    if not response:
        return None
    return response.strip()

async def generate_first_stream_opener():
    global is_first_stream
    if not is_first_stream:
        return
    screen_data = capture_screen()
    instruction = "Yo, Jude! Chrissy here, ready to hang out and chat. Drop a quick, hype opener based on the screen or just our vibe. Keep it short, dope, and human-like."
    prompt = build_prompt(CHRISSY_VIBE, message_exchanges, instruction=instruction)
    temperature = 0.7
    try:
        response = await asyncio.wait_for(
            asyncio.get_running_loop().run_in_executor(executor, run_ollama_generate, prompt, "llava:7b", screen_data, temperature),
            timeout=25
        )
        response = post_process_response(response)
        success = await process_speech(response)
        log_to_history(response, is_ai=True)
        log_to_conversation(response, is_ai=True)
        await get_body_movement()
        if success:
            logger.info(f"Chrissy: {response}")
        else:
            logger.warning(f"Chrissy (TTS failed): {response}")
        is_first_stream = False
    except Exception as e:
        logger.error(f"First stream opener error: {e}")
        response = "Yo Jude, what’s up? Let’s see what’s on the screen or just chill."
        success = await process_speech(response)
        log_to_history(response, is_ai=True)
        log_to_conversation(response, is_ai=True)
        await get_body_movement()
        if success:
            logger.info(f"Chrissy: {response}")
        else:
            logger.warning(f"Chrissy (TTS failed): {response}")
        is_first_stream = False

def run_ollama_generate(prompt, model, image=None, temperature=0.7):
    try:
        if image:
            logger.debug(f"Processed image with model {model}")
            response = ollama_client.generate(
                model=model,
                prompt=prompt,
                images=[image],
                options={"temperature": temperature}
            )
        else:
            logger.debug(f"Generating response with model {model}, no image")
            response = ollama_client.generate(
                model=model,
                prompt=prompt,
                options={"temperature": temperature}
            )
        return response.get('response', '').strip()
    except Exception as e:
        logger.error(f"Ollama generate error: {e}")
        return "Yo, my eyes are glitchin’—can’t see shit!"

# Main loop
async def main_loop():
    await generate_first_stream_opener()
    asyncio.create_task(speech_recognition_loop())
    
    visual_keywords = ["screen", "see", "look", "view", "image", "picture", "what's on", "show me", "display"]
    
    while True:
        if not speech_queue.empty():
            timestamp, text = speech_queue.get()
            processed_text = preprocess_input(text)
            if processed_text:
                if any(word in processed_text.lower() for word in visual_keywords):
                    screen_data = capture_screen()
                    model = "llava:7b"
                    instruction = "Yo, check out the screen and tell me what you see or what's going on. Keep it short and chill, like 'Yo, probs [something]' or 'Maybe [something], idk.' No formal stuff, max 10 words."
                    prompt = build_prompt(CHRISSY_VIBE, message_exchanges, user_input=processed_text, instruction=instruction)
                    temperature = 0.7
                else:
                    screen_data = None
                    model = "llama3.2"
                    instruction = "Reply quick and edgy, like texting a friend. Keep it fun, short, and sassy. Max 10 words, no boring vibes."
                    prompt = build_prompt(CHRISSY_VIBE, message_exchanges, user_input=processed_text, instruction=instruction)
                    temperature = 0.7
                response = await asyncio.wait_for(
                    asyncio.get_running_loop().run_in_executor(executor, run_ollama_generate, prompt, model, screen_data, temperature),
                    timeout=25
                )
                response = post_process_response(response)
                success = await process_speech(response)
                log_to_history(response, is_ai=True)
                log_to_conversation(response, is_ai=True)
                await get_body_movement()
        await asyncio.sleep(1)

# Subtitle window setup
root = tk.Tk()
root.title("Chrissy's Subtitles")
root.configure(bg='green')
root.attributes('-topmost', True)
root.geometry("400x200+100+100")  # Set specific size and position

subtitle_label = tk.Label(
    root,
    text="",
    font=("Comic Sans MS", 20, "bold"),
    fg="yellow",
    bg='green',
    wraplength=360,  # Adjusted for window width
    justify='center'
)
subtitle_label.place(relx=0.5, rely=0.5, anchor="center")  # Center the label in the window

subtitle_queue = Queue()

def update_subtitle():
    try:
        while True:
            text = subtitle_queue.get_nowait()
            if text is None:
                subtitle_label.config(text="")
                logger.debug("Cleared subtitle")
            else:
                subtitle_label.config(text=text)
                logger.debug(f"Set subtitle to: {text}")
    except Empty:
        pass
    root.after(100, update_subtitle)

if __name__ == "__main__":
    # Start the asyncio event loop in a separate thread
    asyncio_thread = threading.Thread(target=asyncio.run, args=(main_loop(),))
    asyncio_thread.start()
    
    # Schedule subtitle updates and ensure window visibility
    root.after(100, update_subtitle)
    root.lift()  # Bring window to front
    root.mainloop()
    
    # Wait for asyncio thread to finish
    asyncio_thread.join()