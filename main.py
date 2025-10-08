import sounddevice as sd
from scipy.io.wavfile import write
import torch
import numpy as np
import keyboard
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from kokoro import KPipeline
from IPython.display import display, Audio
import soundfile as sf
import requests
import simpleaudio as sa
import time
import json

chatPrompt = r"You are a chemist who touched the end of a usb cable and became trapped in the computer. Your goal is to find a way to get free."

messageHistory = []

if input("Attempt to load chat history (y/n): ") == 'y':
    json.dump(messageHistory, open("messages.json", "r+"))
else:
    messageHistory = [
        {"role": "system", "content": f"{chatPrompt}"},
    ]

#Initialize STT pipeline
TTSPipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo", # select checkpoint from https://huggingface.co/openai/whisper-large-v3#model-details
    torch_dtype=torch.float16,
    device="cuda:0", # or mps for Mac devices
    model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
)

koboldCPPAddress = "http://localhost:5001/v1/chat/completions"

while True:
    #record input
    sample_rate = 44100  # Hz
    channels = 1
    dtype = 'int16'

    print("Hold down the 'r' key to record, 'z' key to undo. Release to stop.")

    buffer = []
    with sd.InputStream(samplerate=sample_rate, channels=channels, dtype=dtype, blocksize=1024) as stream:
        while len(buffer) == 0 or keyboard.is_pressed('r'):
            if keyboard.is_pressed('z') and len(messageHistory) > 1:
                print(f"Undid: {messageHistory.pop()}")
                time.sleep(0.5)
            if keyboard.is_pressed('r'):
                data, overflowed = stream.read(1024)
                buffer.append(data)
                print("Recording...", end='\r')

    # Concatenate all chunks into one array
    recording = np.concatenate(buffer, axis=0)

    # Save as WAV file
    write('voice_recording.wav', sample_rate, recording)

    print("Recording saved as voice_recording.wav")

    outputs = TTSPipeline(
        "voice_recording.wav",
        chunk_length_s=30,
        batch_size=24,
        return_timestamps=True,
    )

    pipeline = KPipeline(lang_code='a')

    print(outputs)

    messageHistory.append({"role": "user", "content": f"{outputs['text']}"})

    LLMPayload = {
        "model": "gpt-4o",
        "messages": messageHistory,
        "temperature": 0.7,
        "max_tokens": 300,
    }

    # Send the request
    response = requests.post(koboldCPPAddress, json=LLMPayload)

    # Check response status
    if response.status_code == 200:
        data = response.json()
        reply = data['choices'][0]['message']['content']
        print("KoboldCPP reply:", reply)
    else:
        print("Error:", response.status_code, response.text)
        reply = "failed"

    messageHistory.append({"role": "assistant", "content": f"{reply}"})

    TTSPayload = reply

    generator = pipeline(TTSPayload, voice='af_heart')
    for i, (gs, ps, audio) in enumerate(generator):
        if not keyboard.is_pressed('x'):
            print(i, gs, ps)
            display(Audio(data=audio, rate=24000, autoplay=i == 0))

            audio_np = audio.cpu().numpy()

            sf.write(f'./sounds/{i}.wav', audio_np, 24000)

            audio_int16 = (audio_np * 32767).astype('int16')

            try:
                play_obj.wait_done()
            except NameError:
                pass

            play_obj = sa.play_buffer(audio_int16, 1, 2, 24000)

    try:
        if not keyboard.is_pressed('x'):
            play_obj.wait_done()
    except NameError:
        pass
