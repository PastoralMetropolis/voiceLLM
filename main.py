import sounddevice as sd
from scipy.io.wavfile import write
import torch
import numpy as np
import keyboard
from transformers import pipeline
from transformers.utils import is_flash_attn_2_available
from kokoro import KPipeline
from IPython.display import display, Audio
import requests
import time
import json

chatPrompt = open("prompt.txt", "r+").read()
print(chatPrompt)

messageHistory = []

if input("Attempt to load chat history (y/n): ") == 'y':
    messageHistory = json.load(open("messages.json", "r+"))
else:
    messageHistory = [
        {"role": "system", "content": f"{chatPrompt}"},
    ]

#Initialize STT pipeline
TTSPipeline = pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-large-v3-turbo",
    torch_dtype=torch.float16,
    device="cuda:0",
    model_kwargs={"attn_implementation": "flash_attention_2"} if is_flash_attn_2_available() else {"attn_implementation": "sdpa"},
)

pipeline = KPipeline(lang_code='a', repo_id='hexgrad/Kokoro-82M')

koboldCPPAddress = "http://localhost:5001/v1/chat/completions"

sample_rate = 44100  # Hz
channels = 1
dtype = 'int16'

activationMin = 1100
activationPortion = .4

while True:

    flagDefault = 33
    flag = flagDefault
    buffer = []

    with sd.InputStream(samplerate=sample_rate, channels=channels, blocksize=1024, dtype=dtype) as stream:
        print("Recording Begins.")
        while flag > 0:
            while flag > 0:
                if keyboard.is_pressed('z') and len(messageHistory) > 1 and len(buffer) == 0:
                    print(f"Undid: {messageHistory.pop()}")
                    time.sleep(0.5)
                else:
                    data, overflowed = stream.read(1024)
                    # print(data)
                    activatedCount = 0
                    for i in data:
                        if abs(i) > activationMin:
                            activatedCount += 1
                    if activatedCount/len(data) > activationPortion:
                        buffer.append(data)
                        flag = flagDefault
                    elif len(buffer) > 2:
                        flag -= 1
                        if flag > flagDefault//2:
                            buffer.append(data)

                    print("Recording...", end='\r')

            # Concatenate all chunks into one array
            recording = np.concatenate(buffer, axis=0)

            # Save as WAV file
            write('voice_recording.wav', sample_rate, recording)

            print("Recording saved as voice_recording.wav")
            currentTime = time.time()
            outputs = TTSPipeline(
                "voice_recording.wav",
                chunk_length_s=30,
                batch_size=24,
                return_timestamps=True,
            )
            print(f"task complete in {currentTime-time.time()}")
            print(outputs)

            messageHistory.append({"role": "user", "content": f"{outputs['text']}"})

            LLMPayload = {
                "model": "gpt-4o",
                "messages": messageHistory,
                "temperature": 0.7,
                "max_tokens": 300,
                "stream": True,
            }

            reply = ""

            with requests.post(koboldCPPAddress, json=LLMPayload, stream=True) as response:
                if response.status_code != 200:
                    print(f"Error {response.status_code}: {response.text}")
                    reply = "failed"
                    flag = 20
                else:
                    for line in response.iter_lines(decode_unicode=True):
                        if line:
                            if line.startswith("data:"):
                                responseData = line[len("data:"):].strip()
                            if responseData == "[DONE]":
                                break
                            jsonData = json.loads(responseData)
                            reply += jsonData["choices"][0]["delta"]["content"]

                            while stream.read_available >= 1024:
                                data, overflowed = stream.read(1024)
                                # print(data)
                                activatedCount = 0
                                for i in data:
                                    if abs(i) > activationMin:
                                        activatedCount += 1
                                if activatedCount / len(data) > activationPortion:
                                    buffer.append(data)
                                    flag = flagDefault
                                    break
                            if flag > 0:
                                break


    print("KoboldCPP reply:", reply)

    messageHistory.append({"role": "assistant", "content": f"{reply}"})

    TTSPayload = reply

    generator = pipeline(TTSPayload, voice='af_heart')
    for i, (gs, ps, audio) in enumerate(generator):
        if not keyboard.is_pressed('x'):
            print(i, gs, ps)
            display(Audio(data=audio, rate=24000, autoplay=i == 0))

            audio_np = audio.cpu().numpy()

            #sf.write(f'./sounds/{i}.wav', audio_np, 24000)

            audio_int16 = (audio_np * 32767).astype('int16')

            try:
                sd.wait()
                sd.play(audio_int16,24000)
            except NameError:
                pass
    json.dump(messageHistory, open("messages.json", "r+"))
    sd.wait()