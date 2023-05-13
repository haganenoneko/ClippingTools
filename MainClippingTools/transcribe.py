from pathlib import Path 
from subprocess import Popen 
from google.cloud import speech

KEYPATH = "./MainClippingTools/tonal-depth-384910-a1342e5efb4e.json"
Popen(['powershell', f'$env:GOOGLE_APPLICATION_CREDENTIALS="{KEYPATH}"'])

def extract_audio(vp: Path) -> Path:
	outp = vp.parent / f"{vp.stem}.mp3"
	cmd = f"ffmpeg -i {vp} -vn -map a {outp}"
	Popen(['powershell', cmd])
	return outp

def transcribe_speech(audio_file: Path) -> None:
	client = speech.SpeechClient()

	with open(audio_file, 'rb') as file:
		audio = speech.RecognitionAudio(file.read())

	config = speech.RecognitionConfig(
	encoding=speech.RecognitionConfig.AudioEncoding.MP3,
		sample_rate_hertz=44100,
		language_code="ja-JP",
		model="latest_long",
		audio_channel_count=2,
		enable_automatic_punctuation=True,
		enable_word_confidence=True,
		enable_word_time_offsets=True,
		max_alternatives=2,
	)

	# Detects speech in the audio file
	operation = client.long_running_recognize(config=config, audio=audio)

	print("Waiting for operation to complete...")
	response = operation.result(timeout=90)
	
	outp = audio_file.parent / f"{audio_file.stem}_transcript.txt"
	with open(outp, 'w') as io:
		for result in response.results:
			txt = result.alternatives[0].transcript
			print("Transcript: {}".format(txt))
			io.write(txt)

	print(f"Transcript saved to {outp}")
	return 