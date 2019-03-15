import pyaudio
import wave

CHUNK = 1024

wf = wave.open('audio/output.wav', 'rb')

p = pyaudio.PyAudio()

stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
	channels=wf.getnchannels(), rate=wf.getframerate(),
	output=True)

data = wf.readframes(CHUNK)

while len(data) > 0:
	stream.write(data)
	data = wf.readframes(CHUNK)

stream.stop_stream()
stream.close()

p.terminate()
