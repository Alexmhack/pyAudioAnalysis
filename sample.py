from pyAudioAnalysis import audioSilencePeriods as aSP
from pyAudioAnalysis import audioBasicIO as aIO
from pyAudioAnalysis import audioSegmentation as aS

[Fs, x] = aIO.readAudioFile("custom/audio/output.wav")

segments = aSP.silenceCounter(x, Fs, 0.020, 0.020,
	smoothWindow=1.0, weight=0.3, plot=True)

silences = aS.silenceRemoval(x, Fs, 0.020, 0.020,
	smoothWindow=1.0, weight=0.3, plot=True)

print(silences)
