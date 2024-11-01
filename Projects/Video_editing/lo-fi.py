# Advanced lo-fi
import numpy as np
import soundfile as sf
import librosa

# Load the audio file
data, rate = librosa.load('song.wav', sr=None)

# Downsample the audio
downsampled_data = librosa.resample(data, rate, 22050)

# Add some noise to the downsampled audio
noise = np.random.normal(0, 0.1, downsampled_data.shape)
downsampled_data += noise

# Apply a low pass filter
lowpass = librosa.effects.lowpass(downsampled_data, cutoff=3000, sr=22050)

# Apply compression
compressed = librosa.effects.compression(lowpass, gain=6, ratio=2, threshold=-20)

# Apply distortion
distorted = librosa.effects.harmonic(compressed, margin=2)

# Saving the lofi audio
sf.write('lofi_song.wav', distorted, 22050)


#Method 1
# import numpy as np
# from scipy.io import wavfile
# from scipy import signal

# # Load the audio file
# rate, data = wavfile.read('song.wav')

# # Create a lo-fi version of the song by downsampling the audio
# downsampled_data = signal.decimate(data, rate // 22050)

# # Add some noise to the downsampled audio
# noise = np.random.normal(0, 0.1, downsampled_data.shape)
# downsampled_data += noise

# # Save the lo-fi version of the song
# wavfile.write('lofi_song.wav', 22050, downsampled_data)
