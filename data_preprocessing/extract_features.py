#-----------------------------------------------------------------------------#
#-------------------------------Extract_Features------------------------------#
#-Author: Michael Issa----------------------------------------Date: 2/19/2024-#

import os
import pandas as pd
import numpy as np
import librosa
import csv

# Requisite imports for mfcc function
from scipy.signal import get_window
import scipy.fftpack as fft



#-----------------------------------------------------------------------------#
# This file contains various functions intended to extract features from binaural
# audio sources to localize the sound source. The sound source is localized only
# according to the azimuth and ignores the elevation. Due to uncertainty concerning
# what features of the sound are optimal for localization, I've written a few different
# functions to extract various features that may aid in prediction. 
#-----------------------------------------------------------------------------#



# @brief Extracts interaural time difference (ITD) from stereo audio files.

# @param left_wav_file  Path to the left ear audio file
# @param right_wav_file Path to the right ear audio file
# @param frame_size     Size of the analysis frame in samples
# @param hop_size       Number of samples between consecutive frames
# @return np.array      Array containing ITD values

def extract_ITD(left_wav_file, right_wav_file, frame_size=640, hop_size=320):
    
    # Load left and right ear audio
    left_audio, sr = librosa.load(left_wav_file, sr=None)
    right_audio, sr = librosa.load(right_wav_file, sr=None)

    # Ensure equal lengths by taking the minimum of the two
    min_length = min(len(left_audio), len(right_audio))
    left_audio = left_audio[:min_length]
    right_audio = right_audio[:min_length]

    # Calculate ITD
    ITD = []
    for i in range(0, len(left_audio), hop_size):
        left_frame = left_audio[i:i+frame_size]
        right_frame = right_audio[i: i+frame_size]

        cross_correlation = np.correlate(left_frame, right_frame, mode='full')
        peak = np.argmax(cross_correlation)
        ITD.append((peak-frame_size) / sr)
    
    return np.array(ITD)



# @brief Extracts interaural level difference (ILD) from stereo audio files.

# @param left_wav_file  Path to the left ear audio file
# @param right_wav_file Path to the right ear audio file
# @param frame_size     Size of the analysis frame in samples
# @param hop_size       Number of samples between consecutive frames
# @return np.array      Array containing ILD values

def extract_ILD(left_wav_file, right_wav_file, frame_size=640, hop_size=320):
    # Load left and right ear audio files
    left_audio, sr = librosa.load(left_wav_file, sr=None)
    right_audio, sr = librosa.load(right_wav_file, sr=None)

    # Ensure both audio files have the same length
    min_length = min(len(left_audio), len(right_audio))
    left_audio = left_audio[:min_length]
    right_audio = right_audio[:min_length]

    # Calculate ILD 
    ILD = []
    for i in range(0, len(left_audio), hop_size):
        left_frame = left_audio[i:i+frame_size]
        right_frame = right_audio[i:i+frame_size]

        # Compute root-mean-square for each frame
        left_rms = np.sqrt(np.mean(np.square(left_frame)))
        right_rms = np.sqrt(np.mean(np.square(right_frame)))

        # Calculate ILD for the frame
        ILD.append(20 * np.log10(left_rms / right_rms))

    return np.array(ILD)



# @brief Extracts spectral interaural time difference (ITD) from stereo audio files using STFT.

# @param left_wav_file  Path to the left ear audio file
# @param right_wav_file Path to the right ear audio file
# @param frame_size     Size of the analysis frame in samples
# @param hop_size       Number of samples between consecutive frames
# @param n_fft          Length of the FFT window
# @param sr             Sampling rate of the audio files
# @return np.array      Array containing spectral ITD values

def extract_spectral_itd(left_wav_file, right_wav_file, frame_size=2048, hop_size=256, n_fft=512, sr=44100):
    # Load left and right ear audio files
    left_audio, sr = librosa.load(left_wav_file, sr=sr)
    right_audio, sr = librosa.load(right_wav_file, sr=sr)

    # Ensure both audio files have the same length
    min_length = min(len(left_audio), len(right_audio))
    left_audio = left_audio[:min_length]
    right_audio = right_audio[:min_length]

    # Calculate STFT (Short-Time Fourier Transform) for left and right audio
    left_stft = librosa.stft(left_audio, n_fft=n_fft, hop_length=hop_size)
    right_stft = librosa.stft(right_audio, n_fft=n_fft, hop_length=hop_size)

    # Calculate spectral ITD
    spectral_itd = []
    for i in range(left_stft.shape[1]):
        left_spectrum = np.abs(left_stft[:, i])
        right_spectrum = np.abs(right_stft[:, i])

        # Compute the cross-correlation between left and right spectra
        xcorr = np.correlate(left_spectrum, right_spectrum, mode='full')

        # Find the index of the peak in the cross-correlation
        peak_index = np.argmax(xcorr)

        # Convert peak index to frequency-dependent time delay (spectral ITD)
        itd = (peak_index - len(left_spectrum)) * hop_size / sr
        spectral_itd.append(itd)

    return np.array(spectral_itd)


 
# @brief Extracts spectral interaural level difference (ILD) from stereo audio files using STFT.

# @param left_wav_file  Path to the left ear audio file
# @param right_wav_file Path to the right ear audio file
# @param frame_size     Size of the analysis frame in samples
# @param hop_size       Number of samples between consecutive frames
# @param n_fft          Length of the FFT window
# @param sr             Sampling rate of the audio files
# @return np.array      Array containing spectral ILD values

def extract_spectral_ild(left_wav_file, right_wav_file, frame_size=2048, hop_size=320, n_fft=640, sr=44100):
    # Load left and right ear audio files
    left_audio, sr = librosa.load(left_wav_file, sr=sr)
    right_audio, sr = librosa.load(right_wav_file, sr=sr)

    # Ensure both audio files have the same length
    min_length = min(len(left_audio), len(right_audio))
    left_audio = left_audio[:min_length]
    right_audio = right_audio[:min_length]

    # Calculate STFT (Short-Time Fourier Transform) for left and right audio
    left_stft = librosa.stft(left_audio, n_fft=n_fft, hop_length=hop_size)
    right_stft = librosa.stft(right_audio, n_fft=n_fft, hop_length=hop_size)

    # Calculate spectral ILD
    spectral_ild = []
    for i in range(left_stft.shape[1]):
        left_spectrum = np.abs(left_stft[:, i])
        right_spectrum = np.abs(right_stft[:, i])

        # Compute root-mean-square (RMS) energy for each spectrum
        left_rms = np.sqrt(np.mean(np.square(left_spectrum)))
        right_rms = np.sqrt(np.mean(np.square(right_spectrum)))

        # Calculate ILD for the spectrum
        ild = 20 * np.log10(left_rms / right_rms)
        spectral_ild.append(ild)

    return np.array(spectral_ild)


# @brief Extracts Mel-frequency cepstral coefficients (MFCCs) from stereo audio files.
  
# @param left_wav_file  Path to the left ear audio file
# @param right_wav_file Path to the right ear audio file
# @param num_mfcc       Number of MFCC coefficients to extract
# @param fft_size       Size of the FFT window in samples
# @param hop_size       Number of samples between consecutive frames
# @param num_mel_bins   Number of Mel frequency bins
# @return np.array      Matrix containing MFCCs for each frame

def extract_mfcc(left_wav_file, right_wav_file, num_mfcc=13, fft_size=2048, hop_size=512, num_mel_bins=128):
    # Load left and right ear audio files
    left_audio, sr = librosa.load(left_wav_file, sr=None)
    right_audio, sr = librosa.load(right_wav_file, sr=None)

    # Ensure both audio files have the same length
    min_length = min(len(left_audio), len(right_audio))
    left_audio = left_audio[:min_length]
    right_audio = right_audio[:min_length]

    # Combine the channels
    combined_signal = np.stack((left_audio, right_audio)).mean(axis=0)

    # Compute magnitude spectrum
    window = get_window("hamming", fft_size)
    frames = len(combined_signal) // hop_size + 1
    stft = np.abs(np.array([fft.fft(window * combined_signal[i*hop_size:i*hop_size + fft_size]) for i in range(frames)]))[:, :fft_size//2 + 1]

    # Compute power spectrum
    power = (stft**2) / fft_size

    # Apply mel filter 
    mel_filters = mel_filter(num_mel_bins, fft_size, sr)
    mel_spectrum = np.dot(power, mel_filters.T)
    
    # Convert to dB
    mel_spectrum_db = 20 * np.log10(np.maximum(1e-5, mel_spectrum))
    
    # Compute DCT
    dct_filters = fft.dct(np.eye(num_mfcc), axis=0, norm='ortho')
    mfcc = np.dot(mel_spectrum_db, dct_filters.T)
    
    return mfcc

#-------------------------------- Helper Functions --------------------------------#


# @brief Converts frequency in Hertz to Mel scale.
# @param hz Frequency in Hertz
# @return float Corresponding value in Mel scale
def hz_to_mel(hz):
    return 2595 * np.log10(1 + hz / 700)


# @brief Converts frequency in Mel scale to Hertz.
# @param mel Value in Mel scale
# @return float Corresponding frequency in Hertz
def mel_to_hz(mel):
    return 700 * (10**(mel / 2595) - 1)



# @brief Generates Mel filter bank for use in Mel-frequency cepstral coefficient (MFCC) calculation.
# @param num_filters Number of filters in the filter bank
# @param fft_size Size of the FFT window in samples
# @param sr Sampling rate of the audio signal
# @return np.array Matrix representing the Mel filter bank
def mel_filter(num_filters, fft_size, sr):
    high_mel = hz_to_mel(sr / 2)
    mel_points = np.linspace(0, high_mel, num_filters + 2)
    hz_points = mel_to_hz(mel_points)
    bin_points = np.floor((fft_size + 1) * hz_points / sr).astype(int)

    filters = np.zeros((num_filters, fft_size // 2 + 1))

    for i in range(1, num_filters + 1):
        filters[i - 1, bin_points[i-1]:bin_points[i]] = np.where(bin_points[i] - bin_points[i - 1] != 0, (bin_points[i] - bin_points[i - 1]) / (bin_points[i + 1] - bin_points[i]), 0)
        filters[i - 1, bin_points[i]:bin_points[i + 1]] = np.where(bin_points[i + 1] - bin_points[i] != 0, (bin_points[i + 1] - bin_points[i]) / (bin_points[i + 1] - bin_points[i]), 0)
    
    return filters



# @brief Extracts elevation and azimuth angles from a given filename.
# @param filename Name of the audio file containing elevation and azimuth information
# @return Tuple containing elevation and azimuth angles extracted from the filename

def extract_azimuth_from_filename(filename):
    # Posiiton index for the end of the elevation value and azimuth
    e_index = filename.index("e")
    

    elevation_str = ""
    azimuth_str = ""

    # Find the index for elevation, "e"
    start_index = filename.index("L") + 1 if "L" in filename else filename.index("R") + 1
    if filename[start_index] == "-":
        elevation_str += "-"
        start_index +=1
    
    # Extract value for elevation
    while filename[start_index] != "e":
        elevation_str += filename[start_index]
        start_index +=1

    # Extract three digits of the azimuth
    for i in range(e_index+1, e_index+4):
        azimuth_str += filename[i]


    elevation = int(elevation_str)
    azimuth = int(azimuth_str)

    return elevation, azimuth


import os
import csv

if __name__ == '__main__':
    data_dir = 'data/raw'
    output_dir = 'data/processed_data'

    for elevation_file in os.listdir(data_dir):
        if elevation_file.startswith("elev"):
            elevation_dir = os.path.join(data_dir, elevation_file)

            # Loop through each right and left ear file in the specified elevation
            for left_file in os.listdir(elevation_dir):
                if left_file.startswith("L"):
                    left_wav_file = os.path.join(elevation_dir, left_file)
                    right_wav_file = os.path.join(elevation_dir, left_file.replace("L", "R"))

                    # Extract elevation and azimuth
                    elevation, azimuth = extract_azimuth_from_filename(left_file)

                    # Extract various features
                    itd = extract_ITD(left_wav_file, right_wav_file) # Note: Using default values for frame_size and the hop, which will need to be modified
                    ild = extract_ILD(left_wav_file, right_wav_file)
                    spectral_itd = extract_spectral_itd(left_wav_file, right_wav_file)
                    spectral_ild = extract_spectral_ild(left_wav_file, right_wav_file)
                    left_mfcc, right_mfcc = extract_mfcc(left_wav_file, right_wav_file)

                    output_file = os.path.join(output_dir, f"{elevation}_azimuth.csv")

                    if not os.path.isfile(output_file):
                        # Write labels only once
                        labels = ["Elevation", "Azimuth", "ITD", "ILD", "Spectral ILD", "Spectral ITD", "Left MFCC", "Right MFCC"] #there are two columns, representing the real and imaginary parts of the complex MFCC coefficients
                        with open(output_file, mode='w', newline='') as csv_file:
                            writer = csv.writer(csv_file)
                            writer.writerow(labels)

                    # Append data to the file
                    with open(output_file, mode='a', newline='') as csv_file:
                        writer = csv.writer(csv_file)
                        writer.writerow([elevation, azimuth, itd, ild, spectral_ild, spectral_itd, left_mfcc, right_mfcc])
                        
    data_dir = 'data/processed_data'
    output_file = 'combined_data.csv'

    # Initialize an empty list to store DataFrames
    dfs = []

    # Loop through each CSV file in the data directory
    for csv_file in os.listdir(data_dir):
        if csv_file.endswith(".csv"):
            csv_file_path = os.path.join(data_dir, csv_file)
            
            # Read the CSV file into a DataFrame
            df = pd.read_csv(csv_file_path)
            
            # Append the DataFrame to the list of DataFrames
            dfs.append(df)

    # Concatenate all DataFrames into a single DataFrame
    combined_data = pd.concat(dfs, ignore_index=True)

    # Save the combined DataFrame to a new CSV file
    combined_data.to_csv(output_file, index=False)


                

                
            