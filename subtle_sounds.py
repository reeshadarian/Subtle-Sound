import pyaudio
import webbrowser
import speech_recognition
import wave
import matplotlib.pyplot as plt
import numpy as np
recog = speech_recognition.Recognizer()
import parselmouth
from parselmouth.praat import call
import os
from pyAudioAnalysis import audioSegmentation, ShortTermFeatures
from scipy.io import wavfile
import librosa
import librosa.display
import PySimpleGUI as sg
import matplotlib as mpl
mpl.rcParams['agg.path.chunksize'] = 10000

def make_plots(filename, signal_type,  file_path):
    # Read file to get buffer                                                                                               
    ifile = wave.open(filename)
    samples = ifile.getnframes()
    audio = ifile.readframes(samples)

    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
    max_int16 = 2**15
    audio_normalised = audio_as_np_float32 / max_int16

    t = np.linspace(0, len(audio_normalised)/44100, len(audio_normalised))
    plt.plot(t[:len(audio_normalised)], audio_normalised)
    plt.title("Amplitude ({0})".format(signal_type))
    plt.savefig(file_path+signal_type+"Plot.png", dpi = 1200)
    plt.close()
    
    y, sr = librosa.load(filename)
    librosa.display.waveplot(y, sr=sr);
    hop_length = 512
    D = np.abs(librosa.stft(y, n_fft=2048,  hop_length=128))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log');
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectogram ({0})".format(signal_type))
    plt.savefig(file_path+signal_type+"Spectogram.png", dpi = 1200)
    plt.close()

def measurePitch(file_name, f0min = 20, f0max = 1000, unit = 'Hertz', sound_type = None):
    voiceID = parselmouth.Sound(file_name)
    sound = parselmouth.Sound(voiceID) # read the sound
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 20, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    return hnr, localJitter, localabsoluteJitter, localShimmer, localdbShimmer

def shortTermAnalyses(sound_type, filename, patient_name):
    fs, signal = wavfile.read(filename)
    if sound_type == 'speech':
        s = audioSegmentation.silence_removal(signal, fs, 0.5, 0.1, weight=0.2)
        signal2 = np.concatenate([signal[int((i[0]+0.1)*fs):int((i[1]+0.1)*fs)] for i in s])
        wavfile.write("database/{0}/speechFileSegmented.wav".format(patient_name), fs, signal2)
    return ShortTermFeatures.feature_extraction(signal, fs, 0.05*fs, 0.025*fs, deltas=True)[0][:8]

def display_data(signal_type,  patient_name):
    res = ''
    file_path = 'database/' + patient_name + '/'
    filename = file_path + signal_type + 'File.wav'
    if signal_type == 'heart':
        title = 'Heart Data'
        
    elif signal_type == 'speech':
        title = 'Speech Data'
        segmented = file_path + signal_type + 'FileSegmented.wav'
    else:
        title = 'Breathing Data'
    ifile = wave.open(filename)
    samples = ifile.getnframes()
    audio = ifile.readframes(samples)
    audio_as_np_int16 = np.frombuffer(audio, dtype=np.int16)
    audio_as_np_float32 = audio_as_np_int16.astype(np.float32)
    max_int16 = 2**15
    audio_normalised = audio_as_np_float32 / max_int16
    yy = audio_normalised
    make_plots(filename, signal_type,  file_path)
    if signal_type == 'speech':
        file = open(patient_name+"transcriptFile.txt", "w")
        with speech_recognition.WavFile(filename) as source:              # use "test.wav" as the audio source
            audio = recog.record(source)                        # extract audio data from the file
            try:
                res += 'Transcript: ' + recog.recognize_google(audio) +'<br>'
            except:                                 # speech is unintelligible
                res += 'Transcript: Could not understand audio' + '<br>'
        file.close()
        
    data = shortTermAnalyses(signal_type, filename, patient_name)
    names = ['ZCR', 'Short Time Energy', 'Energy Entropy', 'Spectral Centroid', 'Spectral Spread', 'Spectral Entropy', 'Spectral Flux', 'Spectral Rolloff']
    for i in range(8):
        plt.plot(np.linspace(0, len(yy)/44100, len(data[i])), data[i])
        plt.title("{0} ({2}), Mean = {1:.3e}".format(names[i], np.mean(data[i]), signal_type))
        plt.savefig(file_path+signal_type+names[i]+'.png', dpi = 1200)
        plt.close()
    if signal_type == "speech":
        hnr, localJitter, localabsoluteJitter, localShimmer, localdbShimmer = measurePitch(segmented)
    else:
        hnr, localJitter, localabsoluteJitter, localShimmer, localdbShimmer = measurePitch(filename)
    res += 'HNR: ' + str(hnr) + '<br>'
    res += 'Local Jitter: ' + str(localJitter) + '<br>'
    res += 'Local Absolute Jitter: ' + str(localabsoluteJitter) + '<br>'
    res += 'Local Shimmer: ' + str(localShimmer) + '<br>'
    res += 'Local Shimmer dB: ' + str(localdbShimmer) + '<br>'
    return res

layout = [  [sg.InputText(key = 'patientNameSearch'), sg.Button("Search Patient", key='patientNameSearchButton')],
            [sg.Text('Patient Name'), sg.InputText(key='patientName')],
            
            [sg.Button('Capture Heart', key='heartRec'), sg.Button('Stop', disabled=True, key='heartStop')],
            [sg.Button('Capture Breath', key='breathRec'), sg.Button('Stop', disabled=True, key='breathStop')],
            [sg.Button('Capture Speech', key='speechRec'), sg.Button('Stop', disabled=True, key='speechStop')],
            [sg.Text("Message:", size = (50, 1), key='messages')],
            [sg.Button('Done'), sg.Button('Cancel')]]

window = sg.Window('Subtle Sounds', layout, finalize=True)

def callback(in_data, frame_count, time_info, status):
    Recordframes.append(in_data)
    return (in_data, pyaudio.paContinue)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    elif event == 'patientNameSearchButton':
        patient_name = values['patientNameSearch']
        if os.path.exists('database/{0}/index.html'.format(patient_name)):
            webbrowser.open('file://' + os.path.realpath('database/'+patient_name+'/index.html'))
        else:
            window['messages'].update('Message: File does not exist')
        
    elif event == 'speechRec':
        p = pyaudio.PyAudio()
        window.refresh()
        window['messages'].update('Message: Recording speech')
        window.refresh()
        Recordframes = []
        patient_name = values['patientName']
        if patient_name != '':
            if not os.path.exists('database/{0}/'.format(patient_name)):
                os.makedirs('database/{0}/'.format(patient_name))
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, input_device_index = 2, frames_per_buffer=1024, stream_callback=callback)
            stream.start_stream()
            window['speechRec'].update(disabled=True)
            window['speechStop'].update(disabled=False)
        else:
            window.refresh()
            window['messages'].update('Message: Enter patient name')
            window.refresh()
    elif event == 'heartRec':
        p = pyaudio.PyAudio()
        window.refresh()
        window['messages'].update('Message: Recording heart')
        window.refresh()
        Recordframes = []
        patient_name = values['patientName']
        if patient_name != '':
            if not os.path.exists('database/{0}/'.format(patient_name)):
                os.makedirs('database/{0}/'.format(patient_name))
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, input_device_index = 1, frames_per_buffer=1024, stream_callback=callback)
            stream.start_stream()
            window['heartRec'].update(disabled=True)
            window['heartStop'].update(disabled=False)
        else:
            window.refresh()
            window['messages'].update('Message: Enter patient name')
            window.refresh()
    elif event == 'breathRec':
        p = pyaudio.PyAudio()
        window.refresh()
        window['messages'].update('Message: Recording breath')
        window.refresh()
        Recordframes = []
        patient_name = values['patientName']
        if patient_name != '':
            if not os.path.exists('database/{0}/'.format(patient_name)):
                os.makedirs('database/{0}/'.format(patient_name))
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, input_device_index = 1, frames_per_buffer=1024, stream_callback=callback)
            stream.start_stream()
            window['breathRec'].update(disabled=True)
            window['breathStop'].update(disabled=False)
        else:
            window.refresh()
            window['messages'].update('Message: Enter patient name')
            window.refresh()
        
    elif event == 'speechStop':
        stream.stop_stream()
        stream.close()
        p.terminate()
        window.refresh()
        window['messages'].update('Message:')
        window.refresh()
        patient_name = values['patientName']
        waveFile = wave.open('database/{0}/speechFile.wav'.format(patient_name), 'wb')
        waveFile.setnchannels(1)
        waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(44100)
        waveFile.writeframes(b''.join(Recordframes))
        window['speechRec'].update(disabled=False)
        window['speechStop'].update(disabled=True)
        
    elif event == 'heartStop':
        stream.stop_stream()
        stream.close()
        p.terminate()
        window.refresh()
        window['messages'].update('Message:')
        window.refresh()
        patient_name = values['patientName']
        waveFile = wave.open('database/{0}/heartFile.wav'.format(patient_name), 'wb')
        waveFile.setnchannels(1)
        waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(44100)
        waveFile.writeframes(b''.join(Recordframes))
        window['heartRec'].update(disabled=False)
        window['heartStop'].update(disabled=True)
        
    elif event == 'breathStop':
        stream.stop_stream()
        stream.close()
        p.terminate()
        window.refresh()
        window['messages'].update('Message:')
        window.refresh()
        patient_name = values['patientName']
        waveFile = wave.open('database/{0}/breathFile.wav'.format(patient_name), 'wb')
        waveFile.setnchannels(1)
        waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(44100)
        waveFile.writeframes(b''.join(Recordframes))
        window['breathRec'].update(disabled=False)
        window['breathStop'].update(disabled=True)
        
    elif event =='Done':
        window.refresh()
        window['messages'].update('Message: Performing speech analyses')
        window.refresh()
        speechStats = display_data('speech', patient_name)
        window['messages'].update('Message: Performing heart sound analyses')
        window.refresh()
        heartStats = display_data('heart', patient_name)
        window['messages'].update('Message: Performing lung sound analyses')
        window.refresh()
        breathStats = display_data('breath', patient_name)
        s = """window.onload = function() {
            document.getElementById('patientNameTitle').innerHTML = patientName + ' Report';
            document.getElementById('reportTitle').innerHTML = patientName + ' Report';
            document.getElementById('heartStats').innerHTML = heartStats;
            document.getElementById('breathStats').innerHTML = breathStats;
            document.getElementById('speechStats').innerHTML = speechStats;
        }"""
        contents = "var patientName = '" + patient_name + "';\n var heartStats = '" + heartStats + "';\n"
        contents += "var breathStats = '" + breathStats + "';\n"
        contents += "var speechStats = '" + speechStats + "';\n" + s
        outfile = open('database/'+patient_name+'/control.js', 'w')
        outfile.write(contents);
        outfile.close()
        temp = open('templates/index.html')
        outfile = open('database/'+patient_name+'/index.html', 'w')
        outfile.write(temp.read());
        outfile.close()
        temp.close()
        webbrowser.open('file://' + os.path.realpath('database/'+patient_name+'/index.html'))
        window['messages'].update('Message: Done!')
        window.refresh()
window.close()
