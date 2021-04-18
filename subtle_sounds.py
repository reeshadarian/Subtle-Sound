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

sg.theme('DefaultNoMoreNagging')
leftcol = [ [sg.Text('Patient Name', font=(30)), sg.InputText(key = 'patientName'), sg.Button("Search Patient", key='patientNameSearchButton')],           
            
            [sg.Text('_'*70)],
            [sg.Text('Heart', font=(30))],
            
            [sg.Button('Capture Heart (A)', key='heartRecA'), sg.Button('Stop', disabled=True, key='heartStopA')],
            [sg.Button('Capture Heart (P)', key='heartRecP'), sg.Button('Stop', disabled=True, key='heartStopP')],
            [sg.Button('Capture Heart (T)', key='heartRecT'), sg.Button('Stop', disabled=True, key='heartStopT')],
            [sg.Button('Capture Heart (M)', key='heartRecM'), sg.Button('Stop', disabled=True, key='heartStopM')],
            [sg.Button('Capture Heart (E)', key='heartRecE'), sg.Button('Stop', disabled=True, key='heartStopE')],
            
            [sg.Text('_'*70)],
            [sg.Text('Lung', font=(30))],
            [sg.Button('Capture Breath (L1)', key='breathRecL1'), sg.Button('Stop', disabled=True, key='breathStopL1')],
            [sg.Button('Capture Breath (R1)', key='breathRecR1'), sg.Button('Stop', disabled=True, key='breathStopR1')],
            [sg.Button('Capture Breath (L2)', key='breathRecL2'), sg.Button('Stop', disabled=True, key='breathStopL2')],
            [sg.Button('Capture Breath (R2)', key='breathRecR2'), sg.Button('Stop', disabled=True, key='breathStopR2')],
            [sg.Button('Capture Breath (L3)', key='breathRecL3'), sg.Button('Stop', disabled=True, key='breathStopL3')],
            [sg.Button('Capture Breath (R3)', key='breathRecR3'), sg.Button('Stop', disabled=True, key='breathStopR3')],
            [sg.Button('Capture Breath (L4)', key='breathRecL4'), sg.Button('Stop', disabled=True, key='breathStopL4')],
            [sg.Button('Capture Breath (R4)', key='breathRecR4'), sg.Button('Stop', disabled=True, key='breathStopR4')],
            [sg.Button('Capture Breath (L5)', key='breathRecL5'), sg.Button('Stop', disabled=True, key='breathStopL5')],
            [sg.Button('Capture Breath (R5)', key='breathRecR5'), sg.Button('Stop', disabled=True, key='breathStopR5')],
            [sg.Button('Capture Breath (L6)', key='breathRecL6'), sg.Button('Stop', disabled=True, key='breathStopL6')],
            [sg.Button('Capture Breath (R6)', key='breathRecR6'), sg.Button('Stop', disabled=True, key='breathStopR6')],
            
            [sg.Text('_'*70)],
            [sg.Text('Speech', font=(30))],
            [sg.Button('Capture Speech', key='speechRec'), sg.Button('Stop', disabled=True, key='speechStop')],
            
            [sg.Text('_'*70)],
            [sg.Text("Message:", size = (50, 1), key='messages', font=(30))],
            [sg.Button('Done'), sg.Button('Cancel')]]

rightcol = [[sg.Image('./heart.png')],
            [sg.Image('./lung.png')]]

layout = [[sg.Column(leftcol), sg.VSeperator(),sg.Column(rightcol, element_justification='c')]]

window = sg.Window('Subtle Sounds', layout, finalize=True)

p = pyaudio.PyAudio()
info = p.get_host_api_info_by_index(0)
numdevices = info.get('deviceCount')
mics_idx = []
stetho_idx = 0
for i in range(0, numdevices):
        if (p.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
            if ("AudioBox" in p.get_device_info_by_host_api_device_index(0, i).get('name')):
                mics_idx.append(i)
            if ("Microphone (Realtek" in p.get_device_info_by_host_api_device_index(0, i).get('name')):
                stetho_idx = i
p.terminate()

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
    plt.title("Amplitude")
    plt.savefig(file_path+signal_type+"Plot.png", dpi = 1200)
    plt.close()
    
    y, sr = librosa.load(filename)
    librosa.display.waveplot(y, sr=sr);
    hop_length = 512
    D = np.abs(librosa.stft(y, n_fft=2048,  hop_length=128))
    DB = librosa.amplitude_to_db(D, ref=np.max)
    librosa.display.specshow(DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='log');
    plt.colorbar(format='%+2.0f dB')
    plt.title("Spectogram")
    plt.savefig(file_path+signal_type+"Spectogram.png", dpi = 1200)
    plt.close()

def measurePitch(file_name, f0min = 20, f0max = 1000, unit = 'Hertz', sound_type = None):
    voiceID = parselmouth.Sound(file_name)
    sound = parselmouth.Sound(voiceID) # read the sound
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 20, 0.1, 1.0)
    window.refresh()
    hnr = call(harmonicity, "Get mean", 0, 0)
    window.refresh()
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    window.refresh()
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    window.refresh()
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    window.refresh()
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    window.refresh()
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    window.refresh()
    return hnr, localJitter, localabsoluteJitter, localShimmer, localdbShimmer

def shortTermAnalyses(sound_type, filename, patient_name):
    fs, signal = wavfile.read(filename)
    window.refresh()
    if sound_type == 'speech':
        s = audioSegmentation.silence_removal(signal, fs, 0.5, 0.1, weight=0.2)
        signal2 = np.concatenate([signal[int((i[0]+0.1)*fs):int((i[1]+0.1)*fs)] for i in s])
        wavfile.write("database/{0}/speechFileSegmented.wav".format(patient_name), fs, signal2)
        s1 = ShortTermFeatures.feature_extraction(signal[:, 0], fs, 0.05*fs, 0.025*fs, deltas=True)[0][:8]
        window.refresh()
        s2 = ShortTermFeatures.feature_extraction(signal[:, 1], fs, 0.05*fs, 0.025*fs, deltas=True)[0][:8]
        window.refresh()
        filename = filename[:-4] + "1.wav"
        fs, signal = wavfile.read(filename)
        s = audioSegmentation.silence_removal(signal, fs, 0.5, 0.1, weight=0.2)
        signal2 = np.concatenate([signal[int((i[0]+0.1)*fs):int((i[1]+0.1)*fs)] for i in s])
        wavfile.write("database/{0}/speechFileSegmented1.wav".format(patient_name), fs, signal2)
        s3 = ShortTermFeatures.feature_extraction(signal[:, 0], fs, 0.05*fs, 0.025*fs, deltas=True)[0][:8]
        window.refresh()
        s4 = ShortTermFeatures.feature_extraction(signal[:, 1], fs, 0.05*fs, 0.025*fs, deltas=True)[0][:8]
        window.refresh()
        n = min(s1.shape[0], s2.shape[0], s3.shape[0], s4.shape[0])
        m = min(s1.shape[1], s2.shape[1], s3.shape[1], s4.shape[1])
        return (s1[:n, :m]+s2[:n, :m]+s3[:n, :m]+s4[:n, :m])/4
    else:
        return ShortTermFeatures.feature_extraction(signal, fs, 0.05*fs, 0.025*fs, deltas=True)[0][:8]

def display_data(signal_type,  patient_name):
    res = ''
    file_path = 'database/' + patient_name + '/'
    filename = file_path + signal_type + 'File.wav' 
    if 'heart' in signal_type:
        title = 'Heart ({0}) Data'.format(signal_type[-1])
        
    elif signal_type == 'speech':
        title = 'Speech Data'
        segmented = file_path + signal_type + 'FileSegmented.wav'
        segmented1 = file_path + signal_type + 'FileSegmented1.wav'
    else:
        title = 'Breathing ({0}) Data'.format(signal_type[-2:])
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
        window.refresh()
        with speech_recognition.WavFile(filename) as source:              # use "test.wav" as the audio source
            audio = recog.record(source)                        # extract audio data from the file
            try:
                res += 'Transcript: ' + recog.recognize_google(audio) +'<br>'
            except:                                 # speech is unintelligible
                res += 'Transcript: Could not understand audio' + '<br>'
        window.refresh()
        file.close()
        
    data = shortTermAnalyses(signal_type, filename, patient_name)
    names = ['ZCR', 'Short Time Energy', 'Energy Entropy', 'Spectral Centroid', 'Spectral Spread', 'Spectral Entropy', 'Spectral Flux', 'Spectral Rolloff']
    for i in range(8):
        plt.plot(np.linspace(0, len(yy)/44100, len(data[i])), data[i])
        plt.title("{0}, Mean = {1:.3e}".format(names[i], np.mean(data[i])))
        plt.savefig(file_path+signal_type+names[i]+'.png', dpi = 1200)
        plt.close()
    if signal_type == "speech":
        hnr, localJitter, localabsoluteJitter, localShimmer, localdbShimmer = measurePitch(segmented, sound_type='speech')
        hnr1, localJitter1, localabsoluteJitter1, localShimmer1, localdbShimmer1 = measurePitch(segmented, sound_type='speech')
        hnr, localJitter, localabsoluteJitter, localShimmer, localdbShimmer = hnr + hnr1, localJitter + localJitter1, localabsoluteJitter + localabsoluteJitter1, localShimmer + localShimmer1, localdbShimmer + localdbShimmer1
    else:
        hnr, localJitter, localabsoluteJitter, localShimmer, localdbShimmer = measurePitch(filename)
    res += 'HNR: ' + str(hnr) + '<br>'
    res += 'Local Jitter: ' + str(localJitter) + '<br>'
    res += 'Local Absolute Jitter: ' + str(localabsoluteJitter) + '<br>'
    res += 'Local Shimmer: ' + str(localShimmer) + '<br>'
    res += 'Local Shimmer dB: ' + str(localdbShimmer) + '<br>'
    return res

def callback(in_data, frame_count, time_info, status):
    Recordframes.append(in_data)
    return (in_data, pyaudio.paContinue)

def callback1(in_data1, frame_count1, time_info1, status1):
    Recordframes1.append(in_data1)
    return (in_data1, pyaudio.paContinue)

while True:
    event, values = window.read()
    if event == sg.WIN_CLOSED or event == 'Cancel': # if user closes window or clicks cancel
        break
    elif event == 'patientNameSearchButton':
        patient_name = values['patientName']
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
        Recordframes1 = []
        patient_name = values['patientName']
        if patient_name != '':
            if not os.path.exists('database/{0}/'.format(patient_name)):
                os.makedirs('database/{0}/'.format(patient_name))
            stream = p.open(format=pyaudio.paInt16, channels=2, rate=44100, input=True, input_device_index = mics_idx[0], frames_per_buffer=1024, stream_callback=callback)
            stream.start_stream()
            stream1 = p.open(format=pyaudio.paInt16, channels=2, rate=44100, input=True, input_device_index = mics_idx[1], frames_per_buffer=1024, stream_callback=callback1)
            stream1.start_stream()
            window['speechRec'].update(disabled=True)
            window['speechStop'].update(disabled=False)
        else:
            window.refresh()
            window['messages'].update('Message: Enter patient name')
            window.refresh()
            
    elif 'heartRec' in event:
        p = pyaudio.PyAudio()
        window.refresh()
        window['messages'].update('Message: Recording heart')
        window.refresh()
        Recordframes = []
        patient_name = values['patientName']
        if patient_name != '':
            if not os.path.exists('database/{0}/'.format(patient_name)):
                os.makedirs('database/{0}/'.format(patient_name))
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, input_device_index = stetho_idx, frames_per_buffer=1024, stream_callback=callback)
            stream.start_stream()
            window[event].update(disabled=True)
            window['heartStop'+event[8:]].update(disabled=False)
        else:
            window.refresh()
            window['messages'].update('Message: Enter patient name')
            window.refresh()
            
    elif 'breathRec' in event:
        p = pyaudio.PyAudio()
        window.refresh()
        window['messages'].update('Message: Recording breath')
        window.refresh()
        Recordframes = []
        patient_name = values['patientName']
        if patient_name != '':
            if not os.path.exists('database/{0}/'.format(patient_name)):
                os.makedirs('database/{0}/'.format(patient_name))
            stream = p.open(format=pyaudio.paInt16, channels=1, rate=44100, input=True, input_device_index = stetho_idx, frames_per_buffer=1024, stream_callback=callback)
            stream.start_stream()
            window[event].update(disabled=True)
            window['breathStop'+event[9:]].update(disabled=False)
        else:
            window.refresh()
            window['messages'].update('Message: Enter patient name')
            window.refresh()
        
    elif event == 'speechStop':
        stream.stop_stream()
        stream.close()
        stream1.stop_stream()
        stream1.close()
        p.terminate()
        window.refresh()
        window['messages'].update('Message:')
        window.refresh()
        patient_name = values['patientName']
        waveFile = wave.open('database/{0}/speechFile.wav'.format(patient_name), 'wb')
        waveFile.setnchannels(2)
        waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(44100)
        waveFile.writeframes(b''.join(Recordframes))
        waveFile1 = wave.open('database/{0}/speechFile1.wav'.format(patient_name), 'wb')
        waveFile1.setnchannels(2)
        waveFile1.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        waveFile1.setframerate(44100)
        waveFile1.writeframes(b''.join(Recordframes1))
        window['speechRec'].update(disabled=False)
        window['speechStop'].update(disabled=True)
        
    elif 'heartStop' in event:
        stream.stop_stream()
        stream.close()
        p.terminate()
        window.refresh()
        window['messages'].update('Message:')
        window.refresh()
        patient_name = values['patientName']
        waveFile = wave.open('database/{0}/heart{1}File.wav'.format(patient_name, event[9:]), 'wb')
        waveFile.setnchannels(1)
        waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(44100)
        waveFile.writeframes(b''.join(Recordframes))
        window['heartRec'+event[9:]].update(disabled=False)
        window[event].update(disabled=True)
        
    elif 'breathStop' in event:
        stream.stop_stream()
        stream.close()
        p.terminate()
        window.refresh()
        window['messages'].update('Message:')
        window.refresh()
        patient_name = values['patientName']
        waveFile = wave.open('database/{0}/breath{1}File.wav'.format(patient_name, event[10:]), 'wb')
        waveFile.setnchannels(1)
        waveFile.setsampwidth(p.get_sample_size(pyaudio.paInt16))
        waveFile.setframerate(44100)
        waveFile.writeframes(b''.join(Recordframes))
        window['breathRec'+event[10:]].update(disabled=False)
        window[event].update(disabled=True)
        
    elif event =='Done':
        patient_name = values['patientName']
        s = """window.onload = function() {
            document.getElementById('patientNameTitle').innerHTML = patientName + ' Report';
            document.getElementById('reportTitle').innerHTML = patientName + ' Report';
            document.getElementById('speechStats').innerHTML = speechStats;
            
        """
        
        contents = "var patientName = '" + patient_name + "';\n"
        
        window.refresh()
        window['messages'].update('Message: Performing speech analyses')
        window.refresh()
        speechStats = display_data('speech', patient_name)
        
        window.refresh()
        for i in ['A', 'P', 'T', 'M', 'E']:
            window['messages'].update('Message: Performing Heart ({0}) sound analyses'.format(i))
            window.refresh()
            s += "document.getElementById('heart{0}Stats').innerHTML = heart{0}Stats;\n".format(i)
            contents += "var heart{0}Stats = '".format(i) + display_data('heart'+i, patient_name) + "';\n"
        
        window.refresh()
        for i in ['L', 'R']:
            for j in range(1, 7):
                window['messages'].update('Message: Performing Lung ({0}) sound analyses'.format(i+str(j)))
                window.refresh()
                s += "document.getElementById('breath{0}Stats').innerHTML = breath{0}Stats;\n".format(i+str(j))
                contents += "var breath{0}Stats = '".format(i+str(j)) + display_data('breath'+i+str(j), patient_name) + "';\n"
        
                
        contents += "var speechStats = '" + speechStats + "';\n" + s + "}"
        
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