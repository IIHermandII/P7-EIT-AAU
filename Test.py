def test2():
    import librosa
    import soundfile as sf
    import numpy as np
    sr = 22050  # Sample rate
    t = np.linspace(0, 1, sr)
    x = 0.5 * np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave

    # Save to a file
    sf.write('test.wav', x, sr)

    # Load the file and check if it works
    y, sr_loaded = librosa.load('test.wav', sr=None)
    print("Loaded audio with sample rate:", sr_loaded)

def test1():
    from tqdm import tqdm, trange
    import time
    for i in trange(10):
        time.sleep(0.1)
        print("H")
    print('done')

def main():
    test1()


if __name__ == "__main__":
    main()
