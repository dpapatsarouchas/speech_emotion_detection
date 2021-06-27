from flask import Flask, request

app = Flask(__name__, static_folder='./', static_url_path='/')


@app.errorhandler(404)
def not_found(e):
    return app.send_static_file('index.html')


@app.route('/')
def index():
    return "test"

@app.route('/api/save_audio', methods=["POST"])
def save_audio():
    if request.method == "POST":
        f = request.files['audio_data']
        f.save('../samples/audio.wav')

        # with open('../samples/audio.wav', 'w') as audio:
        #     sf.write("../samples/audio.wav", f, 44100)
        #     f.save(audio)
        # print('file uploaded successfully')

        # Wave
        # print(f)
        # audio = wave.open("../samples/audio.mp3", 'wb')
        # audio.writeframes(f)

        return {"Status": "Success"}
    else:
        return {"Status": "Failure"}