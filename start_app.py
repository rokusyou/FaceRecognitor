
#Flask などの必要なライブラリをインポートする
import os
from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from werkzeug import secure_filename
from extract_face import extract_face

# 自身の名称を app という名前でインスタンス化する
app = Flask(__name__)

# Settings
UPLOAD_FOLDER = './uploads'
FACE_FOLDER = './uploads/face'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'gif'])
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['SECRET_KEY'] = os.urandom(24)
app.config['FACE_FOLDER'] = FACE_FOLDER

# メッセージをランダムに表示するメソッド
def picked_up():
    messages = [
        "こんにちは、あなたの名前を入力してください",
        "やあ！お名前は何ですか？",
        "あなたの名前を教えてね"
    ]
    # NumPy の random.choice で配列からランダムに取り出し
    return np.random.choice(messages)


def allowed_file(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# ここからウェブアプリケーション用のルーティングを記述
# index にアクセスしたときの処理
@app.route('/')
def index():
    title = "Webapp with Flask"
    message = picked_up()
    # index.html をレンダリングする
    return render_template('index.html',
                           message=message, title=title)

@app.route('/face_rg')
def face_rg():
    title = "Face Reccoginitor"
    message = picked_up()
    # face_rg.html をレンダリングする
    return render_template('face_rg.html',
                           message='Upload an Image', title=title)
@app.route('/send',methods=['GET','POST'])
def send():
    if request.method == 'POST':
        img_file = request.files['img_file']
        if img_file and allowed_file(img_file.filename):
            filename = secure_filename(img_file.filename)
            img = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            img_file.save(img)
            # face extract
            face_img = os.path.join(app.config['FACE_FOLDER'], filename)
            extract_face(img,face_img,64)
            img_url = '/uploads/' + filename
            return render_template('index.html', img_url=img_url)
    else:
        return redirect(url_for('index'))

# /post にアクセスしたときの処理
@app.route('/post', methods=['GET', 'POST'])
def post():
    title = "こんにちは"
    if request.method == 'POST':
        # リクエストフォームから「名前」を取得して
        name = request.form['name']
        # index.html をレンダリングする
        return render_template('index.html',
                               name=name, title=title)
    else:
        # エラーなどでリダイレクトしたい場合はこんな感じで
        return redirect(url_for('index'))

if __name__ == '__main__':
    app.debug = True # デバッグモード有効化
    #app.run(host='0.0.0.0') # どこからでもアクセス可能に
    app.run() # どこからでもアクセス可能に


@app.route('/upload', methods=['POST'])
def upload():
    """
    f = open('sample.json')
    json = f.read()
    f.close()
    return Response(response=json, status=200, mimetype="application/json")
    """
    f = request.files['file']
    filename = secure_filename(f.filename)
    (fn, ext) = os.path.splitext(filename)
    input_path = '/tmp/' + uuid.uuid1().hex + ext
    print(input_path)
    f.save(input_path)
    
    faces = detect.detect_face_rotate(input_path, web_dir, 'static/tmp')
    print(faces)
    
    res = mcz_eval.execute(faces, web_dir, deeplearning_dir + '/data/model.ckpt-13000_85per_input56_conv3_fc2')
    
    return jsonify({'results':res})
    


