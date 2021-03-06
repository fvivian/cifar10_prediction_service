

import os
from flask import Flask, flash, request, redirect, url_for
from db_utils import write_data_to_table, predict_class

UPLOAD_FOLDER = '/tmp/upload/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)
            write_data_to_table(filename)
            return redirect(url_for('predict', filename=file.filename))
    return '''
    <!doctype html>
    <title>Upload new File</title>
    <h1>Upload new File</h1>
    <form method=post enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''

@app.route('/predict/<filename>')
def predict(filename):

    result, _ = predict_class(os.path.join(app.config['UPLOAD_FOLDER'], filename))

    return f'''
    <!doctype html>
    <title>Prediction</title>
    <h1>Predicted class: </h1>
    <div>{result}</div>
    '''

if __name__ == '__main__':
    app.run(host='0.0.0.0')

