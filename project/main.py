import sys
sys.path.append('../project/yolo')
import os
from app import app
from flask import flash, request, redirect, url_for, render_template
import cv2
from yolo.models import Yolov4

ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def inference_model(filepath):
    model = Yolov4(weight_path='data/yolov4.weights',
                   class_name_path='yolo/class_names/coco_classes.txt')
    raw_img = cv2.imread(filepath)[:, :, ::-1]
    (output_img, detections) = model.predict_img(raw_img, plot_img=False, return_output=True)
    cv2.imwrite(filepath, output_img)
    return filepath


def detect_objects(file):
    source_filename = file.filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], source_filename)
    file.save(filepath)

    img = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    scale_percent = 60  # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (app.config['IMG_WIDTH'], app.config['IMG_HEIGHT'])

    # resize image
    # resized = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
    new_filename = "resized_" + source_filename
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], new_filename)
    cv2.imwrite(filepath, img)
    target_filename = inference_model(filepath)
    head, tail = os.path.split(target_filename)
    return tail

@app.route('/')
def upload_form():
    return render_template('upload.html')


@app.route('/', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = detect_objects(file)
        flash('Image successfully uploaded and displayed below')
        return render_template('upload.html', filename=filename)
    else:
        flash('Allowed image types are -> png, jpg, jpeg, gif')
        return redirect(request.url)


@app.route('/display/<filename>')
def display_image(filename):
    # print('display_image filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)


if __name__ == "__main__":
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)
