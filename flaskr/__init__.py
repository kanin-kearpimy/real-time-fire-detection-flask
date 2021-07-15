import os
from .ML import Retinanet

from flask import Flask, Response, send_from_directory, render_template

def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True, static_folder="static")
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'flaskr.sqlite')
    )
    # start_retinanet = Retinanet('firemodel_conveted.h5', 'resnet101', {0: 'fail'}, "flaskr/static/video/test-fire.mp4")

    if(test_config == None):
        app.config.from_pyfile('config.py', silent=True)
    else:
        app.config.from_mapping(test_config)
    
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/video_feed')
    def video_feed():
        return Response(gen_frames(), mimetype="multipart/x-mixed-replace;boundary=frame")
    
    @app.route('/')
    def story():
        return render_template('dashboard.html')
        # return render_template('story.html')

    def gen_frames():  # generate frame by frame from camera
        # frame = ImageProcessing("flaskr/static/video/fire_video.mp4").start()
        # retinanet = Retinanet('firemodel_conveted.h5', 'resnet101', {0: 'fail'}, "flaskr/static/video/a.mp4").start()
        retinanet = start_retinanet.start()
        while True:
            if(retinanet.frame_to_show != None):
                modified_frame = retinanet.frame_to_show
                yield (b'--frame\r\n'
                    b'Content-Type: image/jpeg\r\n\r\n' + modified_frame + b'\r\n')  # concat frame one by one and show result


    return app