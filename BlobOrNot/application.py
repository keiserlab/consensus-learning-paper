import os
from flask import Flask, render_template, request, redirect, url_for
from flask_sslify import SSLify

import boto3
import botocore
import pandas as pd
import numpy as np 

from database import db
from database.models import TileAnnotation


def get_url(bucket, key):
    return s3.generate_presigned_url(
        'get_object',
        Params={'Bucket': bucket, 'Key': key},
        ExpiresIn=3600)


def index():
    text = 'Blob or Not Labeling'
    return render_template('index.html', text=text)


def login():
    if request.method == 'POST':
        # todo: replace globals with something that works when running with multiple threads or processes
        # see: https://stackoverflow.com/questions/32815451/are-global-variables-thread-safe-in-flask-how-do-i-share-data-between-requests
        global successful_login, ind

        username = request.form['username']
        password_candidate = request.form['password']

        if username in password:
            if password_candidate == password[username]:
                successful_login[username] = True            
                ind[username] = len(TileAnnotation.query.filter_by(username=username).all())
                return redirect('/label/{}'.format(hash_password[username]))    
                        
            successful_login[username] = False
        
        text = 'Invalid login! Please try again.'
        return render_template('index.html', text=text)

    return render_template('login.html')


def finish(username_hash):
    username = [k for k,v in hash_password.items() if v==int(username_hash)]
    
    if username == []:
        return login()

    username = username[0]

    if username not in successful_login:
        return login()
    
    if not successful_login[username]:
        return login()

    return render_template('finish.html', username=hash_password[username],
                                thanks_picture=THANKS,
                                tile_labels=len(TileAnnotation.query.filter_by(username=username).all()))


def label(username_hash):
    # todo: replace globals with something that works when running with multiple threads or processes
    # see: https://stackoverflow.com/q/32815451
    global ind, successful_login, hash_password

    try:
        username = [k for k,v in hash_password.items() if v==int(username_hash)]
    except:
        return login()
    
    if username == []:
        return login()

    username = username[0]

    if username not in successful_login:
        return login()
    
    if not successful_login[username]:
        return login()

    phase = 'phase 2'
    tile_list = source_lists[username]['source_blob_list']
    raw_list = source_lists[username]['source_tile_list']
    label_list = source_lists[username]['source_label_list']
    rot_list = source_lists[username]['source_rot_list']
    levels = source_levels
    bucket = bucket_name

    # request method is POST when user takes action on the page
    if request.method=='POST':
        # If user quits, set login status to false, nullify the current password, and redirect to login page
        quit_app = request.values.getlist('quit')
        if quit_app==["quit"]:
            successful_login[username] = False
            hash_password[username] = hash(username+'1')
            return redirect('/login')

        # If user selects "undo" button, try deleting previous record, and redirect labeling html page to previous index
        undo = request.values.getlist("undo")
        if undo==["undo"]:
            undo = []
            if ind[username] > 0:
                last_record = TileAnnotation.query.filter_by(tilename=tile_list[ind[username]-1], username=username).first()
                if last_record == None:
                    print('no record')
                    ind[username] -= 1
                    return redirect('/label/{}'.format(hash_password[username]))
                try:
                    db.session.delete(last_record)
                    db.session.commit()        
                    db.session.close()
                    ind[username] -= 1
                except:
                    print('delete error')
                    db.session.rollback()
                    text = 'Database Error! Please contact us.'
                    return render_template('index.html', text=text)
                             
            return redirect('/label/{}'.format(hash_password[username]))

        # If user activates checkbox labels, save records
        tile_label = request.values.getlist("plaquelabel")
        if tile_label != []:
            record = TileAnnotation(tilename=tile_list[ind[username]], rot=rot_list[ind[username]], username=username)
            if 'cored' in tile_label:
                record.cored = True
            if 'diffuse' in tile_label:
                record.diffuse = True
            if 'CAA' in tile_label:
                record.CAA = True 
            if 'notsure' in tile_label:
                record.notsure = True 
            if 'flag' in tile_label:
                record.flag = True
            if 'negative' in tile_label:
                record.negative = True
            
            try:
                db.session.add(record)
                db.session.commit()        
                db.session.close()
                ind[username] += 1
            except:
                print('add error')
                db.session.rollback()
                text = 'Database Error! Please contact us.'
                return render_template('index.html', text=text)
            
            tile_label = []
        
        if ind[username] < len(tile_list):
            return redirect('/label/{}'.format(hash_password[username]))

        return redirect('/finish/{}'.format(hash_password[username]))
        
    # method=="GET"
    # The request method is GET when the annotator needs to see next example
    # If user index has completed all labels, redirect to finish
    if ind[username] >= len(tile_list):
        ind[username] = len(tile_list)
        return redirect('/finish/{}'.format(hash_password[username]))

    # Set user level by index
    for i, l in enumerate(levels):
        if ind[username] < l:
            level = i
            break
        level = len(levels)

    # Generate user level-associated text
    if level==len(levels):
        label_text_up = "Awesome! You are No.1 in Blob or Not!"
        percent = round(ind[username]*100/len(tile_list), 2)
        remains = len(tile_list) - ind[username]
        label_text_down = "{} images to finish!".format(remains)
    else:
        label_text_up = "Current Level: level{}.".format(level)
        percent = round(ind[username]*100/levels[level],2)
        remains = levels[level] - ind[username]
        label_text_down = "{} images to level{}".format(remains, level+1)

    # Format rotation and box coordinates for images
    rot = int(rot_list[ind[username]])
    img_coord = [int(c) for c in label_list[ind[username]][1:-1].split(' ') if c]
    label = {"id":1, "name":"", "xMin":img_coord[0], "xMax":img_coord[0]+img_coord[2], "yMin":img_coord[1], "yMax":img_coord[1]+img_coord[3], "rot":rot}
    # Redirect user to the labeling html page with next example
    return render_template('label.html', username=hash_password[username],
                            phase=phase,
                            level_name='level{}.png'.format(str(level)),
                            raw_path=get_url(bucket, raw_list[ind[username]]), 
                            normalized_path=get_url(bucket, tile_list[ind[username]]), 
                            percent=percent,
                            ind=ind[username],
                            total=len(tile_list),
                            label=label,
                            labels_up=label_text_up,
                            labels_down=label_text_down)


# Start application (EB looks for an 'application' callable by default)
application = app = Flask('blobornot')
# Config file config file is imported
application.config.from_object('config')
# redirect incoming request to HTTPS
if app.config['RUN_CLOUD']:
    sslify = SSLify(app)

# connect to database where annotations are stored
try:
    db.init_app(application)
    db.create_all()  # create db table if necessary
except:
    print('db error')
    raise

# Connection to AWS S3 where images are stored
session = boto3.session.Session(
    aws_access_key_id=app.config['S3_ACCESS_KEY_ID'],
    aws_secret_access_key=app.config['S3_SECRET_KEY'],
    region_name=app.config['S3_REGION_NAME'])
s3 = session.client('s3')

# Populate user credentials
password = np.load(app.config['PASSWORD_FILE']).item()
hash_password = {user: hash(user) for user in password}
successful_login = {user: False for user in password}
# Find number of annotations by user and index of next example
ind = {username: len(TileAnnotation.query.filter_by(username=username).all())
       for username in password}

# CSV images to pull from S3 bucket
bucket_name = app.config['S3_BUCKET_NAME']
source_lists = {}
for user in password:
    # to show different examples to annotators, load a different csv for each user
    # source_file = pd.read_csv(app.config['EXAMPLES_CSV'].format(user))
    source_file = pd.read_csv(app.config['EXAMPLES_CSV'])
    source_lists[user] = dict(
        source_blob_list=list(source_file[app.config['CSV_BLOB_COL']]),
        source_tile_list=list(source_file[app.config['CSV_TILE_COL']]),
        source_label_list=list(source_file[app.config['CSV_COORDS_COL']]),
        source_rot_list=list(source_file[app.config['CSV_ROTATION_COL']])
    )
# Number of examples needed to reach each level (gamification)
# Note: Each level has an image in static directory
source_levels = [10, 100, 500, 1000, 1500, 3000, 5000, 7000, 10000]
# Image shown when complete - should be in static directory
THANKS = 'thanks.gif'

# Register each page to a view function
app.add_url_rule('/', 'index', index)
app.add_url_rule('/label/<username_hash>', 'label', label, methods=['POST', 'GET'])
app.add_url_rule('/finish/<username_hash>', 'finish', finish, methods=['POST', 'GET'])
app.add_url_rule('/login', 'login', login, methods=['POST', 'GET'])

# Debug/Run application
if __name__ == "__main__":
    app.debug = False
    app.run(port=4000)
