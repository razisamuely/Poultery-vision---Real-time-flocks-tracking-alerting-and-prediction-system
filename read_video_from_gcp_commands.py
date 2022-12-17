
import os
from google.cloud import storage
import time
from videoToTable import videoToTable
from google.cloud import storage
import os

# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/raz.shmuely/Documents/privet/chickens/production-project/object_detection_project/TFODCourse/keys/service-keys.json'
storage_client = storage.Client()

''''
crate new bucket
'''
bucket_name = 'data_bucket_test_razi'




def upload_to_bucket(blob_name, file_path, bucket_name):
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        blob.upload_from_filename(file_path)
        return True
    except Exception as e:
        print(e)
        return


file_path = "/TensorFlow/workspace/videos/"
# upload_to_bucket('first_video_uploading', os.path.join(file_path, 'Drinking-slow-motion.mp4'), 'data_bucket_test_razi')


'''
Download files files
'''


def download_file_from_bucket(blob_name, file_path, bucket_name):
    try:
        bucket = storage_client.get_bucket(bucket_name)
        blob = bucket.blob(blob_name)
        with open(file_path,'wb') as f:
            storage_client.download_blob_to_file(blob,f)
        return True
    except Exception as e:

        print(e)
        return

# download_file_from_bucket('first_video_uploading', os.path.join(os.getcwd(),'file1.mp4'), bucket_name)


import firebase_admin
from firebase_admin import credentials
from firebase_admin import storage
import datetime

import urllib.request as req
import cv2

cred = credentials.Certificate('keys/service-keys.json')
app = firebase_admin.initialize_app(cred, {'storageBucket': 'data_bucket_test_razi'}, name='storage')
bucket = storage.bucket(app=app)

def generate_image_url(blob_path):
    """ generate signed URL of a video stored on google storage.
        Valid for 300 seconds in this case. You can increase this
        time as per your requirement.
    """
    blob = bucket.blob(blob_path)
    return blob.generate_signed_url(datetime.timedelta(seconds=300), method='GET')


url = generate_image_url('first_video_uploading')
req.urlretrieve(url, "first_video_uploading")
cap = cv2.VideoCapture('first_video_uploading')

if cap.isOpened():
    print ("File Can be Opened")
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        #print cap.isOpened(), ret
        if frame is not None:
            # Display the resulting frame
            cv2.imshow('frame',frame)
            # Press q to close the video windows before it ends if you want
            if cv2.waitKey(22) & 0xFF == ord('q'):
                break
        else:
            print("Frame is None")
            break
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    print ("Video stop")
else:
    print("Not Working")



video_path = ''
credentials_path = 'keys/service-keys.json'
video_bucket_path = 'hens-videos'
video_name = 'first_video_uploading'



os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/raz.shmuely/Documents/privet/chickens/production-project/object_detection_project/TFODCourse/keys/service-keys.json"

def list_blobs_gcp(bucket_name):
    """Lists all the blobs in the bucket."""
    # bucket_name = "your-bucket-name"

    storage_client = storage.Client()

    # Note: Client.list_blobs requires at least package version 1.17.0.
    blobs = storage_client.list_blobs(bucket_name)
    for i in blobs:
        print(i.name)

# list_blobs_gcp(bucket_name = video_bucket_path)

## TODO 1) new_video_to_old_video  - Read new videos list directly from bucket
## TODO 2) tf_object detector 'video_detection_to_counter_gcp' funciton , detect directly from cloud
## TODO 3) new_video_to_old_video - Delete old videos from gcp