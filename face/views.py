from django.shortcuts import render

from .facenet import *
from django.http import StreamingHttpResponse,HttpResponseServerError,HttpResponse
from django.views.decorators import gzip
from .models import Travel_history,St_face
import cv2
import json
import requests

import urllib.request

from bs4 import BeautifulSoup



def database1():
    database = {}
    # load all the images of individuals to recognize into the database
    i = 0
    for fa in list(St_face.objects.all().values_list('face').order_by('student_who_id')):
        st = list(St_face.objects.all().values_list('student_who_id').order_by('student_who_id'))[i][0]
        i+=1
        q = str.split(str(fa[0]).replace('[[', '').replace(']]', '').replace('\n', '').replace('\r', ''), " ")
        q = np.array(q)
        q = np.delete(q, np.where(q == '')).astype("float32")
        database[st] = q
    return database

class VideoCamera(object):

    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self):
        ret,image = self.video.read()
        return image




def gen(camera):
    global database
    global name
    c=0
    while True:
        frame = camera.get_frame()
        if np.array(process_frame1(frame)).any() != None:
            c+=1
            frame, name = webcam_face_recognizer(database, frame)
        ret,jpeg = cv2.imencode('.jpg',frame)
        frame = jpeg.tobytes()
        yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')
        if name != None:
            break
        if c > 20:
            name = "此人沒有在臉部資料庫"
            break

def gen2(camera,id):
    global take
    take = None
    tStart = time.time()
    while True:
        frame = camera.get_frame()
        if (time.time() - tStart) > 3:
            if np.array(process_frame1(frame)).any()!=None:
                take = "建模完成"
        ret,jpeg = cv2.imencode('.jpg',frame)
        frame1 = jpeg.tobytes()
        yield(b'--frame\r\n'
                  b'Content-Type: image/jpeg\r\n\r\n' + frame1 + b'\r\n\r\n')
        if take != None:
            face_ex = prepare_database1(frame)
            user = Travel_history.objects.only('student_id').get(student_id=id)
            St_face.objects.create(face=face_ex, student_who=user)
            break

@gzip.gzip_page
def facesteam(request):
    try:
        res = StreamingHttpResponse(gen(VideoCamera()), content_type="multipart/x-mixed-replace;boundary=frame")
        return res
    except HttpResponseServerError as e:
        print("aborted")
def namesteam(request):
    global name
    name = None
    while True:
        if str(name) != "None":
            res = HttpResponse(str(name))
            return res

def takefinish(request):
    global take
    take = None
    while True:
        if str(take) != "None":
            res = HttpResponse(str(take))
            return res
def facescreate(request):

    try:
        res = StreamingHttpResponse(gen2(VideoCamera(),id), content_type="multipart/x-mixed-replace;boundary=frame")
        return res
    except HttpResponseServerError as e:
        print("aborted")

# Create your views here.
def hello_view(request):
    global database
    database = database1()
    th = list(Travel_history.objects.all().values_list('student_id','history_3m'))
    list1, list2 = zip(*th)
    return  render(request, 'index.html', {
        'data1':json.dumps(list1),
        'data2': json.dumps(list2),
    })


def create_view(request):
    global id
    id = request.GET.get("stid")
    c = request.GET.get("c")
    f=None
    if c =="c":
        f = 'f'
    th = Travel_history.objects.all().values()
    return  render(request, 'create.html', {
        'data3':th,
        "f":f,
    })

def search_bar(request):
    q = request.GET.get("q")
    q = str.split(q," ")
    st = Travel_history.objects.all()
    st = st.filter(student_id__icontains=q[0])
    if len(list(st))==0:
        a ='此人無資料在健康中心'
        id = q[0]
    else:
        a = "此人有健康資料，前往建立臉部資料"
        id = q[0]
    return render(request,
                  'create.html',{
                       'data4': a,
                        'st_id':id,
                       })

