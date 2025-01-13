import numpy as np
import pickle
import glob
import cv2
from flask import Flask,render_template,request,jsonify
import base64
import io
from PIL import Image

app = Flask(__name__)

#配列をフラットにする
def list_flat(arr):
    flat_arr=[]
    for i in range(len(arr)):
        for j in range(len(arr[i])):
            flat_arr.append(arr[i][j])
    return flat_arr


def make_data(img):

  def make_poly(arr,deg):
    poly = []
    for i in range(deg):
      for j in range(len(arr)):
        poly.append(arr[j]**(i+1))
    return poly

  # 色空間の変換
  rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
  lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

  # 平均値を計算
  rgb = np.mean(np.mean(rgb_img,axis=1),axis=0)
  hsv = np.mean(np.mean(hsv_img,axis=1),axis=0)
  lab = np.mean(np.mean(lab_img,axis=1),axis=0)

  r_MEAN = 106.87626803684113
  r_STD = 79.18476934463334
  h_MEAN = 53.69784646955657
  h_STD = 38.6981975345093
  b_MEAN = 154.32425094322426
  b_STD = 27.213700253049293

  r = (rgb[0] - r_MEAN)/r_STD
  h = (hsv[0] - h_MEAN)/h_STD
  b = (lab[2] - b_MEAN)/b_STD

  reg_data = make_poly([r,h,b],5)

  svm_data =[rgb]
  reg_data = np.array([reg_data])

  return svm_data, reg_data

def editImg(base64_img):
    #base64をそのまま渡すとできないみたいなのでsplitする
    dec_data = base64.b64decode( base64_img.split(',')[1] )

    #バイナリーストリーム <- バリナリデータ
    img_binarystream = io.BytesIO(dec_data)

    #PILイメージ <- バイナリーストリーム
    img_pil = Image.open(img_binarystream)
    # print(img_pil.mode) #この段階だとRGBA

    #numpy配列(RGBA) <- PILイメージ
    img_numpy = np.asarray(img_pil)

    #numpy配列(BGR) <- numpy配列(RGBA)
    img_numpy_bgr = cv2.cvtColor(img_numpy, cv2.COLOR_RGBA2BGR)
    # print(img_numpy_bgr.shape)

    #
    # result,_ = cv2.imencode('.jpg', img_numpy_bgr)
    # trim_base64 = base64.b64encode(trim_data)

    #表示確認
    # cv2.imshow('window title', img_numpy_bgr_trim)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    #画像のパス >> 直接指定する場合
    # pH_img = "1.jpg"
    # target_path = r"C:\Users\2018Student_NO3\Desktop\pH_test"+"/"+pH_img

    # flat_feature = make_data(img_numpy_bgr_trim)


    return img_numpy_bgr

def predictpH(flat_feature_poly):
    filename = "RHb5_model.sav"
    loaded_model = pickle.load(open(filename, 'rb'))
    pH_predict = round(loaded_model.predict(flat_feature_poly)[0],2)
    return pH_predict

def clasifypH(clasification_feature):
   filename = "RGB_SVM_model.sav"
   loaded_model = pickle.load(open(filename, "rb"))
   clasificate_pH = round(loaded_model.predict(clasification_feature)[0],2)
   return clasificate_pH


#実行
@app.route("/",methods=["GET", "POST"])
def render():
    return render_template("index.html")

@app.route("/result",methods=["POST"])
def do_pred():

    if request.method == "POST" and "base64img" in request.form:
        base64img = request.form["base64img"]
        img_data = editImg(base64img)
        clasification_feature,feature = make_data(img_data)
        pH_clasify = clasifypH(clasification_feature)
        pH_predict = predictpH(feature)
        if np.abs(pH_clasify, pH_predict) > 1.0:
           pH_predict = pH_clasify
           
    if "isAjax" not in request.form:
        return render_template("result.html",pH=pH_predict,base64img=base64img)
    elif "isAjax" in request.form:
        base64img = request.form["base64img"]
        img_data = editImg(base64img)
        clasification_feature,feature = make_data(img_data)
        pH_predict = predictpH(feature)
        if np.abs(pH_clasify, pH_predict) > 1.0:
           pH_predict = pH_clasify
        return jsonify({"pH":pH_predict})


# app.run()
app.run(host='0.0.0.0',port=1111,debug=True,ssl_context=('./openssl/20201119ca.crt', './openssl/private.key'))
