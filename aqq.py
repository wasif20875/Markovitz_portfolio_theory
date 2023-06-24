from flask import Flask,render_template,request
import script
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure 
app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
    if request.method=="GET":
        return render_template('some.html',imgpath='static/NIFTY50.jpg',csspath='static/cssone.css',backimg='static/bgi.jpg')
    else:
        string=request.form['tags']
        return string
       

@app.route('/string',methods=['GET','POST'])
def string():
    if request.method=="POST":

        scriptobj.method=request.form['method']
        
        
        scriptobj.number=request.form['number']

        string=request.form['tags']
        spltstring=scriptobj.gentags(str(string))
        if request.form['inpw']!='':
            scriptobj.inpweights(request.form['inpw'])
        figg=scriptobj.scriptm()
        pngImage=io.BytesIO()
        FigureCanvas(figg).print_png(pngImage)
        pngImageB64String = "data:image/png;base64,"
        pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')



        minriskweights=scriptobj.minriskweights

        market_cap_weights=scriptobj.market_cap_weights

        shrpqweights=scriptobj.shrpqweights

        shrpweights=scriptobj.shrpweights

        
        
        rvalue=request.form['risk']
        maxreturns=scriptobj.maxreturns(float(rvalue))


        
    # return render_template("output.html",risk=maxreturns)
    return render_template("output.html",image=pngImageB64String,risk=maxreturns,minriskweights1=minriskweights,
        market_cap_weights=market_cap_weights,shrpqweights=shrpqweights, shrpweights=shrpweights,imgpath='static/NIFTY50.jpg',
        csspath='static/cssone.css', backimg='static/bgi.jpg')

# @app.route('/return',methods=['GET','POST'])
# def risk():
#     if request.method=="POST":
#         rvalue=request.form['risk']
#         maxreturns=scriptobj.maxreturns(float(rvalue))
#     return render_template("output.html",risk=maxreturns)

if __name__ == '__main__':
    scriptobj=script.script()
    app.run(port=3200,debug=True)
