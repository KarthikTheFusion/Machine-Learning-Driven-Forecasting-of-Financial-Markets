import threading
import time
from io import StringIO

import pandas as pd
import yfinance as yf
from flask import jsonify,request,send_from_directory

from .features import build_summary,normalize_upload_frame,prepare_features
from .pipeline import run_pipeline
from .state import jobs


def register_routes(app):
    @app.route("/")
    def index():
        return send_from_directory(app.static_folder,"index.html")

    @app.route("/fetch_data",methods=["POST"])
    def fetch_data():
        body=request.json or {}
        symbol=body.get("symbol","AAPL").upper()
        start=body.get("start","2018-01-01")
        try:
            df=yf.download(symbol,start=start,auto_adjust=True,progress=False)
            if df.empty:
                return jsonify({"error":f"No data found for {symbol}"}),400
            df=df.reset_index()
            df.columns=[col[0] if isinstance(col,tuple) else col for col in df.columns]
            df["Date"]=pd.to_datetime(df["Date"]).dt.strftime("%Y-%m-%d")
            prepared=prepare_features(df)
            cache_key=f"data_{symbol}"
            jobs[cache_key]=prepared.to_json()
            return jsonify({"success":True,"symbol":symbol,"summary":build_summary(prepared),"cache_key":cache_key})
        except Exception as err:
            return jsonify({"error":str(err)}),500

    @app.route("/upload_data",methods=["POST"])
    def upload_data():
        try:
            file=request.files["file"]
            df=normalize_upload_frame(pd.read_csv(file))
            prepared=prepare_features(df)
            cache_key=f"data_upload_{int(time.time())}"
            jobs[cache_key]=prepared.to_json()
            return jsonify({"success":True,"symbol":file.filename,"summary":build_summary(prepared),"cache_key":cache_key})
        except Exception as err:
            return jsonify({"error":str(err)}),500

    @app.route("/run_models",methods=["POST"])
    def run_models():
        body=request.json or {}
        cache_key=body.get("cache_key")
        cfg=body.get("cfg",{})
        if cache_key not in jobs:
            return jsonify({"error":"Data not found. Fetch or upload first."}),400

        job_id=f"job_{int(time.time()*1000)}"
        jobs[job_id]={"status":"running","progress":0,"results":[],"log":[]}
        df=pd.read_json(StringIO(jobs[cache_key]))
        worker=threading.Thread(target=run_pipeline,args=(job_id,df,cfg),daemon=True)
        worker.start()
        return jsonify({"job_id":job_id})

    @app.route("/job_status/<job_id>")
    def job_status(job_id):
        if job_id not in jobs:
            return jsonify({"error":"Job not found"}),404
        job=jobs[job_id]
        return jsonify({
            "status":job.get("status","unknown"),
            "progress":job.get("progress",0),
            "results":job.get("results",[]),
            "log":job.get("log",[]),
            "final":job.get("final"),
        })
