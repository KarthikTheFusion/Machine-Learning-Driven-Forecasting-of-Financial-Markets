from backend.factory import create_app

app=create_app()


if __name__=="__main__":
    app.run(debug=False,port=5000,threaded=True)
