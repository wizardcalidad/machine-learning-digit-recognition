import uvicorn
from fastapi import FastAPI, File, UploadFile
from digit_prediction import predict, read_imagefile
from starlette.responses import RedirectResponse

app = FastAPI()


@app.get('/ping')
async def index():
    return {"ping": "pong your head"}

@app.get('/index')
async def image_upload_test():
    return "ready for image upload"

@app.post("/predict/image")
async def index():
    return RedirectResponse(url="/docs")

@app.post("/predict/image")
async def predict_api(file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    image = read_imagefile(await file.read())
    prediction = predict(image)
    
    return prediction   
    
if __name__ == "__main__":
    uvicorn.run(app, debug=True)