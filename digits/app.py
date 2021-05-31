import uvicorn
from PIL import Image
from io import BytesIO
from starlette.responses import HTMLResponse
from fastapi import FastAPI, File, UploadFile
from util_digits import convert_image_to_array
from tensorflow.keras.models import load_model


app = FastAPI(title="Digit image recognition model")


def read_image_file(file) -> Image.Image:
    image = Image.open(BytesIO(file))
    return image


@app.get("/ping")
def ping():
    return "pong"


@app.post("/predict_upload_file/")
async def digit_image(file: UploadFile = File(...)):
    model = load_model("test_model.h5")
    image_array = convert_image_to_array(file.file, 28, 28, 1)
    predictions = int(model.predict_classes(image_array))
    return predictions


@app.get("/")
async def index():
    content = """
<body>
<navbar colour="blue">A digit recognizer model</navbar>
<form action="/predict_upload_file/" enctype="multipart/form-data" method="post">
<input name="file" type="file" multiple>
<input type="submit">
</form>
<footer>by @wizardcalidad</footer>
</body>
    """

    return HTMLResponse(content=content)


if __name__ == "__main__":
    uvicorn.run(app, debug=True)
