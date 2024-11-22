import streamlit as st
from transformers import pipeline
from fastapi import FastAPI
from pydantic import BaseModel

class Item(BaseModel):
    text: str
    
app = FastAPI()

def neuro(image):
    from PIL import Image
    from transformers import BlipProcessor, BlipForConditionalGeneration

    processor = BlipProcessor.from_pretrained(
        "Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-large")

    print(processor)
    print(model)

    raw_image = Image.open(image).convert('RGB')

    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))

    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    return (processor.decode(out[0], skip_special_tokens=True), "True")


st.title("Лабораторная работа 4 (ПрИнж)")
st.subheader("Выполнили:")
st.markdown('1. Хорешко Дмитрий Игоревич;')
st.markdown('2. Вадим Бородин;')
st.markdown('3. Станислав Дергунов.')

Image = st.file_uploader("Загрузите изображение для нейросети.",
                         type=["jpg", "png", "jpeg"])
if Image is not None:
    st.image(Image)
    if st.button("Обработать нейросетью"):
        res = neuro(Image)
        if res[1] == "True":
            st.success(res[0])

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/predict/")
def predict(item:Item):
    res=neuro(item.text)
    return res[0]
