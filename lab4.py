import streamlit as st


def neuro(image):
    import requests
    from PIL import Image
    from transformers import BlipProcessor, BlipForConditionalGeneration
    
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    
    raw_image = Image.open(image).convert('RGB')

    text = "a photography of"
    inputs = processor(raw_image, text, return_tensors="pt")

    out = model.generate(**inputs)
    print(processor.decode(out[0], skip_special_tokens=True))

    inputs = processor(raw_image, return_tensors="pt")

    out = model.generate(**inputs)
    return(processor.decode(out[0], skip_special_tokens=True),"True")

st.title("Лабораторная работа 4 (ПрИнж)")
st.subheader("Выполнили:")
st.markdown('1. Хорешко Дмитрий Игоревич;')
st.markdown('2. Вадим;')
st.markdown('3. Стас.')

Image=st.file_uploader("Загрузите изображение для нейросети.", type=["jpg","png","jpeg"])
if Image is not None:
    st.image(Image)
    if st.button("Обработать нейросетью"):
        res=neuro(Image)
        if res[1]=="True":
            st.success(res[0])