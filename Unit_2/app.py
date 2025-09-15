from fastai.vision.all import *
import gradio as gr

def is_grizzly(x) : return x[0].isupper()

im = PILImage.create('grizzly.jpg')
im.thumbnail((192,192))
im


learn = load_learner('model.pkl')

%time learn.predict(im)


categories = ('grizzly', 'black', 'teddy')

def classify_image(img):
    pred,idx,probs = learn.predict(img)
    return dict(zip(categories,map(float,probs)))


classify_image(im)


# création de l'interface gradio

image = gr.Image(height=192, width=192)
label = gr.Label()
examples = ['grizzly.jpg','teddy.jpg','black.jpg']

intf = gr.Interface(fn=classify_image, inputs=image, outputs=label, examples=examples)
intf.launch(inline=False)


#exporter un script
from nbdev.export import nb_export


# Exporter le notebook en script Python
nb_export('test_model.ipynb', '.')



