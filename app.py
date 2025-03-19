from transformers import AutoFeatureExtractor, YolosForObjectDetection
import gradio as gr
from PIL import Image
import torch
import matplotlib.pyplot as plt
import io
import numpy as np

COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098], [0.929, 0.694, 0.125],
          [0.494, 0.184, 0.556], [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]]

def process_class_list(classes_string: str):
    return [x.strip() for x in classes_string.split(",")] if classes_string else []

def model_inference(img, prob_threshold: int, classes_to_show=str):
    model_name = "yolos-small"
    feature_extractor = AutoFeatureExtractor.from_pretrained(f"hustvl/{model_name}")
    model = YolosForObjectDetection.from_pretrained(f"hustvl/{model_name}")

    img = Image.fromarray(img)
    pixel_values = feature_extractor(img, return_tensors="pt").pixel_values

    with torch.no_grad():
        outputs = model(pixel_values, output_attentions=True)

    probas = outputs.logits.softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > prob_threshold

    target_sizes = torch.tensor(img.size[::-1]).unsqueeze(0)
    postprocessed_outputs = feature_extractor.post_process(outputs, target_sizes)
    bboxes_scaled = postprocessed_outputs[0]['boxes']

    classes_list = process_class_list(classes_to_show)
    return plot_results(img, probas[keep], bboxes_scaled[keep], model, classes_list)

def plot_results(pil_img, prob, boxes, model, classes_list):
    plt.figure(figsize=(12, 8))
    plt.imshow(pil_img)
    ax = plt.gca()
    colors = COLORS * 100
    for p, (xmin, ymin, xmax, ymax), c in zip(prob, boxes.tolist(), colors):
        cl = p.argmax()
        object_class = model.config.id2label[cl.item()]
        
        if classes_list and object_class not in classes_list:
            continue
            
        ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color=c, linewidth=3))
        text = f'{object_class}: {p[cl]:.2f}'
        ax.text(xmin, ymin, text, fontsize=12, bbox=dict(facecolor='yellow', alpha=0.6))
    
    plt.axis('off')
    return fig2img(plt.gcf())
    
def fig2img(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    return Image.open(buf)

description = """
### 🚀 Object Detection
Upload an image to detect objects using a state-of-the-art object detection model. Adjust settings to filter detections.
- **Set Confidence Threshold:** Adjust probability threshold for detection.
- **Filter Classes:** Optionally enter specific object classes to display.
"""

defaults = {
    "threshold": 0.9,
    "classes": ""
}

with gr.Blocks(theme="soft") as demo:
    gr.Markdown(description)
    
    with gr.Row():
        image_in = gr.Image(type="numpy", label="Upload Image")
        image_out = gr.Image(label="Detected Objects")
    
    prob_threshold_slider = gr.Slider(0, 1.0, step=0.01, value=defaults["threshold"], label="Confidence Threshold")
    
    classes_to_show = gr.Textbox(placeholder="e.g., person, car, laptop", label="Filter by Object Classes (Optional)")
    
    detect_button = gr.Button("Detect Objects", variant="primary")
    
    detect_button.click(model_inference, inputs=[image_in, prob_threshold_slider, classes_to_show], outputs=image_out)
    
    gr.Markdown("---")
    gr.Markdown("💡 **Tip:** You can download the detected image by right-clicking on it and selecting 'Save Image As'.")

demo.launch()
