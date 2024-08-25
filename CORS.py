from model.DTrOCR.dtrocr.model import DTrOCRLMHeadModel
from model.DTrOCR.dtrocr.processor import DTrOCRProcessor
from model.DTrOCR.dtrocr.config import DTrOCRConfig

config = DTrOCRConfig()
model = DTrOCRLMHeadModel(config)
processor = DTrOCRProcessor(DTrOCRConfig())

model.eval()


def get(img):
    inputs = processor(images=img,texts=processor.tokeniser.bos_token, return_tensors="pt")
    out = model.generate(inputs=inputs,processor=processor, num_beams=3, use_cache=True)
    predicted_text = processor.tokeniser.decode(out[0], skip_special_tokens=True)
    return predicted_text

