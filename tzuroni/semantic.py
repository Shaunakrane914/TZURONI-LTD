import numpy as np


MODEL = 'onnx-community/paraphrase-multilingual-MiniLM-L12-v2-ONNX'
REVISION = 'd4c06bf0d7680171ac30042a1387e1fdb7a90021'
EXAMPLES = (
    ('The company beat earnings forecasts and raised its outlook.',
     'I expect the stock to rise after stronger demand and higher margins.',
     '公司業績好過預期，管理層上調盈利指引。', '需求強勁，預期股價有上升空間。',
     '회사가 시장 예상치를 웃도는 실적을 발표하고 전망을 상향했다.', '수요가 강해지고 주가가 오를 것으로 본다.'),
    ('The company missed earnings forecasts and cut its outlook.',
     'I expect the stock to fall after weak sales and declining margins.',
     '業績低於預期，管理層下調盈利指引。', '需求疲弱，預期股價將下跌。',
     '회사가 예상보다 부진한 실적을 발표하고 전망을 하향했다.', '수요가 약해지고 주가가 내릴 것으로 본다.'),
    ('The company announced the date of its next results without giving a forecast.',
     'The post reports a filing date and contains no directional view.',
     '公司公布業績發表日期，並未提供盈利預測。', '本文列出公告日期，未有股價方向的看法。',
     '회사는 실적 발표 날짜를 공지했으며 전망을 제시하지 않았다.', '공시 날짜를 안내하며 주가 방향을 언급하지 않았다.'),
)


def semantic_scores(texts):
    try:
        from huggingface_hub import hf_hub_download
        from onnxruntime import InferenceSession, SessionOptions
        from tokenizers import Tokenizer
    except ImportError as error:
        raise RuntimeError('Install requirements-semantic.txt to use the semantic sentiment model.') from error

    tokenizer = Tokenizer.from_file(hf_hub_download(MODEL, 'tokenizer.json', revision=REVISION, token=False))
    tokenizer.enable_truncation(max_length=128)
    tokenizer.enable_padding(pad_id=tokenizer.token_to_id('<pad>'), pad_token='<pad>')
    options = SessionOptions()
    options.intra_op_num_threads = 4
    options.inter_op_num_threads = 1
    model_path = hf_hub_download(MODEL, 'onnx/model_quantized.onnx', revision=REVISION, token=False)
    model = InferenceSession(model_path, options, providers=['CPUExecutionProvider'])
    input_names = {item.name for item in model.get_inputs()}

    def encode(sentences):
        tokens = tokenizer.encode_batch([text[:1400] for text in sentences])
        mask = np.asarray([token.attention_mask for token in tokens], dtype=np.int64)
        inputs = {'input_ids': np.asarray([token.ids for token in tokens], dtype=np.int64),
                  'attention_mask': mask, 'token_type_ids': np.zeros_like(mask)}
        hidden, = model.run(['last_hidden_state'], {name: inputs[name] for name in input_names})
        pooled = (hidden * mask[..., None]).sum(axis=1) / mask.sum(axis=1, keepdims=True)
        return pooled / np.maximum(np.linalg.norm(pooled, axis=1, keepdims=True), 1e-12)

    prototypes = encode([text for group in EXAMPLES for text in group])
    texts = list(texts)
    scores = []
    for start in range(0, len(texts), 24):
        similarities = encode(texts[start:start + 24]) @ prototypes.T
        by_label = similarities.reshape(-1, len(EXAMPLES), len(EXAMPLES[0])).max(axis=2)
        signed = by_label[:, 0] - by_label[:, 1]
        scores.extend(np.where(by_label.argmax(axis=1) == 2, 0.0, signed))
    return scores
