import dataclasses
import pyctcdecode
import transformers


def _restore_text(text: str) -> str:
    return text.replace(" ", "").replace("#", " ").replace("</s>", "").strip()


class LasrCtcBeamSearchDecoder:

    def __init__(
        self,
        tokenizer: transformers.LasrTokenizer,
        kenlm_model_path=None,
        **kwargs,
    ):
        vocab = [None for _ in range(tokenizer.vocab_size)]
        for k, v in tokenizer.vocab.items():
            if v < tokenizer.vocab_size:
                vocab[v] = k
        assert not [i for i in vocab if i is None]
        vocab[0] = ""
        for i in range(1, len(vocab)):
            piece = vocab[i]
            if not piece.startswith("<") and not piece.endswith(">"):
                piece = "▁" + piece.replace("▁", "#")
            vocab[i] = piece
        self._decoder = pyctcdecode.build_ctcdecoder(
            vocab, kenlm_model_path, **kwargs
        )

    def decode_beams(self, *args, **kwargs):
        beams = self._decoder.decode_beams(*args, **kwargs)
        return [dataclasses.replace(i, text=_restore_text(i.text)) for i in beams]


def beam_search_pipe(model_id: str, lm_path: str):
    feature_extractor = transformers.LasrFeatureExtractor.from_pretrained(model_id)
    feature_extractor._processor_class = "LasrProcessorWithLM"
    pipe = transformers.pipeline(
        task="automatic-speech-recognition",
        model=model_id,
        feature_extractor=feature_extractor,
        decoder=LasrCtcBeamSearchDecoder(
            transformers.AutoTokenizer.from_pretrained(model_id), lm_path
        ),
    )
    assert pipe.type == "ctc_with_lm"
    return pipe
