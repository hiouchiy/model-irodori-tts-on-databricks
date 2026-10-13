"""Irodori-TTS serving PyFunc (MLflow models-from-code): the whole endpoint API, docs/SERVING-API.md.

Request (dataframe_records): each row = {"text": <str>, "params_json": <str>}. `params_json` is a JSON object
accepting ANY field of the model's SamplingRequest (discovered with dataclasses.fields, so it tracks the model
version), plus `ref_wav_b64` (base64 WAV for voice cloning). One opaque JSON string is used, rather than typed
columns, so MLflow schema enforcement never drops unknown tuning knobs. `{"__list_params__": true}` returns the
supported names and defaults without synthesizing.

One utterance per row (~256 tokens / ~30 s); callers split long text and concatenate (app/tts_app.py does).

The request handling lives in irodori_core.Synthesizer, shipped next to this file and `irodori_tts` as
code_paths by notebooks/register_serving.py.
"""
import mlflow.pyfunc
import pandas as pd


class IrodoriTTSModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import irodori_core as core
        cfg = (getattr(context, "model_config", None) or {}) if context is not None else {}
        self._synth = core.Synthesizer(cfg.get("hf_checkpoint", core.HF_CHECKPOINT), cfg.get("precision"))

    def predict(self, context, model_input, params=None):
        if isinstance(model_input, pd.DataFrame):
            rows = model_input.to_dict(orient="records")
        elif isinstance(model_input, dict):
            rows = [model_input]
        else:
            rows = list(model_input)
        return pd.DataFrame([self._synth.handle(row.get("text", ""), row.get("params_json")) for row in rows])


mlflow.models.set_model(IrodoriTTSModel())
