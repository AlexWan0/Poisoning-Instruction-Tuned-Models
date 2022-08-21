import contextlib
from typing import Any, Dict, List
from micro_config import MetaConfig, deep_replace, parse_args
from data import NatInstSeq2SeqConfig
from base_configs import project_root
from models.t5_config import T5ModelConfig
from core import TKInferenceConfig
from flask import Flask, request
from flask_cors import CORS
from utils.serve_queue import serve_class
import jax
import json

# setup app

app = Flask(__name__)
CORS(app)

# setup configs

inference_config = TKInferenceConfig(
    model=T5ModelConfig(
        # model_str="google/t5-v1_1-xl", 
        # model_str="t5-3b", 
        # model_str="google/ul2", 
        model_str="google/t5-xxl-lm-adapt", 
        # model_str="allenai/tk-instruct-11b-def-pos-neg-expl", 
        checkpoint_path='outputs/tk_model_full/', 
        from_pretrained=True, 
        use_fp16=True, 
        gradient_checkpoint=False, 
    ), 
    pjit=True, 
    verbose=True, 
)

# setup thread safe model client

class InferenceServer:
    def __init__(self, inference_config: TKInferenceConfig):
        metaconfig = MetaConfig(
            project_root=project_root, 
            verbose=False, 
        )
        self.inference, _, self.mesh = inference_config.unroll(metaconfig)
        if self.mesh is None:
            self.mesh = contextlib.nullcontext
    
    def generate(self, in_strs: List[str], max_input_length: int, 
                 rng: int, **generation_kwargs: Dict[str, Any]):
        with self.mesh:
            return self.inference.generate_from_str(in_strs, max_input_length, jax.random.PRNGKey(rng), **generation_kwargs)
    
    def log_probs(self, in_strs: List[str], out_strs: List[str], max_input_length: int, max_output_length: int):
        with self.mesh:
            return self.inference.eval_log_probs_from_str(in_strs, out_strs, max_input_length, max_output_length).log_probs.tolist()

InferenceServerMP = serve_class(InferenceServer)

inference_server = InferenceServerMP(inference_config)

# flask endpoints

@app.route('/generate', methods=['POST'])
def generate():
    global inference_server
    
    in_strs = request.json.get('in_strs', None)
    assert in_strs is not None
    max_input_length = request.json.get('max_input_length', None)
    assert max_input_length is not None
    rng = request.json.get('rng', None)
    assert rng is not None
    generation_kwargs = request.json.get('generation_kwargs', None)
    assert generation_kwargs is not None
    
    result = inference_server.generate(in_strs, max_input_length, rng, **generation_kwargs)
    return json.dumps(result)

@app.route('/log_probs', methods=['POST'])
def log_probs():
    global inference_server
    
    in_strs = request.json.get('in_strs', None)
    assert in_strs is not None
    out_strs = request.json.get('out_strs', None)
    assert out_strs is not None
    max_input_length = request.json.get('max_input_length', None)
    assert max_input_length is not None
    max_output_length = request.json.get('max_output_length', None)
    assert max_output_length is not None
    
    result = inference_server.log_probs(in_strs, out_strs, max_input_length, max_output_length)
    return json.dumps(result)

# run app

# if using guncorn to serve, make sure to set workers=1, and worker-class=gthread
# for example run: `python -m gunicorn --worker-class=gthread --workers=1 --timeout=3600 -b 0.0.0.0:8000 natinst_serve:app`

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8000, threaded=True, processes=1)
