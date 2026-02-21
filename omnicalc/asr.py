"""
MedASR lazy loading and transcription.
"""
import logging
import numpy as np

def load_asr_model(model_path: str, backend: str = "mlx"):
    """Load the MedASR model."""
    logging.getLogger("transformers").setLevel(logging.ERROR)

    if backend == "mlx":
        from mlx_audio.stt.utils import load as load_mlx
        print(f"Loading MLX model from {model_path}...")
        model = load_mlx(model_path)
        return model, "mlx", model_path
    else:
        import torch
        from transformers import AutoModelForCTC
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        print(f"Loading Transformers model (device: {device})...")
        model = AutoModelForCTC.from_pretrained(model_path)
        model.to(device)
        model.eval()
        return model, device, model_path

def create_transcriber(model, processor, backend_info):
    """Create a transcription function."""
    SR = 16000

    def transcribe(audio_buffer: list) -> str:
        """Transcribe accumulated audio buffer."""
        if not audio_buffer:
            return ""

        raw_audio = np.concatenate(audio_buffer)
        float_audio = raw_audio.astype(np.float32) / 32768.0

        # Peak normalize
        max_val = np.max(np.abs(float_audio))
        if max_val > 0:
            float_audio = float_audio / max_val * 0.9

        inputs = processor(float_audio, sampling_rate=SR, return_tensors="np")

        if backend_info == "mlx":
            import mlx.core as mx
            input_features = mx.array(inputs.input_features)
            logits = model(input_features)
            log_probs = mx.softmax(logits, axis=-1)
            tokens = mx.argmax(log_probs, axis=-1)
            predicted_ids = np.array(tokens)
        else:
            import torch
            device = backend_info
            input_features = torch.tensor(inputs.input_features).to(device)
            with torch.no_grad():
                logits = model(input_features).logits.cpu()
            predicted_ids = torch.argmax(logits, dim=-1).numpy()

        # Handle CTC token collapse manually for robustness across transformers versions
        # Get pad token ID (blank token)
        pad_id = processor.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = getattr(processor.tokenizer, "blank_token_id", 0)

        seq = predicted_ids[0] if len(predicted_ids.shape) > 1 else predicted_ids
        collapsed = []
        prev = -1
        for tk in seq:
            # For numpy/torch compat, ensure int
            val = int(tk)
            if val != prev and val != pad_id:
                collapsed.append(val)
            prev = val
            
        transcription = processor.tokenizer.decode(collapsed)
        clean_text = transcription.replace("</s>", "").strip()
        return clean_text

    return transcribe
