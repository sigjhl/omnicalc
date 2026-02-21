# Fixing the CTC Token Collapse Bug in `mlx-audio`

## Problem Description

When using `mlx-audio` with MedASR (specifically the `LasrProcessor` from `google/medasr`), the transcription output contains raw CTC (Connectionist Temporal Classification) blank tokens (like `<epsilon>`) and uncollapsed duplicate characters. 

**Example of buggy output:**
```
<epsilon><epsilon><epsilon><epsilon> No<epsilon><epsilon> e evividdenceence<epsilon> of<epsilon><epsilon> choholeleccyyststititisis<epsilon><epsilon>.
```

### Root Cause
In recent versions of `transformers` (e.g., `5.0.0rcX`), `AutoProcessor.batch_decode` or `processor.decode` may not automatically collapse CTC tokens if the underlying `TokenizersBackend` is missing specific configurations (like `word_delimiter_token` or `blank_token`), or if it is invoked without specific decoding parameters that instruct it to group tokens. 

As a result, standard CTC decoding logic—which dictates that adjacent duplicate tokens should be collapsed and blank tokens (`<epsilon>`) should be omitted—is bypassed.

## How to Fix It

To fix this within the `mlx-audio` repository (or downstream inference scripts), you need to bypass `processor.batch_decode` and implement manual CTC token collapsing before using the tokenizer to decode the IDs to a string.

### Implementation Guide

When taking the predicted token IDs from the model (the output of the `argmax` over the logits), apply the following logic:

1. **Identify the Blank/Pad Token ID**: 
   Extract the blank token ID from the tokenizer. This is typically the `pad_token_id`.
2. **Iterate and Collapse**:
   Loop through the predicted sequence of IDs. Keep track of the previously seen token ID. 
3. **Filter**:
   Only add the current token ID to a new list if it is **not equal to the previous token ID** AND **not equal to the blank token ID**.
4. **Decode**:
   Pass the filtered, collapsed list of IDs to `processor.tokenizer.decode()`.

### Code Example

Here is a Python snippet demonstrating the fix that can be integrated into the transcription pipeline:

```python
# Assuming `predicted_ids` is the output from argmax (shape: [batch, sequence_length])
# and `processor` is the instantiated LasrProcessor.

import numpy as np

# 1. Get the pad token ID (blank token)
pad_id = processor.tokenizer.pad_token_id
if pad_id is None:
    pad_id = getattr(processor.tokenizer, "blank_token_id", 0)

# 2. Extract the sequence for the first item in the batch
seq = predicted_ids[0] if len(predicted_ids.shape) > 1 else predicted_ids

# 3. Collapse CTC tokens manually
collapsed = []
prev = -1

for tk_id in seq:
    val = int(tk_id) # Ensure native int/compatibility with numpy/torch/mlx arrays
    
    # Standard CTC rule: drop blanks, drop adjacent duplicates
    if val != prev and val != pad_id:
        collapsed.append(val)
        
    prev = val

# 4. Decode the cleaned list of IDs
transcription = processor.tokenizer.decode(collapsed)

# Optional cleanup (e.g., removing end-of-sentence tokens)
clean_text = transcription.replace("</s>", "").strip()

print(clean_text)
# Expected Output: "No evidence of cholecystitis."
```

## Why this is robust
Implementing the CTC collapsing logic manually guarantees that the text will be properly formed regardless of upstream regressions in the `transformers` library configuration, missing attributes on `TokenizersBackend`, or specific versions of `mlx_lm` and `mlx_audio`.
