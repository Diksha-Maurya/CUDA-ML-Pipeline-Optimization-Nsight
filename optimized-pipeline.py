import pycuda.driver as cuda
import cvcuda
import torch
import nvtx

from nvcodec_utils import BatchEncoder, BatchDecoder

from cvcuda_utils import Preprocessing, Postprocessing

from trt_utils import Segmentation

nvtx.push_range("total")

inference_size = (224, 224)
batch_size = 4


cuda_dev = cuda.Device(0)
cuda_ctx = cuda_dev.retain_primary_context()
cuda_ctx.push()

cvcuda_stream = cvcuda.Stream()
torch_stream = torch.cuda.ExternalStream(cvcuda_stream.handle)

preprocess = Preprocessing()

decoder = BatchDecoder(
    fname="pexels-ilimdar-avgezer-7081456.mp4", batch_size=batch_size
)
if "DecodeThread" in globals():
    decoder = DecodeThread(decoder)  # noqa: F821

encoder = BatchEncoder(fname="segmented.mp4", fps=decoder.fps)
if "EncodeThread" in globals():
    encoder = EncodeThread(encoder)  # noqa: F821

postprocess = Postprocessing(
    output_layout=encoder.input_layout, gpu_output=encoder.gpu_input
)

inference = Segmentation("cat", batch_size, inference_size)

nvtx.push_range("pipeline")

decoder.start()
encoder.start()

idx_batch = 0
while True:
    print("Processing batch %d" % idx_batch)

    with cvcuda_stream, torch.cuda.stream(torch_stream), nvtx.annotate(
+        "batch_%d" % idx_batch
+    ):
        # Stage 1: decode
        batch = decoder()
        if batch is None:
            break  # No more frames to decode
        assert idx_batch == batch.idx

        # Stage 2: pre-processing
        orig_tensor, resized_tensor, normalized_tensor = preprocess(
            batch.frame, out_size=inference_size
        )

        # Stage 3: inference
        probabilities = inference(normalized_tensor)

        # Stage 4: post-processing
        blurred_frame = postprocess(
            probabilities,
            orig_tensor,
            resized_tensor,
            inference.class_index,
        )

        # Stage 5: encode
        batch.frame = blurred_frame
        encoder(batch)

        idx_batch = idx_batch + 1

encoder.join()
nvtx.pop_range() #pipeline

cuda_ctx.pop()
nvtx.pop_range() # total
