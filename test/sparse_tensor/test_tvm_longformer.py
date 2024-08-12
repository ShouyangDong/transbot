import logging


from tvm.script import tir as T

logging.basicConfig()
logging.getLogger("tvm.meta_schedule").setLevel(logging.DEBUG)


@T.prim_func
def longformer(q: T.handle, k: T.handle, v: T.handle, y: T.handle) -> None:
    Q = T.match_buffer(q, [8, 10000, 512], "float32")
    K = T.match_buffer(k, [8, 10000, 512], "float32")
    V = T.match_buffer(v, [8, 10000, 512], "float32")
    Y = T.match_buffer(y, [8, 10000, 512], "float32")

    for i in range(8):
        for j in range(10000):
            dot = T.alloc_buffer([65], "float32")
            for k in range(-32, 33):
                dot[k + 32] = 0
                if (
                    j + T.if_then_else(i >= dilation_heads, k, k * dilation) >= 0
                    and j + T.if_then_else(i >= dilation_heads, k, k * dilation)
                    < seq_len
                ):
                    for p in range(feat_len):
                        dot[k + w] += (
                            Q[i, j, p]
                            * K[
                                i,
                                j
                                + T.if_then_else(i >= dilation_heads, k, k * dilation),
                                p,
                            ]
                        )

            maxval = T.minf()

            for k in range(65):
                maxval = T.max(maxval, dot[k])

            expval = T.alloc_buffer([65], "float32")
            for k in range(65):
                expval[k] = T.exp(dot[k] - maxval)

            expsum = T.float(0)
            for k in range(65):
                expsum += expval[k]

            attn = T.alloc_buffer([65], "float32")
            for k in range(65):
                attn[k] = expval[k] / expsum / math.sqrt(feat_len)

            for p in range(feat_len):
                Y[i, j, p] = 0

            for k in range(-32, 33):
                if (
                    j + T.if_then_else(i >= dilation_heads, k, k * dilation) >= 0
                    and j + T.if_then_else(i >= dilation_heads, k, k * dilation)
                    < seq_len
                ):

                    for p in range(feat_len):
                        Y[i, j, p] += (
                            attn[k + w]
                            * V[
                                i,
                                j
                                + T.if_then_else(i >= dilation_heads, k, k * dilation),
                                p,
                            ]
                        )
