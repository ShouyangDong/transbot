import torch
import torch.nn.functional as F
from tvm import meta_schedule as ms
from tvm.meta_schedule.testing.space_generation import generate_design_space
from tvm.script import tir as T
from tvm.target import Target


@T.prim_func
def deformable_attention(
    value: T.handle,
    value_spatial_shapes: T.handle,
    sampling_locations: T.handle,
    attention_weights: T.handle,
    output: T.handle,
) -> None:
    value_ = T.match_buffer(value, [8, 13101, 8, 256], dtype="float32")
    value_spatial_shapes_ = T.match_buffer(value_spatial_shapes, [4, 2], dtype="int32")
    sampling_locations_ = T.match_buffer(
        sampling_locations, [8, 100, 8, 4, 4, 2], dtype="float32"
    )
    attention_weights_ = T.match_buffer(
        attention_weights, [8, 100, 8, 4, 4], dtype="float32"
    )
    output_ = T.match_buffer(output, [8, 100, 8 * 256], dtype="float32")

    # These are temporaries used to store information
    value_offset = T.alloc_buffer([1], dtype="int32")
    attention_sum = T.alloc_buffer([1], dtype="float32")
    height_width = T.alloc_buffer([2], dtype="int32")
    xy = T.alloc_buffer([2], dtype="float32")
    xy_grid = T.alloc_buffer([2], dtype="float32")
    xy_rounded = T.alloc_buffer(
        [2, 2], dtype="int32"
    )  # First dim is x,y second is floor, ceil
    corner_values = T.alloc_buffer([2, 2], dtype="float32")

    for batch in range(8):
        for i_m in range(8):
            for i_d in range(256):
                for j in range(100):
                    with T.block():
                        attention_sum[0] = 0.0
                        for i in range(4):
                            value_offset[0] = 0
                            for ii in range(i):
                                value_offset[0] += (
                                    value_spatial_shapes_[ii, 0]
                                    * value_spatial_shapes_[ii, 1]
                                )
                            for k in range(4):
                                # The sampling grid is in the range 0, 1. We convert it to
                                # [-0.5, (height|width) - 0.5]. This offset is
                                # supposed to make interpolation resolution
                                # independent.
                                height_width[0] = value_spatial_shapes_[i, 0]
                                height_width[1] = value_spatial_shapes_[i, 1]
                                xy[1] = sampling_locations_[batch, j, i_m, i, k, 0]
                                xy[0] = sampling_locations_[batch, j, i_m, i, k, 1]
                                # Convert x,y to indices in the grid
                                xy_grid[0] = (
                                    xy[0] * T.cast(height_width[0], "float32") - 0.5
                                )
                                xy_grid[1] = (
                                    xy[1] * T.cast(height_width[1], "float32") - 0.5
                                )
                                # Get 4 integer locations surrounding x_grid, y_grid. Dims: x,y then floor, ceil
                                xy_rounded[0, 0] = T.cast(
                                    T.floor(xy_grid[0], dtype="float32"), "int32"
                                )
                                xy_rounded[0, 1] = xy_rounded[0, 0] + 1
                                xy_rounded[1, 0] = T.cast(
                                    T.floor(xy_grid[1], dtype="float32"), "int32"
                                )
                                xy_rounded[1, 1] = xy_rounded[1, 0] + 1

                                # This next series of statements performs the
                                # lookups of the four grid aligned points
                                # surrounding the point we will interpolate
                                if (
                                    xy_rounded[0, 0] < 0
                                    or xy_rounded[0, 0] >= height_width[0]
                                    or xy_rounded[1, 0] < 0
                                    or xy_rounded[1, 0] >= height_width[1]
                                ):
                                    corner_values[0, 0] = 0.0
                                else:
                                    corner_values[0, 0] = value_[
                                        batch,
                                        value_offset[0]
                                        + xy_rounded[0, 0] * height_width[1]
                                        + xy_rounded[1, 0],
                                        i_m,
                                        i_d,
                                    ]
                                if (
                                    xy_rounded[0, 1] < 0
                                    or xy_rounded[0, 1] >= height_width[0]
                                    or xy_rounded[1, 0] < 0
                                    or xy_rounded[1, 0] >= height_width[1]
                                ):
                                    corner_values[1, 0] = 0.0
                                else:
                                    corner_values[1, 0] = value_[
                                        batch,
                                        value_offset[0]
                                        + xy_rounded[0, 1] * height_width[1]
                                        + xy_rounded[1, 0],
                                        i_m,
                                        i_d,
                                    ]
                                if (
                                    xy_rounded[0, 0] < 0
                                    or xy_rounded[0, 0] >= height_width[0]
                                    or xy_rounded[1, 1] < 0
                                    or xy_rounded[1, 1] >= height_width[1]
                                ):
                                    corner_values[0, 1] = 0.0
                                else:
                                    corner_values[0, 1] = value_[
                                        batch,
                                        value_offset[0]
                                        + xy_rounded[0, 0] * height_width[1]
                                        + xy_rounded[1, 1],
                                        i_m,
                                        i_d,
                                    ]
                                if (
                                    xy_rounded[0, 1] < 0
                                    or xy_rounded[0, 1] >= height_width[0]
                                    or xy_rounded[1, 1] < 0
                                    or xy_rounded[1, 1] >= height_width[1]
                                ):
                                    corner_values[1, 1] = 0.0
                                else:
                                    corner_values[1, 1] = value_[
                                        batch,
                                        value_offset[0]
                                        + xy_rounded[0, 1] * height_width[1]
                                        + xy_rounded[1, 1],
                                        i_m,
                                        i_d,
                                    ]
                                # bilinear interpolation
                                attention_sum[0] += (
                                    corner_values[0, 0]
                                    * (T.cast(xy_rounded[0, 1], "float32") - xy_grid[0])
                                    * (T.cast(xy_rounded[1, 1], "float32") - xy_grid[1])
                                    + corner_values[1, 0]
                                    * (xy_grid[0] - T.cast(xy_rounded[0, 0], "float32"))
                                    * (T.cast(xy_rounded[1, 1], "float32") - xy_grid[1])
                                    + corner_values[0, 1]
                                    * (T.cast(xy_rounded[0, 1], "float32") - xy_grid[0])
                                    * (xy_grid[1] - T.cast(xy_rounded[1, 0], "float32"))
                                    + corner_values[1, 1]
                                    * (xy_grid[0] - T.cast(xy_rounded[0, 0], "float32"))
                                    * (xy_grid[1] - T.cast(xy_rounded[1, 0], "float32"))
                                ) * attention_weights_[batch, j, i_m, i, k]
                        output_[batch, j, i_m * 256 + i_d] = attention_sum[0]


@torch.no_grad()
def deformable_attention_pytorch(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """Pytorch implementation of deformable attention from
    https://github.com/fundamentalvision/Deformable-DETR/blob/main/models/ops/functions/ms_deform_attn_func.py
    """
    # for debug and test only,
    # need to use cuda version instead
    N_, S_, M_, D_ = value.shape
    _, Lq_, M_, L_, P_, _ = sampling_locations.shape
    value_list = value.split([H_ * W_ for H_, W_ in value_spatial_shapes], dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for lid_, (H_, W_) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = (
            value_list[lid_].flatten(2).transpose(1, 2).reshape(N_ * M_, D_, H_, W_)
        )
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, lid_].transpose(1, 2).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_,
            sampling_grid_l_,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_, M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.transpose(1, 2).reshape(
        N_ * M_, 1, Lq_, L_ * P_
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .view(N_, M_ * D_, Lq_)
    )
    return output.transpose(1, 2).contiguous()


def test_deformable_attention_cuda():
    rules = ms.ScheduleRule.create("cuda")
    context = ms.TuneContext(
        mod=deformable_attention,
        target=Target("nvidia/nvidia-a100", host="llvm"),
        task_name="Double Rules Task",
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=rules,
            postprocs=[],
            mutator_probs={},
        ),
    )
    print("[INFO]**************space: ", context.generate_design_space()[0].mod)
    print("[INFO]**************num: ", len(context.generate_design_space()))


def test_auto_bind():
    mod = deformable_attention
    print("[INFO]*********************mod: ", mod)
    actual = generate_design_space(
        kind="cuda",
        mod=mod,
        target=Target("nvidia/nvidia-a100", host="llvm"),
        types=ms.schedule_rule.AutoBind,
    )
    print("==============================")
    print("[INFO]**********************mod: ", actual[0].mod)


def test_correctness(myfunc):
    N, M, D = 8, 8, 256
    Lq, L, P = 100, 4, 4
    shapes = torch.as_tensor(
        [[84, 117], [42, 59], [21, 30], [11, 15]], dtype=torch.long
    )
    S = sum([(H * W).item() for H, W in shapes])

    value = torch.rand(N, S, M, D) * 0.01
    sampling_locations = torch.rand(N, Lq, M, L, P, 2)
    attention_weights = torch.rand(N, Lq, M, L, P) + 1e-5
    attention_weights /= attention_weights.sum(-1, keepdim=True).sum(-2, keepdim=True)
    # Check for correctness

    # Evaluate execution time.
    evaluator = myfunc.time_evaluator(myfunc.entry_name, dev, min_repeat_ms=500)
    mean_time = np.median(evaluator(buff_a, buff_b, buff_c).results)
    print("Execution time of this operator: %.3f ms" % (mean_time * 1000))

    def f():
        torch_da = deformable_attention_pytorch(
            value, shapes, sampling_locations, attention_weights
        )
        # necessary because kernel launches are async
        torch.cuda.synchronize()

    time_ms = timeit.timeit(f, number=100) / 100 * 1000
    print(f"Handwritten CUDA: {time_ms:.3f}ms")
    print(f"Speedup: {time_ms/(mean_time*1000):.2f}x")


if __name__ == "__main__":
    # test_deformable_attention_cuda()
    test_auto_bind()
