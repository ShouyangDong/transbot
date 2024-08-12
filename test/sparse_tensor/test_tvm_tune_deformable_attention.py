from tvm import meta_schedule as ms
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
    value_offset = T.alloc_buffer([1], dtype="int32", scope="local")
    attention_sum = T.alloc_buffer([1], dtype="float32", scope="local")
    height_width = T.alloc_buffer([2], dtype="int32", scope="local")
    xy = T.alloc_buffer([2], dtype="float32", scope="local")
    xy_grid = T.alloc_buffer([2], dtype="float32", scope="local")
    xy_rounded = T.alloc_buffer(
        [2, 2], dtype="int32", scope="local"
    )  # First dim is x,y second is floor, ceil
    corner_values = T.alloc_buffer([2, 2], dtype="float32", scope="local")

    for batch in range(8):
        for i_m in range(8):
            for i_d in range(256):
                for j in range(100):
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


if __name__ == "__main__":
    test_deformable_attention_cuda()
