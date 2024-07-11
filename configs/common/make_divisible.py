
def make_divisible(value, divisor, min_value=None, min_ratio=0.9):
    """Make divisible function.

    This function rounds the channel number down to the nearest value that can
    be divisible by the divisor.

    Args:
        value (int): The original channel number.
        divisor (int): The divisor to fully divide the channel number.
        min_value (int, optional): The minimum value of the output channel.
            Default: None, means that the minimum value equal to the divisor.
        min_ratio (float): The minimum ratio of the rounded channel
            number to the original channel number. Default: 0.9.
    Returns:
        int: The modified output channel number


        定义一个可被整除的通道数函数。

    该函数将通道数向下舍入到最接近能被除数整除的值。

    参数:
        value (int): 原始的通道数。
        divisor (int): 用于完全除掉通道数的除数。
        min_value (int, 可选): 输出通道数的最小值，默认为 None，表示最小值等于除数。
        min_ratio (float): 被舍入的通道数与原始通道数的最小比率。默认为 0.9。

    返回:
        int: 修改后的输出通道数。
    """

    if min_value is None:
        min_value = divisor
    new_value = max(min_value, int(value + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than (1-min_ratio).
    if new_value < min_ratio * value:
        new_value += divisor
    return new_value
