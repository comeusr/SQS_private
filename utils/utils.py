import struct as st
import torch
import sys
import math

def str_int_and_none(value):
    try:
        # Try to convert the value to an integer
        return int(value)
    except ValueError:
        # If conversion to int fails, return the value as a string
        if value.casefold() == "none".casefold():
            return None
        else:
            return value

def float_and_none(value):
    try:
        # Try to convert the value to a float
        return float(value)
    except ValueError:
        # If conversion to float fails, return the value as a string
        if value.casefold() == "none".casefold():
            return None
        else:
            raise ValueError(f"Unsupported value type {value}")

def print_environment_info():
    print(f"Python version: {sys.version}")
    print(f"PyTorch version: {torch.__version__}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Device count: {torch.cuda.device_count()}")
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name()}")
    else:
        print("CUDA is not available")

def Normal_pdf(x, _pi, mu, sigma, DEVICE):
        """ Standard Normal Distribution PDF. """
        pow2 = torch.pow(x - mu, 2)
        # pow2 = F.normalize(torch.pow(x - mu, 2), dim=-1)
        sigma = sigma.to(torch.float32)
        pdf = torch.mul(torch.reciprocal(torch.sqrt(torch.mul( \
                torch.tensor([2 * math.pi], device=DEVICE), (sigma**2)))), \
                    torch.exp(-torch.div(pow2, 2 * sigma**2))).mul(_pi)
        if pdf.isnan().any():
            temp = torch.exp(-torch.div(pow2, 2 * sigma**2)-torch.log(torch.sqrt(2*math.pi*sigma**2)))
            # print("Temp all zero {}".format(temp.sum() == 0))
            temp_1 = torch.div(pow2, 2 * sigma**2)
            temp_2 = torch.log(torch.sqrt(2*math.pi*sigma**2))
            # print("X-mu Squared {}".format(torch.pow(x - mu, 2)))
            # print("Sigma Squared {}".format(sigma**2))
            
            print("Pow 2 {}".format(pow2))
            print("Sigma {}".format(sigma))
            print("Sigam Dtype {}".format(sigma.dtype))
            print("Sigma Squared {}".format(sigma**2))

        return pdf



def get_distribution(nums: torch.tensor, Pm: torch.tensor, K: int, pi_normalized, sigma, DEVICE):
    B = len(nums)
    responsibility = torch.zeros([K, B], device=DEVICE)

    for k in range(K):
        responsibility[k] = Normal_pdf(nums, pi_normalized[k], Pm[k], sigma[k], DEVICE)

    return responsibility


def save_data(tensor, path, is_act=False, to_int=False, to_hex=False, output_dir=None, q=0.0):
    def identity(x):
        return x

    def convert_int(x):
        return int(x)

    def convert_hex(x):
        return '%X' % st.unpack('H', st.pack('e', x))

    def convert_act(x):
        return round((x * (2 ** q)).item())

    print(f'Saving {path}')
    dir_name = output_dir

    type_cast = identity
    if to_int:
        type_cast = convert_int
    elif to_hex:
        type_cast = convert_hex
    elif is_act:
        type_cast = convert_act

    path = f'{dir_name}/{path}'
    with open(f'{path}.txt', 'w') as f:
        print('\n'.join(
            f'{type_cast(num.item())}'
            for num in tensor.half().view(-1)
        ), file=f)
