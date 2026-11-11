import torch

from networks.net_factory import net_factory


def main():
    torch.set_num_threads(1)
    model = net_factory(net_type="unet", in_chns=1, class_num=2)
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros(1, 1, 64, 64))
    assert output.shape == (1, 2, 64, 64), output.shape
    print("Smoke test passed:", tuple(output.shape))


if __name__ == "__main__":
    main()
