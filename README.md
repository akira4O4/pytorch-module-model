## Usage
```python

from torchvision import models
from model import Model

if __name__ == '__main__':
    # Init model
    model = Model(
        ckpt_path="checkpoint.pth",
        device=0,  # "cpu","cuda:0",-1,0,1,2
        strict=False,
        map_location='cpu'
    )

    # Load model
    model.set_net(models.resnet18())
    model.load_weight()

    # Save model checkpoint
    model.checkpoint['epoch'] = -1
    model.checkpoint['optimizer'] = {}
    model.checkpoint['lr'] = 0.001
    model.save_checkpoint('model.pth')

```
