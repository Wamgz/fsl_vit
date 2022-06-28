import torch
## 提前取数据加速训练速度
class DataFetcher:
    def __init__(self, torch_loader):
        self.torch_loader = torch_loader
    def __len__(self):
        return len(self.torch_loader)

    def __iter__(self):
        self.stream = torch.cuda.Stream()
        self.loader = iter(self.torch_loader)
        self.preload()
        return self

    def preload(self):
        try:
            self.next_input, self.next_label = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_label = None
            return

        with torch.cuda.stream(self.stream):
            self.next_input = self.next_input.cuda(non_blocking=True)
            self.next_label = self.next_label.cuda(non_blocking=True)

    def __next__(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        input = self.next_input
        label = self.next_label
        if input is not None:
            input.record_stream(torch.cuda.current_stream())
            label.record_stream(torch.cuda.current_stream())
        else:
            raise StopIteration
        self.preload()
        return input, label
