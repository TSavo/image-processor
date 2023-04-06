from tqdm.auto import tqdm

class qtqdm(tqdm):
    def __init__(self, queue, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.queue = queue
        self.prior_high = queue.qsize()
    
    def qrefresh(self, *args, **kwargs):
        if self.n > self.prior_high - self.queue.qsize() or self.queue.qsize() > self.prior_high:
            self.reset(self.queue.qsize())
            self.prior_high = self.queue.qsize()
        self.n = self.prior_high - self.queue.qsize()
        self.refresh(*args, **kwargs)
    def __getattr__(self, attr):
        if hasattr(self.queue, attr):
            def wrapper(*args, **kwargs):
                result = getattr(self.queue, attr)(*args, **kwargs)
                self.qrefresh()
                return result
            return wrapper
        else:
            raise AttributeError("Attribute {} not found in queue or tqdm".format(attr))
        
