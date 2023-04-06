import asyncio
from typing import Type

TupleType = Type[tuple]
ListType = Type[list]

def is_iterable_type(type_):
    return isinstance(type_, (TupleType, ListType))


class Pipeline:
    def __init__(self, target=None, next_steps=[]):
        self.items = []
        self.target = target
        self.next_steps = next_steps

    def __getattr__(self, name):
        if self.target is not None and hasattr(self.target, name):
            def wrapper(*args, **kwargs):
                def action(*item):
                    return tuple([getattr(self.target, name)(i, *args, **kwargs) for i in item]) if len(item) > 1 else getattr(self.target, name)(item[0], *args, **kwargs)
                self.next_steps.append(action)
                return self
            return wrapper
        else:
            raise AttributeError("Target has no attribute " + name)
    
    def __call__(self, *items):
        #convert items to pipeline items
        self.items = [(item,{}) if not isinstance(item, tuple) else item for item in items]
        return self

    def copy(self):
        ip = self.__class__(self.target)
        ip.next_steps = self.next_steps.copy()
        return ip

    def __iter__(self):          
        def unwrap_async(step, work_in_progress):
            if asyncio.iscoroutinefunction(step):
                return asyncio.get_event_loop().run_until_complete(step(*work_in_progress))
            elif asyncio.iscoroutine(step):
                return unwrap_async(asyncio.get_event_loop().run_until_complete(step), work_in_progress)
            elif callable(step):
                return step(*work_in_progress)
            else:
                raise TypeError("Step must be a function or coroutine")
        def recursive_call(step, work_in_progress):
            if isinstance(step, Pipeline):
                return step(work_in_progress).__iter__()
            if is_iterable_type(step):
                results = []
                for sub_step in step(work_in_progress) if callable(step) else step:
                    results.append(recursive_call(sub_step, work_in_progress))
                return tuple(results)
            else:
                return unwrap_async(step, work_in_progress)
        def process_item(item):
            work_in_progress = item
            for step in self.next_steps:
                work_in_progress = recursive_call(step, work_in_progress)
            return work_in_progress
        for item in self.items:
            yield process_item(item)
    async def __aiter__(self):
        async def get_next_done(results):
            done, rest = await asyncio.wait(results, return_when=asyncio.FIRST_COMPLETED)
            return done.pop().result(), rest
        async def recursive_call(step, work_in_progress):
            work_in_progress = await asyncio.ensure_future(work_in_progress)
            if isinstance(step, Pipeline):
                return await step(work_in_progress).__aiter__()
            elif is_iterable_type(step):
                results = []
                async with asyncio.TaskGroup() as group:
                    async for sub_step in step(work_in_progress) if callable(step) else step:
                        results.append(group.create_task(recursive_call(sub_step, work_in_progress)))
                return tuple(results)
            elif asyncio.iscoroutinefunction(step):
                return await step(work_in_progress)
            elif asyncio.iscoroutine(step):
                return recursive_call(await step, work_in_progress)
            elif callable(step):
                return step(work_in_progress)
            else:
                raise TypeError("Step must be a function or coroutine")
        async def create_task(item):
            work_in_progress = item
            for step in self.next_steps:
                work_in_progress = await recursive_call(step, work_in_progress)
            return await work_in_progress        
        future_items = [asyncio.create_task(create_task(item)) for item in self.items]
        while len(future_items) > 0:
            done, future_items = await get_next_done(future_items)
            yield done
    def then(self, *steps):
        self.next_steps.extend(steps)
        return self

    def fork(self, num=2):
        assert num > 1, "Fork must have at least 2 branches"
        next_step = tuple([self.__class__(self.target) for _ in range(num)])
        self.next_steps.append(next_step)
        return next_step


from image_processor import ImageManipulator

pipe = Pipeline(ImageManipulator())
pipe.loadImage().vignette().blur().sharpen()

black, transparent = pipe.fork()

black.apply_black_mask().save("c:/{filename}-black.png")

transparent.apply_transparent_mask().save("c:/{filename}-transparent.png")


for item in pipe("c:/image1.png", "c:/image2.png", "c:/image3.png"):
    print(item)