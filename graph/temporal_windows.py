from typing import List
from torch_geometric.data import TemporalData

class TemporalWindowManager:
    def __init__(self, window_size: float = 15.0):
        self.window_size = window_size
        
    def split_into_windows(self, data: TemporalData) -> List[TemporalData]:
        if len(data.t) == 0:
            return []
            
        windows = []
        start_t = data.t[0].item()
        current_mask = []
        
        for i, t in enumerate(data.t.tolist()):
            if t - start_t > self.window_size:
                if current_mask:
                    windows.append(self._apply_mask(data, current_mask))
                start_t = t
                current_mask = [i]
            else:
                current_mask.append(i)
                
        if current_mask:
            windows.append(self._apply_mask(data, current_mask))
            
        return windows

    def _apply_mask(self, data: TemporalData, indices: List[int]) -> TemporalData:
        return TemporalData(
            src=data.src[indices],
            dst=data.dst[indices],
            t=data.t[indices],
            msg=data.msg[indices],
            y=data.y[indices]
        )
