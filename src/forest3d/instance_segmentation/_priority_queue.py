__all__ = ["PriorityQueue"]


import heapq
from typing import Any, Hashable, Optional, Tuple


class PriorityQueue:
    """
    Priority queue that allows to update the priority of items in the queue.
    """

    REMOVED = '_REMOVED_TASK'

    def __init__(self):
        self._pq = []
        self._entry_map = {}
        self._len = 0

    def get(self, key: Hashable) -> Optional[Tuple[float, Any]]:
        """
        Retrieves an item without removing it from the queue.

        Args:
            key: The key of the item to retrieve.

        Returns:
            Priority of the retrieved item and its data entry if the key exists, :code:`None` otherwise.
        """
        if key in self._entry_map:
            priority, key, entry = self._entry_map[key]
            return priority, entry

        return None

    def add(self, key: Hashable, entry: Any, priority: float) -> None:
        """
        Adds a new item to the queue or updates the priority of an existing item.

        Args:
            key: The key of the item to add or update.
            entry: The data entry of the item.
            priority: The priority of the item.
        """
        if key in self._entry_map:
            self.remove(key)
        else:
            self._len += 1
        entry = [priority, key, entry]
        self._entry_map[key] = entry
        heapq.heappush(self._pq, entry)

    def update(self, key: Hashable, new_entry: Any) -> None:
        """
        Updates the data entry of an item in the queue without changing its priority.
    
        Args:
            key: The key of the item to update.
            new_entry: New data entry for the item.

        Raises:
            KeyError: If the :code:`key` does not exist.
        """
        if key not in self._entry_map:
            raise KeyError(f"The key {key} does not exist.")
    
        self._entry_map[key][-1] = new_entry

    def remove(self, key: Hashable) -> None:
        """
        Marks an existing item as REMOVED.

        Args:
            key: The key of the item to be removed.

        Raises:
            KeyError: If the :code:`key` does not exist.
        """
        entry = self._entry_map.pop(key)
        entry[-1] = PriorityQueue.REMOVED

    def pop(self) -> Tuple[float, Hashable, Any]:
        """
        Removes and returns the lowest priority item.

        Returns:
            The priority, key, and data entry of the lowest priority item.
        
        Raises:
            KeyError: If the queue is empty.
        """
        while self._pq:
            priority, key, entry = heapq.heappop(self._pq)
            if entry is not PriorityQueue.REMOVED:
                self._len -= 1
                del self._entry_map[key]
                return priority, key, entry
        raise KeyError("Cannot pop from an empty priority queue.")

    def __len__(self) -> int:
        """
        Returns:
            Length of the priority queue.
        """
        return self._len
