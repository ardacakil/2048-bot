"""One computing slot plus a capped FIFO queue (asyncio side; call only from the event loop).

A ticket is either running (it holds the slot) or waiting. join() grants the slot at once if it
is free and nobody waits, raises QueueFull if `cap` tickets already wait, else queues the ticket.
release() hands the slot to the next waiting ticket; leave() removes a waiting ticket (client gone).
Every change wakes wait_change(), which queued streams use to report their new position.
"""

import asyncio
import collections
import time


class QueueFull(Exception):
    pass


class Ticket:
    def __init__(self, target):
        self.target = target
        self.created = time.monotonic()
        self.granted = asyncio.Event()
        self.granted_at = None
        self.closed = False


class ComputeSlot:
    def __init__(self, cap):
        self.cap = cap
        self.running = None
        self.waiting = collections.deque()
        self._changed = asyncio.Event()

    def join(self, target):
        if self.running is not None and len(self.waiting) >= self.cap:
            raise QueueFull
        ticket = Ticket(target)
        if self.running is None and not self.waiting:
            self._grant(ticket)
        else:
            self.waiting.append(ticket)
        self._notify()
        return ticket

    def position(self, ticket):
        """0 while running, 1 = next in line, ...; None once it left."""
        if ticket is self.running:
            return 0
        try:
            return self.waiting.index(ticket) + 1
        except ValueError:
            return None

    def ahead(self, ticket):
        """Tickets that will compute before this one (the running one first)."""
        pos = self.position(ticket)
        if not pos:
            return []
        return ([self.running] if self.running else []) + list(self.waiting)[:pos - 1]

    def leave(self, ticket):
        if ticket in self.waiting:
            self.waiting.remove(ticket)
            self._notify()

    def release(self, ticket):
        if ticket is not self.running:
            return
        self.running = None
        while self.waiting:
            nxt = self.waiting.popleft()
            if not nxt.closed:
                self._grant(nxt)
                break
        self._notify()

    async def wait_change(self, timeout):
        """True if the queue changed within `timeout` seconds."""
        event = self._changed
        try:
            await asyncio.wait_for(event.wait(), timeout)
            return True
        except asyncio.TimeoutError:
            return False

    def _grant(self, ticket):
        self.running = ticket
        ticket.granted_at = time.monotonic()
        ticket.granted.set()

    def _notify(self):
        self._changed.set()
        self._changed = asyncio.Event()
