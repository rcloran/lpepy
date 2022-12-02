import socket
import time


class WLEDControl:
    def __init__(self, leds, address, port):
        self._address = address
        self._port = port
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)

        self._num_leds = leds
        self.fill([0, 0, 0])

    def show(self):
        duration = 10
        msg = f"\x02{duration:c}" + "".join(
            [f"{r:c}{g:c}{b:c}" for r, g, b in self._pixels]
        )
        msg_bytes = msg.encode("iso8859")

        for i in range(5):
            # Problems with packets being lost somewhere, brute force it
            self._sock.sendto(msg_bytes, (self._address, self._port))
            time.sleep(0.01)

    def fill(self, colour):
        self._pixels = [colour] * self._num_leds

    def __getitem__(self, y):
        return self._pixels[y]

    def __setitem__(self, key, value):
        if isinstance(key, slice):
            start, stop, step = key.indices(self._num_leds)
            self._pixels[start:stop:step] = value[: stop - start]
            return

        self._pixels[key] = value

    def __len__(self):
        return len(self._pixels)
