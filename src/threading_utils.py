import sys
import threading
import selectors
import socket
import logging



class Barrier:
    """A class that allows external notification of a desire to proceed, and a cheap (sleeping) wait function until that notification comes."""

    def __init__(self):
        self._proceed_flag = False
        self._cv = threading.Condition(threading.Lock())

    def proceed(self):
        """Any thread can call the 'proceed' function, which will cause the wait() function to fall through."""
        with self._cv:
            self._proceed_flag = True
            self._cv.notify_all()

    def wait(self):
        with self._cv:
            while not self._proceed_flag:
                self._cv.wait()

class WaitConsoleThread(threading.Thread):
    """The WaitConsoleThread runs until console input is available (or it is asked to quit before)."""

    def __init__(self, quit_barrier):
        super().__init__(name="console")
        self._quit_barrier = quit_barrier
        self._socketpair = socket.socketpair()

    def close(self):
        for sock in self._socketpair:
            sock.close()

    def run(self):
        """Wait until stdin has input.

        The run method executes in its own thread.
        """
        selector = selectors.DefaultSelector()
        key_socketpair = selector.register(
            self._socketpair[0], selectors.EVENT_READ
        )
        key_stdin = selector.register(sys.stdin, selectors.EVENT_READ)
        logging.info("Console wait thread started.")
        quitflag = False
        while not quitflag:
            for (key, events) in selector.select():
                if key == key_socketpair:
                    quitflag = True
                elif key == key_stdin:
                    quitflag = True

        self._quit_barrier.proceed()

        logging.info("Console wait thread stopped.")

    def request_quit(self):
        """Called from the any thread to request that we quit."""
        self._socketpair[1].send(b"\x00")
