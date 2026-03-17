"""Communication from Javascript/HTML to IPython/Jupyter lab using a local server and requests.
Initialization by user is not recommended. It should happen automatically when creating a network.Network.

Remember that you need AT LEAST ONE AVAILABLE (and sometimes also forwarded)
PORT BETWEEN min_port AND max_port IN ORDER FOR IT TO WORK."""

import http.server
import random
import string
import threading
from typing import Any, Callable
import logging
from time import sleep
import socket
import json


def random_word(k: int) -> str:
    """Generate a random word of length *k*.

    :param k: Length of the random word.
    :returns: A random string of ASCII letters.
    """
    return "".join(random.choices(string.ascii_letters, k=k))


enable_server: bool = True
"""Disable if you don't want to use an internal communication server. Some features might break."""

localhost_address: str = "127.0.0.1"

min_port = 8889
max_port = 8905
port_range = range(min_port, max_port)
"""The range of ports that stromvogel uses. They should all be forwarded if you're on an http tunnel."""

server_port: int = 8888
"""Global variable storing the port that is being used by this process. Changes when initialize_server is called."""

events: dict[str, Callable[[str], Any]] = {}
"""Dictionary that stores currently active events, along with their function, hashed by randomly generated ids."""

server_running: bool = False
"""Global variable that is set to true when the server is running."""

server: "CommunicationServer | None" = None
"""Global variable holding the server used for this notebook. None if not initialized."""


class CommunicationServer:
    """Run a web server in the background to receive Javascript communications.

    The server maintains a list of events, each with a unique id.
    The Javascript code sends a POST request to the server with the id and the data.
    The server then looks up the event with that id and calls the function associated with it.
    """

    def __init__(self, server_port: int = 8080) -> None:
        """Initialize the communication server.

        :param server_port: Port to run the server on.
        """
        import IPython.display as ipd

        self.server_port: int = server_port
        # Define a function within javascript that posts.
        js = """
function return_id_result(url, id, data) {
        fetch(url, {
            method: 'POST',
            body: JSON.stringify({
                'id': id,
                'data': data
            })
        })
    }
"""
        ipd.display(ipd.HTML(f"<script>{js}</script>"))
        ipd.display(ipd.Javascript(js))
        # These should both do the same thing, but just in case.

        # This inner class actually runs the server.
        class InnerServer(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                """Handle POST requests.

                Call the function associated with the id in the request body.
                The argument is passed as the body of the request.
                """
                content_length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(content_length).decode("utf-8"))
                id = body["id"]
                data = json.dumps(body["data"])
                logging.info(f"Received request: {id}\n{data}")
                f = events[id]
                f(data)

            def log_message(self, format, *args):
                # Overrides default log message
                pass

        self.web_server: http.server.HTTPServer = http.server.HTTPServer(
            (localhost_address, self.server_port), InnerServer
        )
        thr = threading.Thread(target=self.__run_server)
        thr.start()

    def __run_server(self):
        """Run the server on a background thread.

        Set the global variable ``server_running`` to ``True`` once started.
        This prevents making requests too early.
        """
        global server_running
        try:
            logging.info(
                f"CommunicationsServer started http://{localhost_address}:{self.server_port}"
            )
            server_running = True
            self.web_server.serve_forever()
        except KeyboardInterrupt:
            pass

    def add_event(self, js: str, function: Callable[[str], Any]) -> str:
        """Add an event using some JavaScript code.

        Within the *js* code, use the special function ``FUNCTION(...)`` to call
        the Python *function*.

        Example::

            js = "FUNCTION(37 + 42);"
            function = lambda data: print(data)

        The arithmetic is performed in Javascript, and the result is printed in Python.
        Note that the function is called with the result of the arithmetic as a string.

        :param js: JavaScript code containing a ``FUNCTION(...)`` call.
        :param function: Python callable invoked with the JavaScript result as a string.
        :returns: Event id which can be used to remove the event later.
        """
        import IPython.display as ipd

        global server_running, server
        if server is None:
            raise TimeoutError("There is no server running.")
        while not server_running:
            sleep(0.1)
            logging.debug("Waiting for server to finish booting up.")

        id = random_word(k=20)
        # Parse the RETURN.
        returning_js = js.replace(
            "FUNCTION(",
            f"return_id_result('http://127.0.0.1:{self.server_port}', '{id}', ",
        )
        ipd.display(ipd.Javascript(returning_js))
        events[id] = function
        return id

    def remove_event(self, event_id: str) -> Callable[[str], Any]:
        """Remove the event associated with the given event id.

        :param event_id: The id of the event to remove.
        :returns: The callable that was associated with the event.
        """
        return events.pop(event_id)

    def result(self, js: str, timeout_seconds: float = 2.0) -> str:
        """Execute some JavaScript and return the result.

        Use the special function ``RETURN(...)`` in *js* to send the result back.

        Example::

            js = "RETURN(37 + 42);"

        The arithmetic is executed in Javascript, and ``"79"`` is returned as a string.

        :param js: JavaScript code containing a ``RETURN(...)`` call.
        :param timeout_seconds: Seconds to wait for a result before raising.
        :returns: The result from JavaScript as a string.
        :raises TimeoutError: If no result is received within *timeout_seconds*.
        """
        # We implement this by using our add_event function, and waiting for the result to be set.
        result = None  # Variable that will contain the result when received.

        def on_result(data: str):  # Called when the javascript returns a result.
            nonlocal result
            result = data

        id = self.add_event(js.replace("RETURN", "FUNCTION"), on_result)
        passed_seconds = 0
        DELTA = 0.1
        while result is None and passed_seconds < timeout_seconds:
            sleep(DELTA)
            passed_seconds += DELTA

        self.remove_event(id)
        if result is None:
            raise TimeoutError(
                f"CommunicationServer.request did not receive result in time for request {id}:\n{js}"
            )
        else:
            return result


def __warn_request():
    print(
        "Test request failed. See 'Implementation details' in docs. Disable warning by use_server=False."
    )


def __warn_server():
    print(
        "Could not start server. See 'Implementation details' in docs. Disable warning by use_server=False."
    )


def __warn_no_free_port():
    print(
        f"""No free port [{min_port, max_port}). See 'Implementation details' in docs. Disable warning by use_server=False."""
    )


def is_port_free(port: int) -> bool:
    """Return ``True`` if the specified port is free on ``localhost_address``.

    :param port: Port number to check.
    :returns: ``True`` if the port is free, ``False`` otherwise.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((localhost_address, port)) != 0


def find_free_port() -> int:
    """Find a free port in the configured port range.

    :returns: A free port number, or ``-1`` if none are available.
    """
    for port_no in port_range:
        if is_port_free(port_no):
            return port_no
    return -1


def initialize_server() -> CommunicationServer | None:
    """Initialize or return the global communication server.

    If the server has not been created yet, create one on the first free port
    in the configured range and store it in the global ``server`` variable.
    If already initialized, return the existing server.

    :returns: The server instance, or ``None`` if initialization failed or is disabled.
    """
    import ipywidgets as widgets
    import IPython.display as ipd

    global server, server_port, enable_server
    if not enable_server:
        return None

    output = widgets.Output()
    with output:
        print(
            "Initializing communication server and sending a test message. This might take a couple of seconds."
        )
    ipd.display(output)
    try:
        if (
            server is None
        ):  # If the server is not initialized yet, try to initialize it.
            server_port = find_free_port()
            if server_port == -1:  # Could not find a free port.
                __warn_no_free_port()
                with output:
                    ipd.clear_output()
                return None
            server = CommunicationServer(server_port=server_port)
            logging.info("Succesfully initialized server.")
            try:
                server.result("RETURN('test message')")
                logging.info("Succesfully received test message.")
            except TimeoutError:  # Test request failed.
                __warn_request()
        with output:
            ipd.clear_output()
        return server
    except OSError:
        logging.warning("Server port likely taken.")
        __warn_server()
        with output:
            ipd.clear_output()
