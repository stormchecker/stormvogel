"""Communication from Javascript/HTML to IPython/Jupyter lab using a local server and requests.
Initialization by user is not recommended. It should happen automatically when creating a network.Network.

Remember that you need AT LEAST ONE AVAILABLE (and sometimes also forwarded)
PORT BETWEEN min_port AND max_port IN ORDER FOR IT TO WORK."""

import http.server
import random
import string
import threading
from typing import Callable
import logging
from time import sleep
import socket
import json


def random_word(k: int) -> str:
    """Random word of length k"""
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

events: dict[str, Callable] = {}
"""Dictionary that stores currently active events, along with their function, hashed by randomly generated ids."""

server_running: bool = False
"""Global variable that is set to true when the server is running."""

server: "CommunicationServer | None" = None
"""Global variable holding the server used for this notebook. None if not initialized."""


class CommunicationServer:
    def __init__(self, server_port: int = 8080) -> None:
        """Run a web server in the background to receive Javascript communications.
        It works by having a list of events, each with a unique id.
        The Javascript code sends a POST request to the server with the id and the data.
        The server then looks up the event with that id and calls the function associated with it.

        Args:
            server_port (int, optional): Defaults to 8080.
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

        class InnerServer(http.server.BaseHTTPRequestHandler):
            def do_POST(self):
                """Handle POST requests.
                Call the function associated with the id in the request body.
                The argument is passed as the body of the request."""
                content_length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(content_length).decode("utf-8"))
                id = body["id"]
                data = json.dumps(body["data"])
                logging.info(f"Received request: {id}\n{data}")
                f = events[id]
                f(data)

            def log_message(self, format, *args):
                """To prevent an unwanted default log message from coming up. Not used by stormvogel."""
                pass

        self.web_server: http.server.HTTPServer = http.server.HTTPServer(
            (localhost_address, self.server_port), InnerServer
        )
        thr = threading.Thread(target=self.__run_server)
        thr.start()

    def __run_server(self):
        """Run the server (used to put it on a thread).
        Waits 0.5 seconds and then sets global variable server_running to true.
        This is to prevent making requests too early."""
        global server_running
        try:
            logging.info(
                f"CommunicationsServer started http://{localhost_address}:{self.server_port}"
            )
            server_running = True
            self.web_server.serve_forever()
        except KeyboardInterrupt:
            pass

    def add_event(self, js: str, function: Callable) -> str:
        """Add an event using some JavaScript code.
        Within your js, use the special function FUNCTION(...) to call the Python function.

        Example:
            js = "FUNCTION(37 + 42);"
            function = lambda data: print(data)
            Then the arithmetic is performed in Javascript, and the result is printed in Python.
            Note that the function is called with the result of the arithmetic as a string.


        Returns event id which can be used to remove it later.
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
        # ipd.display(ipd.HTML(f"<script>{returning_js}</script>"))
        ipd.display(ipd.Javascript(returning_js))
        # print(returning_js)
        events[id] = function
        return id

    def remove_event(self, event_id: str) -> Callable:
        """Remove the event associated with this event id."""
        return events.pop(event_id)

    def result(self, js: str, timeout_seconds: float = 2.0) -> str:
        """Execute some JavaScript, then use the special function RETURN(...) to return the result.

        Example:
            js = "RETURN(37 + 42);"
            Then the function result returns "79" as a string.
        """
        result = None

        def on_result(data: str):
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
    """Return true iff the specified port is free on localhost_address.
    Thanks to StackOverflow user Rugnar
    https://stackoverflow.com/questions/2470971/fast-way-to-test-if-a-port-is-in-use-using-python"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((localhost_address, port)) != 0


def find_free_port() -> int:
    for port_no in port_range:
        if is_port_free(port_no):
            return port_no
    return -1


def initialize_server() -> CommunicationServer | None:
    """If server is None, then create a new server and store it in global variable server.
    Use the port stored in global variable server_port.

    Returns:
        CommunicationServer | None: The server if successful.
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
        if server is None:
            server_port = find_free_port()
            if server_port == -1:
                __warn_no_free_port()
                with output:
                    ipd.clear_output()
                return None
            server = CommunicationServer(server_port=server_port)
            logging.info("Succesfully initialized server.")
            try:
                server.result("RETURN('test message')")
                logging.info("Succesfully received test message.")
            except TimeoutError:
                __warn_request()
        with output:
            ipd.clear_output()
        return server
    except OSError:
        logging.warning("Server port likely taken.")
        __warn_server()
        with output:
            ipd.clear_output()
