import ipywidgets as widgets
import IPython.display as ipd


class Displayable:
    """Abstract class for displaying something."""

    def __init__(
        self,
        output: widgets.Output | None = None,
        do_display: bool = True,
        debug_output: widgets.Output = widgets.Output(),
        spam: widgets.Output = widgets.Output(),
    ) -> None:
        """Initialize the displayable.

        :param output: Output window. If ``None``, a new ``widgets.Output()`` is created.
        :param do_display: Control whether output is displayed.
        :param debug_output: Output widget useful for debugging.
        :param spam: Output widget used for side-effect displays.
        """
        if output is None:
            self.output = widgets.Output()
        else:
            self.output = output
        self.do_display: bool = do_display
        self.debug_output: widgets.Output = debug_output
        self.spam = spam
        with self.output:
            ipd.display(self.spam)

    def maybe_display_output(self):
        """Display iff do_display is enabled."""
        if self.do_display:
            ipd.display(self.output)

    def spam_side_effects(self):
        """Display self.spam and clear its output immediately."""
        with self.output:
            ipd.display(self.spam)
        with self.spam:
            ipd.clear_output()
