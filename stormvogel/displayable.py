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
        """Abstract class for displaying something.

        Args:
            output (widgets.Output, optional): Output window. Defaults to widgets.Output().
            do_display (bool, optional): Controls if it should display. Defaults to True.
            debug_output (widgets.Output, optional): Useful for debugging. Defaults to widgets.Output().
            spam (widgets.Output, optional): Used for side-effect displays. Defaults to widgets.Output().
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
