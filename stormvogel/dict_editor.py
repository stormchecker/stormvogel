"""Generate an interactive editor from a specified schema using ipywidgets.
This is used to create the layout editing menu."""

import stormvogel.displayable
import stormvogel.rdict

import ipywidgets as widgets
import IPython.display as ipd
from typing import Any, Callable
import copy


class WidgetWrapper:
    """Creates a widget specified in the arguments."""

    convert_dict = {
        "IntSlider": widgets.IntSlider,
        "FloatSlider": widgets.FloatSlider,
        "ColorPicker": widgets.ColorPicker,
        "Checkbox": widgets.Checkbox,
        "Text": widgets.Text,
        "Dropdown": widgets.Dropdown,
        "TagsInput": widgets.TagsInput,
        "IntText": widgets.IntText,
    }

    def __init__(
        self,
        description: str,
        widget: str,
        path: list[str],
        initial_value: Any,
        update_dict: dict,
        on_update: Callable[[], None],
        **kwargs,
    ):
        """Creates a widget which automatically updates a dictonary value.

        Args:
            description (str): 'description' of the widget. Will be shown in the UI.
            widget (str): A str name for a widget.
            path (list[str]): path to the value in update_dict to be changed if the user changes this widget.
            initial_value (Any): initial value of the widget (aka. 'value')
            update_dict (dict): The dict that should be updated.
                When the user sets the value of the widget to X, the value at update_dict[path] is set to X.
            on_update (Callable[[], None]): A function that is called whenever a value is updated.
                When the user sets the value of the widget to something, on_update() is called.
        """
        self.update_dict: dict = update_dict
        self.path: list[str] = path
        self.on_update: Callable = on_update
        # The button is a special case with a slightly different API.
        if widget == "Button":
            self.widget = widgets.Button(description=description, **kwargs)
            self.widget.on_click(self.on_edit)
        else:
            w = self.convert_dict[widget]
            self.widget = widgets.interactive(
                self.on_edit,
                x=w(value=initial_value, description=description, **kwargs),
            )

    def on_edit(self, x: Any) -> None:
        """This method is called when the user sets the value of a widget to x.
        It updates self.update_dict[self.path], and calls the on_update function."""
        stormvogel.rdict.rset(self.update_dict, self.path, x)
        self.on_update()


class DictEditor(stormvogel.displayable.Displayable):
    """Create an interactive json editor from a schema using ipy widgets."""

    def __init__(
        self,
        schema: dict,
        update_dict: dict,
        on_update: Callable[[], None],
        output: widgets.Output | None = None,
        do_display: bool = True,
        debug_output: widgets.Output = widgets.Output(),
    ) -> None:
        """Create an interactive json editor from a schema using ipy widgets. Display it using the show() method.

        Args:
            schema (dict): The dict that specifies the schema.
                Quite closely follows the structure of the actual dict. Supports macros.
                Macros are defined as a dict with the key "__macros" and the value is the name of the macro.
                A macro can be used in the schema by using the key "__use_macro" and the value is the name of the macro.
                See layouts/schema.json for an example.
            update_dict (dict): The dict that will be updated.
            on_update (Callable[[], None]): Function that is called whenever the dict is updated.
            output (widgets.Output | None, optional): The output widget to use.
            do_display (bool, optional): Whether to display the widget immediately.
            debug_output (widgets.Output, optional): Output widget for debug information.
        """
        super().__init__(output, do_display, debug_output)
        self.on_update: Callable = on_update
        self.update_dict: dict = update_dict
        self.macros: dict = {}
        self.schema: dict = schema

    def show(self):
        with self.output:
            ipd.clear_output()
            ipd.display(self.recurse_create(self.schema, []))
        self.maybe_display_output()

    def recurse_create(self, sub_schema: dict, path: list) -> widgets.Widget:
        acc_items = []
        for k, v in sub_schema.items():
            new_path = copy.deepcopy(path)
            new_path.append(k)
            if k == "__html":
                acc_items.append(widgets.HTML(v))
            if k == "__macros":
                self.macros = v
            elif isinstance(v, dict):
                if "__html" in v and "__widget" in v:
                    acc_items.append(widgets.HTML(v["__html"]))
                if "__use_macro" in v:
                    macro_value = self.macros[v["__use_macro"]]
                    acc_items.append(self.recurse_create(macro_value, new_path))
                elif (
                    "__description" in v
                ):  # v is a widget, because it has a defined __description.
                    if "__kwargs" in v:  # Also pass arguments if relevant.
                        w = WidgetWrapper(
                            description=v["__description"],
                            widget=v["__widget"],
                            initial_value=stormvogel.rdict.rget(
                                self.update_dict, new_path
                            ),
                            path=new_path,
                            update_dict=self.update_dict,
                            on_update=self.on_update,
                            **v["__kwargs"],
                        )
                    else:
                        w = WidgetWrapper(
                            description=v["__description"],
                            widget=v["__widget"],
                            initial_value=stormvogel.rdict.rget(
                                self.update_dict, new_path
                            ),
                            path=new_path,
                            update_dict=self.update_dict,
                            on_update=self.on_update,
                        )
                    acc_items.append(w.widget)
                else:  # v is not a widget or macro, then call recursively.
                    acc_items.append(self.recurse_create(v, new_path))
        if "__collapse" in sub_schema and sub_schema["__collapse"]:
            acc = widgets.Accordion(
                children=[widgets.VBox(children=acc_items)], titles=[path[-1]]
            )
        else:
            acc = widgets.VBox(children=acc_items)
        return acc
