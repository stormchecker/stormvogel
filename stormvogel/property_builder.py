import stormvogel.model
import ipywidgets as widgets
from stormvogel.dict_editor import DictEditor
import IPython.display as ipd

# & I think the property builder could use some love.
# * A lot of features are missing.
# * There is a bug if you choose compare, it will remove max and min
# * "Beta version. More complicated paths in the future." - maybe remove this line or implement more complicated paths.
# * Maybe we could add some colored text to highlight the different parts of the property.
# It might be easier to implement this for me since I was forced to understand the wierd quirks of ipywidgets.


def build_property_string(model: stormvogel.model.Model):
    """Lets the user build a property string using a widget"""

    schema = {
        "type of task": {
            "__collapse": True,
            "type": {
                "__html": "<p>Do you want to learn about probabilities or rewards?</p>",
                "__description": "Choose one",
                "__widget": "Dropdown",
                "__kwargs": {
                    "options": [
                        "probability",
                        "reward",
                    ]
                },
            },
            "task": {
                "__html": "<p>Do you want to compare values to check if a certain property holds or obtain a value?</p>",
                "__description": "Choose one",
                "__widget": "Dropdown",
                "__kwargs": {
                    "options": [
                        "obtain",
                        "compare",
                    ]
                },
            },
            "maxmin": {
                "__html": "<p>Since your model has actions your result depends on the scheduler. Do you want the maximum or minimum value?</p>",
                "__description": "Choose one",
                "__widget": "Dropdown",
                "__kwargs": {
                    "options": [
                        "max",
                        "min",
                    ]
                },
            },
            "operator": {
                "__html": "<p>If you chose 'compare', you should select a comparison operator.</p>",
                "__description": "Choose one",
                "__widget": "Dropdown",
                "__kwargs": {
                    "options": [
                        "<",
                        ">",
                        ">=",
                        "<=",
                        "=",
                    ]
                },
            },
            "value": {
                "__html": "<p>If you chose 'compare', you need to choose a value.</p>",
                "__description": "Choose one",
                "__widget": "FloatSlider",
            },
        },
        "path": {
            "__collapse": True,
            "path": {
                "__html": "<p>Beta version. More complicated paths in the future.</p>",
                "__description": "State:",
                "__widget": "Text",
            },
        },
    }

    class Values:
        """We need a stateful object so that we can refer to out if on_update is called"""

        x = 0
        values = {
            "type of task": {
                "type": "probability",
                "task": "obtain",
                "maxmin": "max",
                "operator": "=",
                "value": 0.00,
            },
            "path": {"path": "init"},
        }
        out = widgets.Output()

        def calculate_prop_string(self):
            """Calculate the property string based on the values attribute."""
            prop = ""
            if self.values["type of task"]["type"] == "reward":
                prop += "R"
            else:
                prop += "P"

            if self.values["type of task"]["task"] == "obtain":
                prop += (
                    f"{"max" if self.values["type of task"]["maxmin"] == "max" else "min"}=?"
                    if model.supports_actions()
                    else "=?"
                )
            else:
                prop += str(self.values["type of task"]["operator"])
                prop += str(self.values["type of task"]["value"])
            prop += f' [F "{self.values['path']['path']}"]'

            return prop

        def on_update(self):
            self.x = self.x + 1
            with self.out:
                ipd.clear_output()
                print(self.calculate_prop_string())

    v = Values()

    de = DictEditor(schema, v.values, v.on_update)
    de.show()

    ipd.display(v.out)


if (
    __name__ == "__main__"
):  # & I assume this was for testing, should probably be removed.
    import examples.monty_hall

    mdp = examples.monty_hall.create_monty_hall_mdp()

    build_property_string(mdp)
