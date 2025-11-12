import stormvogel.model


def build_property_string(model: stormvogel.model.Model):
    """Lets the user build a property string using a widget."""
    import IPython.display as ipd
    import ipywidgets as widgets
    from stormvogel.dict_editor import DictEditor

    # The first part of the property builder: choose P, R or S. The small 'r' stands for 'root' and is a quirk of the DictEditor.
    first_part_schema = {
        "r": {
            "__collapse": False,
            "type": {
                "__html": "<h4>Property builder</h4>",
                "__description": "Choose the property you are interested in:",
                "__widget": "ToggleButtons",
                "__kwargs": {"options": ["P", "R", "S"]},
            },
            "__html": """<ul>
            <li><b>P (probability). </b> The probability that a path property (LTL, CTL etc.) holds.</li>
            <li><b>R (rewards). </b> The reward of the model under some constraint.</li>
            <li><b>S (steady-state behaviour). </b> The probability that a property holds, while the amount of steps tends to infinity.</li>
            </ul>""",
        },
    }

    # Choose an optimization direction. Relevant if the model has actions.
    opt = {
        "__description": "The model has actions, choose an optimization direction:",
        "__widget": "ToggleButtons",
        "__kwargs": {"options": ["max", "min"]},
    }

    # Explanation for operands.
    operand_explanation = 'An operand may use either labels, or variables with bounds. For example: "init" means that the state has label "init", and x>=5 means that the value of variable x is at least 5.'

    # Explanation for time bounds.
    time_bound_explanation = 'The time bound is used for bounded properties. For example "<=10" means that the property must hold within 10 time steps. Leave empty for unbounded properties.'

    # Choose a path property. Relevant if P was chosen.
    path_prop_schema = {
        "__html": "<h4>Path property</h4>",
        "r": {
            "__collapse": False,
            "operator": {
                "__description": "Choose the operator",
                "__widget": "ToggleButtons",
                "__kwargs": {"options": ["F", "G", "X", "U"]},
            },
            "__html": f"""<ul>
                <li><b>F (eventually). </b> The probability that operand 1 will eventually hold.</li>
                <li><b>G (always). </b> The probability that operand 1 always holds.</li>
                <li><b>X (next). </b> The probability that operand 1 holds in the next step.</li>
                <li><b>U (until). </b> The probability that operand 1 holds, until operand 2 holds.</li>
                </ul>
                {operand_explanation}""",
            "operand1": {
                "__description": "Operand 1",
                "__widget": "Text",
            },
            "operand2": {
                "__description": "Operand 2",
                "__widget": "Text",
            },
            "time_bound": {
                "__html": time_bound_explanation,
                "__description": "Time bound",
                "__widget": "Text",
            },
        },
    }

    # Choose a reward property. Relevant if R was chosen.
    reward_prop_schema = {
        "__html": "<h4>Reward property</h4>",
        "r": {
            "__collapse": False,
            "operator": {
                "__description": "Choose the operator",
                "__widget": "ToggleButtons",
                "__kwargs": {"options": ["F", "C", "LRA"]},
            },
            "__html": f"""<ul>
                <li><b>F (eventually). </b> The reward that is accumulated until the operand holds.</li>
                <li><b>C (total, cummulative reward). </b> The total reward that is accumulated when the system runs forever.</li>
                <li><b>LRA (gain / mean payoff / long run average). </b> The average expected reward per step if the system runs forever.</li>
                </ul>
                {operand_explanation}""",
            "operand1": {
                "__description": "Operand",
                "__widget": "Text",
            },
            "time_bound": {
                "__html": time_bound_explanation,
                "__description": "Time bound",
                "__widget": "Text",
            },
        },
    }

    # Select a reward model. Relevant if R was chosen, and there are multiple reward models.
    reward_model = {
        "__html": "The model has multiple reward models, select one",
        "__description": "Rew. model",
        "__widget": "Dropdown",
        "__kwargs": {"options": [""], "width": "auto"},
    }

    # Select a steady-state property. Relevant if S was chosen.
    steady_prop_schema = {
        "r": {
            "__collapse": False,
            "operand": {
                "__html": f"<h4>Steady-state property</h4>{operand_explanation}",
                "__description": "Operand",
                "__widget": "Text",
            },
        }
    }

    def values_to_path_reward_property(v: dict) -> str:
        """Convert the dictionary of values to a property string for path or reward properties."""
        op1 = v["operand1"].replace(" ", "")
        if v["operator"] == "U":
            op2 = v["operand2"].replace(" ", "")
            return f"[{op1} U{v['time_bound']} {op2}]"
        elif v["operator"] == "LRA":
            return f"[{v['operator']}]"
        elif v["operator"] == "C":
            return f"[{v['operator']}{v['time_bound']}]"
        else:
            return f"[{v['operator']}{v['time_bound']} {op1}]"

    def values_to_first_part_property(v: dict, reward_model="") -> str:
        """Convert the dictionary of values to a property string for the first part."""
        type_ = v["r"]["type"]
        opt_ = ""
        if model.supports_actions():
            opt_ = v["r"]["opt"]
        return (
            type_ + opt_ + "=?"
            if reward_model == ""
            else type_ + '{"' + reward_model + '"}' + opt_ + "=?"
        )

    if model.supports_actions():
        first_part_schema["r"]["opt"] = opt
    if len(model.rewards) > 1:
        old_operator = reward_prop_schema["r"]
        new_operator = {"reward_model": reward_model}
        new_operator["reward_model"]["__kwargs"]["options"] = [
            x.name for x in model.rewards
        ]
        new_operator.update(old_operator)  # Necessary to insert in front of the dict.

        reward_prop_schema["r"] = new_operator

    first_part_output = widgets.Output()
    second_part_output = widgets.Output()
    property_string_output = widgets.Output()

    class Values:
        # Small 'r' stands for 'root'.
        first_part_values = {
            "r": {
                "type": "P",
                "opt": "max",
            }
        }

        path_prop_values = {
            "r": {
                "operator": "F",
                "operand1": '"init"',
                "operand2": "x>=5",
                "time_bound": "",
            }
        }
        reward_prop_values = {
            "r": {
                "reward_model": "",
                "operator": "F",
                "operand1": '"init"',
                "time_bound": "",
            }
        }

        steady_prop_values = {"r": {"operand": "x>=5"}}

        def on_update_first_part(self):
            with second_part_output:
                ipd.clear_output()
            type_ = self.first_part_values["r"]["type"]
            if type_ == "P":
                de_path_prop.show()
            if type_ == "R":
                de_reward_prop.show()
            if type_ == "S":
                de_steady_prop.show()

        def on_update_path_prop(self):
            with property_string_output:
                ipd.clear_output()
                first_part = values_to_first_part_property(self.first_part_values)
                second_part = values_to_path_reward_property(self.path_prop_values["r"])
                print(first_part + " " + second_part)

        def on_update_reward_prop(self):
            with property_string_output:
                ipd.clear_output()
                first_part = values_to_first_part_property(self.first_part_values)
                second_part = values_to_path_reward_property(
                    self.reward_prop_values["r"]
                )
                print(first_part + " " + second_part)

        def on_update_steady_prop(self):
            with property_string_output:
                ipd.clear_output()
                first_part = values_to_first_part_property(self.first_part_values)
                second_part = "[" + self.steady_prop_values["r"]["operand"] + "]"
                print(first_part + " " + second_part)

    v = Values()

    de_first_part = DictEditor(
        first_part_schema,
        v.first_part_values,
        v.on_update_first_part,
        do_display=False,
        output=first_part_output,
    )
    de_path_prop = DictEditor(
        path_prop_schema,
        v.path_prop_values,
        v.on_update_path_prop,
        do_display=False,
        output=second_part_output,
    )
    de_reward_prop = DictEditor(
        reward_prop_schema,
        v.reward_prop_values,
        v.on_update_reward_prop,
        do_display=False,
        output=second_part_output,
    )
    de_steady_prop = DictEditor(
        steady_prop_schema,
        v.steady_prop_values,
        v.on_update_steady_prop,
        do_display=False,
        output=second_part_output,
    )

    de_first_part.show()

    ipd.display(first_part_output)
    ipd.display(second_part_output)
    ipd.display(property_string_output)
