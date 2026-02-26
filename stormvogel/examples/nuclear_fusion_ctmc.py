import stormvogel.model


def create_nuclear_fusion_ctmc():
    # Create a new model
    ctmc = stormvogel.model.new_ctmc()

    # hydrogen fuses into helium
    ctmc.states[0].set_choices([(3, ctmc.new_state("helium"))])

    # helium fuses into carbon
    ctmc.states[1].set_choices([(2, ctmc.new_state("carbon"))])

    # carbon fuses into iron
    ctmc.states[2].set_choices([(7, ctmc.new_state("iron"))])

    # supernova
    ctmc.states[3].set_choices([(12, ctmc.new_state("Supernova"))])

    # we add self loops to all states with no outgoing choices
    ctmc.add_self_loops()

    return ctmc


if __name__ == "__main__":
    # Print the resulting model in dot format.
    print(create_nuclear_fusion_ctmc().to_dot())
