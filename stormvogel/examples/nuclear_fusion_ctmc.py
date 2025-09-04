import stormvogel.model


def create_nuclear_fusion_ctmc():
    # Create a new model with the name "Nuclear fusion"
    ctmc = stormvogel.model.new_ctmc()

    # hydrogen fuses into helium
    ctmc.get_state_by_id(0).set_choice([(3, ctmc.new_state("helium"))])

    # helium fuses into carbon
    ctmc.get_state_by_id(1).set_choice([(2, ctmc.new_state("carbon"))])

    # carbon fuses into iron
    ctmc.get_state_by_id(2).set_choice([(7, ctmc.new_state("iron"))])

    # supernova
    ctmc.get_state_by_id(3).set_choice([(12, ctmc.new_state("Supernova"))])

    # we add the rates which are equal to whats in the choices since the probabilities are all 1
    rates = [3, 2, 7, 12, 0]
    for i in range(5):
        ctmc.set_rate(ctmc.get_state_by_id(i), rates[i])

    # we add self loops to all states with no outgoing choices
    ctmc.add_self_loops()

    return ctmc


if __name__ == "__main__":
    # Print the resulting model in dot format.
    print(create_nuclear_fusion_ctmc().to_dot())
