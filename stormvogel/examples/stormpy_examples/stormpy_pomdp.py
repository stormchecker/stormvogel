def example_building_pomdps_01():
    import stormpy
    import stormpy.info

    import stormpy.examples
    import stormpy.examples.files

    import stormpy.pomdp

    # Check support for parameters

    path = stormpy.examples.files.prism_pomdp_maze
    prism_program = stormpy.parse_prism_program(path)
    formula_str = 'P=? [!"bad" U "goal"]'
    properties = stormpy.parse_properties_for_prism_program(formula_str, prism_program)
    # construct the POMDP
    options = stormpy.BuilderOptions([p.raw_formula for p in properties])
    options.set_build_observation_valuations()
    options.set_build_choice_labels()
    pomdp = stormpy.build_sparse_model_with_options(prism_program, options)
    # make its representation canonic.
    pomdp = stormpy.pomdp.make_canonic(pomdp)

    return pomdp


if __name__ == "__main__":
    print(example_building_pomdps_01())
